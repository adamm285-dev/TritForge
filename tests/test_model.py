"""
Tests for tritforge.model and tritforge.eval (Phase 4)
======================================================
Validates full model assembly, GQA Attention with KV caching, Asymmetric Transformer
blocks, streaming .trit container execution, autoregressive generation loop,
hardware footprint audit, and empirical generation fidelity.
"""

from pathlib import Path
import tempfile
import numpy as np
import pytest

from tritforge.container import save_trit_layer
from tritforge.eval import (
    GenerationFidelityEvaluator,
    ModelFootprintAuditor,
    log_phase4_evals_to_disk,
)
from tritforge.model import (
    AsymmetricAttention,
    AsymmetricTransformerBlock,
    GenerationResult,
    StreamingMoELayer,
    TritForgeConfig,
    TritForgeModel,
    create_dummy_asymmetric_model,
)
from tritforge.moe import MoERouter, SharedExpert, TernaryExpert
from tritforge.quantize import absmax_ternary_quantize


def test_tritforge_config():
    """Verify configuration dimensions, GQA ratios, and serialization."""
    cfg = TritForgeConfig(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
    )
    assert cfg.head_dim == 32
    assert cfg.kv_dim == 64
    d = cfg.to_dict()
    assert d["hidden_size"] == 256
    cfg2 = TritForgeConfig.from_dict(d)
    assert cfg2.hidden_size == 256
    assert cfg2.head_dim == 32


def test_asymmetric_attention_and_kv_cache():
    """Verify Grouped-Query Attention forward pass, causal masking, and KV cache update."""
    rng = np.random.RandomState(42)
    b_sz, seq_len = 2, 4
    hidden_size = 128
    num_heads, num_kv_heads = 4, 2
    head_dim = hidden_size // num_heads  # 32
    kv_dim = num_kv_heads * head_dim  # 64

    cfg = TritForgeConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
    )

    q_proj = rng.randn(hidden_size, hidden_size).astype(np.float32) * 0.02
    k_proj = rng.randn(kv_dim, hidden_size).astype(np.float32) * 0.02
    v_proj = rng.randn(kv_dim, hidden_size).astype(np.float32) * 0.02
    o_proj = rng.randn(hidden_size, hidden_size).astype(np.float32) * 0.02

    attn = AsymmetricAttention(cfg, q_proj, k_proj, v_proj, o_proj)

    x = rng.randn(b_sz, seq_len, hidden_size).astype(np.float32)

    # Initial forward pass without cache
    out1, kv1 = attn.forward(x, causal_mask=True, kv_cache=None)
    assert out1.shape == (b_sz, seq_len, hidden_size)
    assert np.isfinite(out1).all()
    assert kv1[0].shape == (b_sz, num_kv_heads, seq_len, head_dim)
    assert kv1[1].shape == (b_sz, num_kv_heads, seq_len, head_dim)

    # Incremental single-token forward pass with past cache
    x_next = rng.randn(b_sz, 1, hidden_size).astype(np.float32)
    out2, kv2 = attn.forward(x_next, causal_mask=True, kv_cache=kv1)
    assert out2.shape == (b_sz, 1, hidden_size)
    assert kv2[0].shape == (b_sz, num_kv_heads, seq_len + 1, head_dim)
    assert kv2[1].shape == (b_sz, num_kv_heads, seq_len + 1, head_dim)


def test_full_model_forward():
    """Verify end-to-end multi-layer forward pass and output shape."""
    model = create_dummy_asymmetric_model(
        vocab_size=200,
        hidden_size=64,
        intermediate_size=32,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        num_experts=4,
        num_experts_per_tok=2,
        block_size=16,
    )

    input_ids = np.array([[10, 20, 30], [5, 15, 25]], dtype=np.int64)
    logits, kv_caches, telemetries = model.forward(input_ids, collect_metrics=True)

    assert logits.shape == (2, 3, 200)
    assert np.isfinite(logits).all()
    assert len(kv_caches) == 2
    assert len(telemetries) == 2
    assert telemetries[0] is not None
    assert "active_experts" in telemetries[0]


def test_model_greedy_generation_determinism():
    """Verify deterministic greedy decoding produce identical sequences for identical inputs."""
    model = create_dummy_asymmetric_model(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=32,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        num_experts=4,
        num_experts_per_tok=2,
        block_size=16,
    )

    prompt = [12, 34, 56]
    gen1 = model.generate(prompt, max_new_tokens=5, temperature=0.0)
    gen2 = model.generate(prompt, max_new_tokens=5, temperature=0.0)

    assert gen1.sequences.shape == (1, 3 + 5)
    assert np.array_equal(gen1.sequences, gen2.sequences)
    assert gen1.generated_tokens == gen2.generated_tokens
    assert len(gen1.per_step_latencies_ms) == 5
    assert gen1.tokens_per_second > 0


def test_model_sampling_diversity():
    """Verify temperature and top-p sampling produce valid token sequences."""
    model = create_dummy_asymmetric_model(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=32,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        num_experts=4,
        num_experts_per_tok=2,
        block_size=16,
    )

    prompt = [1, 2, 3]
    gen_stochastic = model.generate(
        prompt,
        max_new_tokens=6,
        temperature=0.8,
        top_p=0.9,
        seed=123,
    )
    assert gen_stochastic.sequences.shape == (1, 9)
    assert len(gen_stochastic.generated_tokens[0]) == 6
    assert all(0 <= tok < 100 for tok in gen_stochastic.generated_tokens[0])


def test_streaming_moe_layer_with_trit_container():
    """Verify StreamingMoELayer loads active experts on-the-fly from a .trit container file."""
    rng = np.random.RandomState(42)
    b_sz, hidden_size, inter_size, num_experts, top_k = 2, 64, 32, 8, 2
    block_size = 16

    gate_w = rng.randn(num_experts, hidden_size).astype(np.float32) * 0.01
    router = MoERouter(gate_w, num_experts=num_experts, num_experts_per_tok=top_k)

    shared = SharedExpert(
        gate_proj=rng.randn(inter_size, hidden_size).astype(np.float32) * 0.01,
        up_proj=rng.randn(inter_size, hidden_size).astype(np.float32) * 0.01,
        down_proj=rng.randn(hidden_size, inter_size).astype(np.float32) * 0.01,
    )

    experts: dict[int, TernaryExpert] = {}
    for e_id in range(num_experts):
        w_g = rng.randn(inter_size, hidden_size).astype(np.float32) * 0.01
        w_u = rng.randn(inter_size, hidden_size).astype(np.float32) * 0.01
        w_d = rng.randn(hidden_size, inter_size).astype(np.float32) * 0.01
        experts[e_id] = TernaryExpert(
            expert_idx=e_id,
            gate_proj=absmax_ternary_quantize(w_g, scaling_mode="block", block_size=block_size),
            up_proj=absmax_ternary_quantize(w_u, scaling_mode="block", block_size=block_size),
            down_proj=absmax_ternary_quantize(w_d, scaling_mode="block", block_size=block_size),
        )

    # Save to a temporary .trit container file
    with tempfile.TemporaryDirectory() as tmp_dir:
        trit_path = Path(tmp_dir) / "test_layer_stream.trit"
        save_trit_layer(
            trit_path,
            router_weight=gate_w,
            experts=experts,
            shared_expert=shared,
            layer_idx=0,
            packing_mode="base3",
            scales_dtype="float16",
        )
        assert trit_path.exists()

        # Instantiate StreamingMoELayer with ZERO resident experts initially
        stream_layer = StreamingMoELayer(
            router=router,
            trit_path=trit_path,
            experts=None,  # Not in memory
            shared_expert=shared,
            num_experts=num_experts,
            num_experts_per_tok=top_k,
            max_cached_experts=4,  # Small cache to test eviction
        )
        assert len(stream_layer.experts) == 0

        x = rng.randn(b_sz, hidden_size).astype(np.float32)
        out, metrics = stream_layer.forward(x, residual=True, collect_metrics=True)

        assert out.shape == (b_sz, hidden_size)
        assert np.isfinite(out).all()
        # Active experts were streamed and cached
        assert len(stream_layer.experts) > 0
        assert len(stream_layer.experts) <= 4


def test_model_footprint_auditor_reduction():
    """Verify theoretical and container footprint audit guarantees >= 75% savings."""
    report = ModelFootprintAuditor.audit_qwen_2_4t()

    assert report.total_parameters > 2.0e12  # ~2.39T parameters
    assert report.routed_ratio_pct > 95.0   # Routed experts dominate architecture
    assert report.memory_reduction_vs_fp8_pct >= 75.0  # Must exceed 75% savings
    assert report.tritforge_base3_bytes < 550 * (1024 ** 3)  # Must be < 550 GB
    assert report.tritforge_spark_nodes <= 2  # Slashing hardware from 24 to 1-2 Sparks
    assert report.streaming_workstation_ram_gb <= 256.0  # Single workstation RAM ceiling


def test_generation_fidelity_evaluator():
    """Verify GenerationFidelityEvaluator computes trajectory cosine and agreement metrics."""
    m_ref = create_dummy_asymmetric_model(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=32,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        num_experts=4,
        num_experts_per_tok=2,
        block_size=16,
        seed=10,
    )

    m_quant = create_dummy_asymmetric_model(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=32,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        num_experts=4,
        num_experts_per_tok=2,
        block_size=16,
        seed=10,
    )

    prompt = [5, 10, 15]
    metrics = GenerationFidelityEvaluator.evaluate(
        reference_model=m_ref,
        quantized_model=m_quant,
        prompt_token_ids=prompt,
        max_new_tokens=4,
        temperature=0.0,
    )

    assert metrics.total_tokens_evaluated == 4
    # Because both models have identical weights here, metrics should be 1.0
    assert np.isclose(metrics.mean_cosine_similarity, 1.0, atol=1e-5)
    assert metrics.top1_agreement_rate == 1.0
    assert metrics.trajectory_blend_score >= 0.90



def test_streaming_moe_cache_does_not_evict_active_batch_experts():
    """C-5: Verify StreamingMoELayer does not drop active experts when active > cache size."""
    import tempfile
    from tritforge.container import StreamingTritWriter, load_trit_layer
    from tritforge.quantize import absmax_ternary_quantize
    from tritforge.moe import MoERouter, TernaryExpert

    hid, inter, n_exp, k = 64, 32, 64, 8
    rng = np.random.RandomState(42)

    with tempfile.TemporaryDirectory() as tmp_dir:
        trit_path = Path(tmp_dir) / "layer_test.trit"
        w = StreamingTritWriter(trit_path, layer_idx=0)
        experts_mem = {}
        for e in range(n_exp):
            q_g = absmax_ternary_quantize(rng.randn(inter, hid).astype(np.float32), block_size=16)
            q_u = absmax_ternary_quantize(rng.randn(inter, hid).astype(np.float32), block_size=16)
            q_d = absmax_ternary_quantize(rng.randn(hid, inter).astype(np.float32), block_size=16)
            w.add_quantized_matrix(f"experts.{e}.gate_proj", q_g)
            w.add_quantized_matrix(f"experts.{e}.up_proj", q_u)
            w.add_quantized_matrix(f"experts.{e}.down_proj", q_d)
            experts_mem[e] = TernaryExpert(e, q_g, q_u, q_d)
        w.close()

        gate_w = rng.randn(n_exp, hid).astype(np.float32)
        router = MoERouter(gate_weight=gate_w, num_experts=n_exp, num_experts_per_tok=k)

        # 16 tokens x top-8 activates most of the 64 experts
        x = rng.randn(16, hid).astype(np.float32)

        # Cache size of 8 (far smaller than ~40-60 active experts)
        # Pre-load reference experts directly from container so FP16 scale quantization matches exactly
        experts_from_container = {e: load_trit_layer(trit_path, expert_idx=e) for e in range(n_exp)}

        stream_layer = StreamingMoELayer(router, trit_path=trit_path, num_experts=n_exp, num_experts_per_tok=k, max_cached_experts=8)
        full_layer = StreamingMoELayer(router, experts=experts_from_container, num_experts=n_exp, num_experts_per_tok=k, max_cached_experts=1000)

        y_stream, info_stream = stream_layer.forward(x, residual=False, collect_metrics=True)
        y_full = full_layer.forward(x, residual=False)

        # Reconstructed outputs must match in-memory ground truth without dropped experts
        np.testing.assert_allclose(y_stream, y_full, rtol=1e-4, atol=1e-5)


def test_streaming_moe_zero_centered_rmsnorm_runtime_c1():
    """C-1 runtime test: verify StreamingMoELayer with zero-centered norm matches reference and fails if reverted to standard norm."""
    dim = 64
    n_exp = 32
    k = 4
    rng = np.random.RandomState(42)
    x = rng.randn(2, dim).astype(np.float32)
    w_norm = (rng.randn(dim) * 0.1).astype(np.float32)
    gate_w = rng.randn(n_exp, dim).astype(np.float32)
    router = MoERouter(gate_w, num_experts=n_exp, num_experts_per_tok=k)

    # 1. Independent reference top-k computation using x_hat * (1 + w)
    var = np.mean(x ** 2, axis=-1, keepdims=True)
    x_norm_ref = (x / np.sqrt(var + 1e-6)) * (1.0 + w_norm)
    logits_ref = np.dot(x_norm_ref, gate_w.T)
    expected_top_k = np.argsort(-logits_ref, axis=-1)[:, :k]

    # Standard (non-zero centered) top-k to prove separation
    x_norm_bad = (x / np.sqrt(var + 1e-6)) * w_norm
    logits_bad = np.dot(x_norm_bad, gate_w.T)
    bad_top_k = np.argsort(-logits_bad, axis=-1)[:, :k]
    assert not np.array_equal(expected_top_k, bad_top_k), "Test requires top-k to differ between conventions"

    # 2. Test against transformers Qwen3_5MoeRMSNorm if available
    try:
        import torch
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeRMSNorm
        ref_layer = Qwen3_5MoeRMSNorm(dim=dim, eps=1e-6)
        ref_layer.weight.data = torch.from_numpy(w_norm).float()
        with torch.no_grad():
            x_norm_hf = ref_layer(torch.from_numpy(x).float()).numpy()
        np.testing.assert_allclose(x_norm_ref, x_norm_hf, rtol=1e-5, atol=1e-6)
    except ImportError:
        pass

    # 3. Create dummy expert to avoid forward pass error
    q_dummy = absmax_ternary_quantize(rng.randn(16, dim).astype(np.float32), block_size=16)
    q_down = absmax_ternary_quantize(rng.randn(dim, 16).astype(np.float32), block_size=16)
    experts = {e: TernaryExpert(e, q_dummy, q_dummy, q_down) for e in range(n_exp)}

    layer = StreamingMoELayer(
        router=router,
        experts=experts,
        norm_weight=w_norm,
        num_experts=n_exp,
        num_experts_per_tok=k,
    )

    # 4. Execute forward pass
    out, metrics = layer.forward(x, residual=False, collect_metrics=True)
    assert metrics.total_tokens == 2
    assert metrics.top_k == k

    # Active experts in DispatchMetrics must match the expected top-k set exactly
    expected_active_experts = set(expected_top_k.flatten())
    actual_active_experts = set(metrics.expert_token_counts.keys())
    assert actual_active_experts == expected_active_experts

    # And must NOT match the wrong (standard norm) expert set
    bad_active_experts = set(bad_top_k.flatten())
    assert actual_active_experts != bad_active_experts
