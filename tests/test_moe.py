"""
Tests for tritforge.moe
=======================
Validates Layer 0 Asymmetric MoE assembly, Gatekeeper routing determinism,
Ternary GEMM zero-multiply mathematical equivalence, and empirical covariance collection.
"""

from pathlib import Path
import numpy as np
import pytest

from tritforge.moe import (
    Layer0AsymmetricMoE,
    MoERouter,
    SharedExpert,
    TernaryExpert,
    assemble_layer0_moe,
    collect_empirical_expert_covariances,
    rmsnorm,
    silu,
    ternary_gemm,
)
from tritforge.quantize import absmax_ternary_quantize

MASTER_DIR = Path(r"H:\AI_Memory_Vault\models\Qwen3.8-2.4T-A95B-FP8")


def test_rmsnorm():
    """Verify RMSNorm normalization and weight scaling."""
    rng = np.random.RandomState(42)
    x = rng.randn(4, 128).astype(np.float32)
    w = np.ones(128, dtype=np.float32) * 2.0
    normed = rmsnorm(x, w, zero_centered=False)
    assert normed.shape == (4, 128)
    # RMS of normalized output should be close to 2.0
    rms = np.sqrt(np.mean(np.square(normed), axis=-1))
    assert np.allclose(rms, 2.0, rtol=1e-3)


def test_silu():
    """Verify SiLU mathematical properties."""
    x = np.array([-10.0, 0.0, 10.0], dtype=np.float32)
    y = silu(x)
    assert y[0] < 0.0
    assert y[1] == 0.0
    assert np.isclose(y[2], 10.0, rtol=1e-3)


def test_moe_router_routing():
    """Verify router top-k selection, probability normalization, and determinism."""
    rng = np.random.RandomState(42)
    b_dim, hidden_size, num_experts, top_k = 4, 256, 32, 4
    gate_weight = rng.randn(num_experts, hidden_size).astype(np.float32) * 0.01

    router = MoERouter(gate_weight, num_experts=num_experts, num_experts_per_tok=top_k)
    x = rng.randn(b_dim, hidden_size).astype(np.float32)

    topk_indices, topk_probs = router.forward(x, top_k=top_k)

    assert topk_indices.shape == (b_dim, top_k)
    assert topk_probs.shape == (b_dim, top_k)
    # Probabilities must sum to 1.0 for each token
    prob_sums = np.sum(topk_probs, axis=-1)
    assert np.allclose(prob_sums, 1.0, atol=1e-5)
    # Expert IDs must be in valid range
    assert (topk_indices >= 0).all() and (topk_indices < num_experts).all()

    # Determinism check: identical inputs yield identical outputs
    idx2, probs2 = router.forward(x, top_k=top_k)
    assert np.array_equal(topk_indices, idx2)
    assert np.allclose(topk_probs, probs2)


def test_ternary_gemm_mathematical_identity():
    """Verify zero-multiply ternary GEMM is mathematically identical to float GEMM."""
    rng = np.random.RandomState(42)
    b_dim, m_dim, n_dim = 4, 128, 512
    block_size = 64

    x = rng.randn(b_dim, n_dim).astype(np.float32)
    w = rng.randn(m_dim, n_dim).astype(np.float32) * 0.02

    # Block-wise quantization
    qmat = absmax_ternary_quantize(w, scaling_mode="block", block_size=block_size)
    w_hat = qmat.reconstruct()

    # Float GEMM reference
    y_ref = x @ w_hat.T

    # Zero-multiply Ternary GEMM
    y_ternary = ternary_gemm(x, qmat, block_size=block_size)

    assert y_ternary.shape == (b_dim, m_dim)
    diff = np.max(np.abs(y_ref - y_ternary))
    assert diff < 1e-5, f"Ternary GEMM drift exceeded tolerance: {diff}"


def test_ternary_expert_swiglu():
    """Verify TernaryExpert SwiGLU forward pass."""
    rng = np.random.RandomState(42)
    b_dim, hidden_size, inter_size = 2, 256, 64
    block_size = 32

    w_gate = rng.randn(inter_size, hidden_size).astype(np.float32) * 0.01
    w_up = rng.randn(inter_size, hidden_size).astype(np.float32) * 0.01
    w_down = rng.randn(hidden_size, inter_size).astype(np.float32) * 0.01

    q_gate = absmax_ternary_quantize(w_gate, scaling_mode="block", block_size=block_size)
    q_up = absmax_ternary_quantize(w_up, scaling_mode="block", block_size=block_size)
    q_down = absmax_ternary_quantize(w_down, scaling_mode="block", block_size=block_size)

    expert = TernaryExpert(
        expert_idx=0,
        gate_proj=q_gate,
        up_proj=q_up,
        down_proj=q_down,
    )

    x = rng.randn(b_dim, hidden_size).astype(np.float32)
    out = expert.forward(x)

    assert out.shape == (b_dim, hidden_size)
    assert np.isfinite(out).all()


def test_asymmetric_moe_forward_pipeline():
    """Verify complete asymmetric Layer 0 MoE block forward pass."""
    rng = np.random.RandomState(42)
    b_dim, hidden_size, inter_size, num_experts, top_k = 4, 128, 64, 8, 2
    block_size = 32

    gate_w = rng.randn(num_experts, hidden_size).astype(np.float32) * 0.01
    router = MoERouter(gate_w, num_experts=num_experts, num_experts_per_tok=top_k)

    experts = {}
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

    # Shared expert in high precision
    shared = SharedExpert(
        gate_proj=rng.randn(inter_size, hidden_size).astype(np.float32) * 0.01,
        up_proj=rng.randn(inter_size, hidden_size).astype(np.float32) * 0.01,
        down_proj=rng.randn(hidden_size, inter_size).astype(np.float32) * 0.01,
    )

    norm_w = np.ones(hidden_size, dtype=np.float32)

    layer = Layer0AsymmetricMoE(
        router=router,
        experts=experts,
        shared_expert=shared,
        post_attention_layernorm_weight=norm_w,
        num_experts=num_experts,
        num_experts_per_tok=top_k,
    )

    x = rng.randn(b_dim, hidden_size).astype(np.float32)
    out, info = layer.forward(x, residual=True, return_routing_info=True)

    assert out.shape == (b_dim, hidden_size)
    assert np.isfinite(out).all()
    assert "topk_indices" in info
    assert "topk_probs" in info
    assert len(info["active_experts"]) > 0


def test_collect_empirical_expert_covariances():
    """Verify empirical covariance accumulation from router activations."""
    rng = np.random.RandomState(42)
    b_dim, hidden_size, num_experts, top_k = 16, 64, 8, 2
    gate_w = rng.randn(num_experts, hidden_size).astype(np.float32) * 0.01
    router = MoERouter(gate_w, num_experts=num_experts, num_experts_per_tok=top_k)

    activations = rng.randn(b_dim, hidden_size).astype(np.float32)
    covs = collect_empirical_expert_covariances(router, activations, target_experts=[0, 1], top_k=top_k)

    assert 0 in covs and 1 in covs
    for e_id, H_e in covs.items():
        assert H_e.shape == (hidden_size, hidden_size)
        # Check symmetry
        assert np.allclose(H_e, H_e.T, atol=1e-5)
        # Check positive definiteness via eigenvalues
        eigvals = np.linalg.eigvalsh(H_e)
        assert (eigvals > 0.0).all()


@pytest.mark.skipif(not (MASTER_DIR / "model-00001-of-00213.safetensors").exists(), reason="Master shard missing")
@pytest.mark.vault
def test_live_assemble_layer0_moe():
    """Verify live assembly of Layer 0 MoE for Expert 0 from master shards."""
    moe_layer = assemble_layer0_moe(
        base_dir=MASTER_DIR,
        expert_indices=[0],
        calibrate_with_real_tokens=True,
        sample_token_ids=[101, 102, 103, 104],
    )
    assert moe_layer.router.gate_weight.shape == (512, 8192)
    assert 0 in moe_layer.experts
    assert moe_layer.shared_expert is not None



def test_zero_centered_rmsnorm_convention():
    """C-1: Verify that zero-centered RMSNorm computes y = x_hat * (1.0 + weight)."""
    dim = 64
    rng = np.random.RandomState(42)
    x = rng.randn(2, dim).astype(np.float32)
    w = (rng.randn(dim) * 0.1).astype(np.float32)

    # 1. Zero-centered (default for decoder norms)
    out_zc = rmsnorm(x, w, zero_centered=True)
    var = np.mean(x ** 2, axis=-1, keepdims=True)
    expected_zc = (x / np.sqrt(var + 1e-6)) * (1.0 + w)
    np.testing.assert_allclose(out_zc, expected_zc, rtol=1e-5, atol=1e-6)

    # 2. Non-zero centered (standard)
    out_std = rmsnorm(x, w, zero_centered=False)
    expected_std = (x / np.sqrt(var + 1e-6)) * w
    np.testing.assert_allclose(out_std, expected_std, rtol=1e-5, atol=1e-6)

    # 3. Test against PyTorch / transformers Qwen3_5MoeRMSNorm if available
    try:
        import torch
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeRMSNorm

        ref_norm = Qwen3_5MoeRMSNorm(dim=dim, eps=1e-6)
        ref_norm.weight.data = torch.from_numpy(w).float()
        with torch.no_grad():
            ref_out = ref_norm(torch.from_numpy(x).float()).numpy()
        np.testing.assert_allclose(out_zc, ref_out, rtol=1e-5, atol=1e-6)
    except ImportError:
        pass
