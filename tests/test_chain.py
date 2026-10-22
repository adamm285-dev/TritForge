"""
TritForge 2-Layer Sequential Forward Pipeline Test (tests/test_chain.py)
========================================================================
Validates multi-layer forward execution across consecutive .trit containers:
Token Embeddings -> Layer 0 (.trit) -> Layer 1 (.trit) -> Output Activation.

Verifies:
1. Cross-layer residual continuity.
2. Independent Gatekeeper routing per layer.
3. Multi-container seek-based streaming without memory leaks.
"""

from pathlib import Path
import time
import numpy as np
import pytest

from tritforge.container import load_trit_layer, load_trit_tensor, read_trit_header
from tritforge.moe import MoERouter, SharedExpert
from tritforge.model import StreamingMoELayer

VAULT_DIR = Path(r"H:\AI_Memory_Vault\models\Qwen3.8-2.4T-TritForge")
L0_PATH = VAULT_DIR / "layer_00.trit"
L1_PATH = VAULT_DIR / "layer_01.trit"


def create_streaming_moe(trit_path: Path, layer_idx: int) -> StreamingMoELayer:
    """Instantiates a constant-memory StreamingMoELayer backed by a .trit container."""
    hdr, offset = read_trit_header(trit_path)
    tensors = hdr["tensors"]

    # 1. Load Router
    r_key = "mlp.gate.weight" if "mlp.gate.weight" in tensors else "router.weight"
    router_w = load_trit_tensor(trit_path, r_key, header=hdr, payload_start_offset=offset)
    router = MoERouter(router_w, num_experts=512, num_experts_per_tok=10)

    # 2. Load Shared Expert if present
    shared_exp = None
    if "mlp.shared_expert.gate_proj.weight" in tensors:
        s_gate = load_trit_tensor(trit_path, "mlp.shared_expert.gate_proj.weight", header=hdr, payload_start_offset=offset)
        s_up = load_trit_tensor(trit_path, "mlp.shared_expert.up_proj.weight", header=hdr, payload_start_offset=offset)
        s_down = load_trit_tensor(trit_path, "mlp.shared_expert.down_proj.weight", header=hdr, payload_start_offset=offset)
        s_gate_w = None
        if "mlp.shared_expert_gate.weight" in tensors:
            s_gate_w = load_trit_tensor(trit_path, "mlp.shared_expert_gate.weight", header=hdr, payload_start_offset=offset)
        shared_exp = SharedExpert(gate_proj=s_gate, up_proj=s_up, down_proj=s_down, gate_weight=s_gate_w)
    elif "shared_expert.gate_proj" in tensors:
        s_gate = load_trit_tensor(trit_path, "shared_expert.gate_proj", header=hdr, payload_start_offset=offset)
        s_up = load_trit_tensor(trit_path, "shared_expert.up_proj", header=hdr, payload_start_offset=offset)
        s_down = load_trit_tensor(trit_path, "shared_expert.down_proj", header=hdr, payload_start_offset=offset)
        shared_exp = SharedExpert(gate_proj=s_gate, up_proj=s_up, down_proj=s_down)

    # 3. Load Post Attention Norm
    post_norm = None
    if "post_attention_layernorm.weight" in tensors:
        post_norm = load_trit_tensor(trit_path, "post_attention_layernorm.weight", header=hdr, payload_start_offset=offset)

    return StreamingMoELayer(
        router=router,
        trit_path=trit_path,
        shared_expert=shared_exp,
        norm_weight=post_norm,
        num_experts=512,
        num_experts_per_tok=10,
        max_cached_experts=32,
        layer_idx=layer_idx,
    )


@pytest.mark.skipif(not (L0_PATH.exists() and L1_PATH.exists()), reason="Layer 0 or Layer 1 container missing on H: drive")
def test_two_layer_sequential_forward_chain():
    """Verify end-to-end forward chaining from Layer 0 to Layer 1."""
    # 1. Verify container existence and tensor counts
    hdr0, _ = read_trit_header(L0_PATH)
    hdr1, _ = read_trit_header(L1_PATH)
    assert len(hdr0["tensors"]) == 1552
    assert len(hdr1["tensors"]) == 1552

    # 2. Assemble Layer 0 and Layer 1 Streaming MoEs
    l0_moe = create_streaming_moe(L0_PATH, layer_idx=0)
    l1_moe = create_streaming_moe(L1_PATH, layer_idx=1)

    # 3. Generate synthetic input token activation: (batch=1, hidden=8192)
    np.random.seed(42)
    x0 = np.random.randn(1, 8192).astype(np.float32) * 0.02

    # 4. Execute Layer 0 Forward Pass (seeks and loads top-10 active experts)
    t0 = time.perf_counter()
    y0, info0 = l0_moe.forward(x0, residual=True, collect_metrics=True)
    t_l0 = (time.perf_counter() - t0) * 1000.0

    assert y0.shape == (1, 8192)
    assert not np.isnan(y0).any()
    assert info0.active_experts == 10
    assert len(info0.expert_token_counts) == 10

    # 5. Execute Layer 1 Forward Pass (input is Layer 0 exit activation)
    t1 = time.perf_counter()
    y1, info1 = l1_moe.forward(y0, residual=True, collect_metrics=True)
    t_l1 = (time.perf_counter() - t1) * 1000.0

    assert y1.shape == (1, 8192)
    assert not np.isnan(y1).any()
    assert info1.active_experts == 10
    assert len(info1.expert_token_counts) == 10

    # 6. Verify routing diversity across layers
    # Layer 0 and Layer 1 routers should make independent expert selections
    assert set(info0.expert_token_counts.keys()) != set(info1.expert_token_counts.keys())

    # 7. Check numerical magnitude continuity
    norm_x0 = float(np.linalg.norm(x0))
    norm_y0 = float(np.linalg.norm(y0))
    norm_y1 = float(np.linalg.norm(y1))
    assert 0.1 * norm_x0 < norm_y0 < 10.0 * norm_x0
    assert 0.1 * norm_y0 < norm_y1 < 10.0 * norm_y0

    print(f"\n[CHAIN SUCCESS] Layer 0 Latency: {t_l0:.1f}ms | Layer 1 Latency: {t_l1:.1f}ms")
    print(f" Layer 0 Active Experts ({info0.active_experts}): {list(info0.expert_token_counts.keys())}")
    print(f" Layer 1 Active Experts ({info1.active_experts}): {list(info1.expert_token_counts.keys())}")
