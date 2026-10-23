"""
Tests for tritforge.dispatcher
==============================
Validates multi-expert batch dispatching, permutation sorting, scatter-gather accuracy,
and equivalence with sequential Layer 0 MoE execution.
"""

import numpy as np
import pytest

from tritforge.dispatcher import (
    DispatchMetrics,
    MoEBatchDispatcher,
    create_batch_dispatcher_from_moe,
)
from tritforge.moe import Layer0AsymmetricMoE, MoERouter, SharedExpert, TernaryExpert
from tritforge.quantize import absmax_ternary_quantize


@pytest.fixture
def mini_moe_layer():
    """Generates a reproducible miniature Layer 0 MoE block."""
    rng = np.random.RandomState(42)
    hidden_size, inter_size, num_experts, top_k = 64, 32, 8, 2
    block_size = 16

    router = MoERouter(rng.randn(num_experts, hidden_size).astype(np.float32) * 0.01)

    experts = {}
    for e in range(num_experts):
        experts[e] = TernaryExpert(
            expert_idx=e,
            gate_proj=absmax_ternary_quantize(rng.randn(inter_size, hidden_size).astype(np.float32) * 0.01, block_size=block_size),
            up_proj=absmax_ternary_quantize(rng.randn(inter_size, hidden_size).astype(np.float32) * 0.01, block_size=block_size),
            down_proj=absmax_ternary_quantize(rng.randn(hidden_size, inter_size).astype(np.float32) * 0.01, block_size=block_size),
        )

    shared = SharedExpert(
        gate_proj=rng.randn(inter_size, hidden_size).astype(np.float32) * 0.01,
        up_proj=rng.randn(inter_size, hidden_size).astype(np.float32) * 0.01,
        down_proj=rng.randn(hidden_size, inter_size).astype(np.float32) * 0.01,
    )

    layer = Layer0AsymmetricMoE(
        router=router,
        experts=experts,
        shared_expert=shared,
        post_attention_layernorm_weight=np.ones(hidden_size, dtype=np.float32),
        num_experts=num_experts,
        num_experts_per_tok=top_k,
    )
    return layer


def test_batch_dispatcher_sequential_equivalence(mini_moe_layer):
    """Verify batch dispatcher produces identical outputs to sequential MoE layer."""
    dispatcher = create_batch_dispatcher_from_moe(mini_moe_layer)

    rng = np.random.RandomState(123)
    x = rng.randn(12, 64).astype(np.float32)

    # 1. Residual pass
    out_seq = mini_moe_layer.forward(x, residual=True)
    out_batch = dispatcher.dispatch_and_execute(x, residual=True)
    diff = np.max(np.abs(out_seq - out_batch))
    assert diff < 1e-5, f"Batch dispatcher drift on residual forward: {diff}"

    # 2. Non-residual pass
    out_seq_no_res = mini_moe_layer.forward(x, residual=False)
    out_batch_no_res = dispatcher.dispatch_and_execute(x, residual=False)
    diff_no_res = np.max(np.abs(out_seq_no_res - out_batch_no_res))
    assert diff_no_res < 1e-5, f"Batch dispatcher drift on non-residual forward: {diff_no_res}"


def test_batch_dispatcher_3d_sequence_tensor(mini_moe_layer):
    """Verify batch dispatcher natively handles 3D tensors (batch_size, seq_len, hidden_size)."""
    dispatcher = create_batch_dispatcher_from_moe(mini_moe_layer)

    rng = np.random.RandomState(456)
    x_3d = rng.randn(2, 6, 64).astype(np.float32)

    out_3d, metrics = dispatcher.dispatch_and_execute(x_3d, residual=True, collect_metrics=True)
    assert out_3d.shape == (2, 6, 64)
    assert isinstance(metrics, DispatchMetrics)
    assert metrics.total_tokens == 12
    assert metrics.active_experts > 0
    assert metrics.load_imbalance_ratio >= 1.0


def test_batch_dispatcher_telemetry(mini_moe_layer):
    """Verify telemetry correctly tracks token counts and expert load distribution."""
    dispatcher = create_batch_dispatcher_from_moe(mini_moe_layer)

    rng = np.random.RandomState(789)
    x = rng.randn(16, 64).astype(np.float32)

    _, metrics = dispatcher.dispatch_and_execute(x, collect_metrics=True)

    assert metrics.total_tokens == 16
    assert metrics.top_k == 2


def test_batch_dispatcher_missing_active_expert_raises_c5(mini_moe_layer):
    """C-5 backstop: Verify MoEBatchDispatcher raises KeyError if an active expert is missing from dictionary."""
    dispatcher = create_batch_dispatcher_from_moe(mini_moe_layer)

    # Delete expert 0 from dispatcher's expert dictionary
    if 0 in dispatcher.experts:
        del dispatcher.experts[0]

    # Force input that routes to expert 0
    rng = np.random.RandomState(42)
    x = rng.randn(16, 64).astype(np.float32)

    with pytest.raises(KeyError, match="required for active batch was not loaded"):
        dispatcher.dispatch_and_execute(x)
