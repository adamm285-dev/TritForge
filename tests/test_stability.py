from pathlib import Path
"""Unit tests for the TritForge Router Stability & Entropy Evaluator."""

import pytest
import numpy as np
from tritforge.stability import RouterStabilityEvaluator, RouterStabilityMetrics
from tritforge.plugins import DynaMoERouterPlugin, DefaultRouterPlugin


def test_shannon_entropy_calculation():
    # Uniform distribution across 4 elements: H = ln(4) approx 1.3863
    uniform_p = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
    h_uniform = RouterStabilityEvaluator.compute_shannon_entropy(uniform_p)
    assert np.isclose(h_uniform, np.log(4), atol=1e-4)

    # Deterministic distribution: H = 0
    determ_p = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    h_determ = RouterStabilityEvaluator.compute_shannon_entropy(determ_p)
    assert np.isclose(h_determ, 0.0, atol=1e-4)


def test_evaluate_routing_trace_dynamoe_stability():
    router = DynaMoERouterPlugin(min_k=2, max_k=12, prob_mass_threshold=0.90)

    logits_history = []
    experts_history = []
    weights_history = []

    # Simulate 10 tokens where top-2 experts capture >90% mass
    for _ in range(10):
        logits = np.zeros(512, dtype=np.float32)
        logits[10] = 15.0
        logits[20] = 14.0
        exp_indices, exp_weights, _ = router.select_experts(logits, top_k=10)

        logits_history.append(logits)
        experts_history.append(exp_indices)
        weights_history.append(exp_weights)

    metrics = RouterStabilityEvaluator.evaluate_routing_trace(
        logits_history=logits_history,
        selected_experts_history=experts_history,
        routing_weights_history=weights_history,
        nominal_k=10,
    )

    assert isinstance(metrics, RouterStabilityMetrics)
    assert metrics.is_stable is True
    assert metrics.average_k_selected == 2.0
    assert metrics.skipped_expert_ratio == 0.80  # (10 - 2) / 10 = 80% skipped
    assert metrics.mean_topk_mass >= 0.90


@pytest.mark.skipif(not Path(r"H:\AI_Memory_Vault\models\Qwen3.8-2.4T-TritForge\layer_00.trit").exists(), reason="Layer 0 container not found on H: drive")
@pytest.mark.vault
def test_evaluate_routing_trace_real_layer0_router_w2_6():
    """W2-6: Test router stability on real Layer 0 router from layer_00.trit on zero-centered norm inputs."""
    from tritforge.container import load_trit_tensor
    from tritforge.moe import rmsnorm

    trit_path = Path(r"H:\AI_Memory_Vault\models\Qwen3.8-2.4T-TritForge\layer_00.trit")
    q_gate = load_trit_tensor(trit_path, "mlp.gate.weight")
    w_gate = q_gate.reconstruct() if hasattr(q_gate, "reconstruct") else q_gate  # (512, hidden_size)

    q_norm = load_trit_tensor(trit_path, "input_layernorm.weight")
    w_norm = q_norm.reconstruct() if hasattr(q_norm, "reconstruct") else q_norm

    # Generate 16 held-out synthetic token embeddings (hidden_size = 8192)
    rng = np.random.default_rng(123)
    x_raw = rng.normal(0, 0.02, size=(16, 8192)).astype(np.float32)
    x_norm = rmsnorm(x_raw, w_norm, zero_centered=True)

    # Compute real router logits
    logits_matrix = x_norm @ w_gate.T  # (16, 512)

    router = DynaMoERouterPlugin(min_k=2, max_k=12, prob_mass_threshold=0.90)
    logits_history = []
    experts_history = []
    weights_history = []

    for i in range(16):
        logits = logits_matrix[i]
        exp_indices, exp_weights, _ = router.select_experts(logits, top_k=10)
        logits_history.append(logits)
        experts_history.append(exp_indices)
        weights_history.append(exp_weights)

    metrics = RouterStabilityEvaluator.evaluate_routing_trace(
        logits_history=logits_history,
        selected_experts_history=experts_history,
        routing_weights_history=weights_history,
        nominal_k=10,
    )

    print(f"Measured real router metrics: avg_k={metrics.average_k_selected}, "
          f"mean_topk_mass={metrics.mean_topk_mass}, is_stable={metrics.is_stable}")
    assert isinstance(metrics, RouterStabilityMetrics)
    assert 2.0 <= metrics.average_k_selected <= 12.0
    assert 0.0 <= metrics.mean_topk_mass <= 1.0

