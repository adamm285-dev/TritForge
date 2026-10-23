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
    assert metrics.noise_reduction_ratio == 0.80  # (10 - 2) / 10 = 80% noise trimmed!
    assert metrics.mean_topk_mass >= 0.90
