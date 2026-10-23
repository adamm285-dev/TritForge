"""
Tests for tritforge.benchmark
=============================
Validates synthetic activation generation, evaluation metrics pipeline, and logging.
"""

import numpy as np
import pytest

from tritforge.benchmark import (
    evaluate_quantization,
    generate_synthetic_activations,
)
from tritforge.quantize import absmax_ternary_quantize


def test_generate_synthetic_activations():
    """Verify synthetic activation generation matches expected shape and scale."""
    n_samples = 32
    hidden_size = 256
    X = generate_synthetic_activations(n_samples=n_samples, hidden_size=hidden_size, seed=99)
    assert X.shape == (n_samples, hidden_size)
    assert X.dtype == np.float32

    # Check that norm is scaled approximately to sqrt(hidden_size)
    row_norms = np.linalg.norm(X, axis=1)
    assert np.allclose(row_norms, np.sqrt(hidden_size), rtol=1e-3)


def test_evaluate_quantization_pipeline():
    """Verify that evaluate_quantization produces valid metrics container."""
    m, n = 256, 1024
    rng = np.random.RandomState(42)
    W = rng.randn(m, n).astype(np.float32) * 0.02
    X = generate_synthetic_activations(n_samples=16, hidden_size=n, seed=42)
    H = (X.T @ X) / 16 + 0.01 * np.eye(n, dtype=np.float32)
    Y_gt = X @ W.T

    qmat = absmax_ternary_quantize(W, scaling_mode="block", block_size=128)
    w_hat = qmat.reconstruct()

    meta = {"layer_idx": 0, "expert_idx": 0, "proj": "test_proj"}

    # Missing held-out evaluation data must raise ValueError (B5)
    with pytest.raises(ValueError, match="Held-out evaluation data"):
        evaluate_quantization(
            W_fp8=W,
            qmat=qmat,
            W_hat=w_hat,
            X=X,
            H=H,
            Y_ground_truth=Y_gt,
            elapsed_s=0.05,
            meta=meta,
        )

    # Disjoint held-out dataset (I-6)
    X_ho = generate_synthetic_activations(n_samples=16, hidden_size=n, seed=999)
    Y_ho = X_ho @ W.T

    result = evaluate_quantization(
        W_fp8=W,
        qmat=qmat,
        W_hat=w_hat,
        X=X,
        H=H,
        Y_ground_truth=Y_gt,
        elapsed_s=0.05,
        meta=meta,
        X_held_out=X_ho,
        Y_held_out=Y_ho,
    )

    assert result.layer_idx == 0
    assert result.expert_idx == 0
    assert result.proj_type == "test_proj"
    assert 0.0 < result.rel_frobenius_err < 1.0
    assert 0.0 < result.weight_cosine_sim <= 1.0
    assert 0.0 < result.output_cosine_sim <= 1.0
    assert 0.0 < result.held_out_drift < 1.0
    assert 0.0 < result.held_out_cosine <= 1.0
    assert result.compression_ratio_base3 > 70.0


def test_benchmark_held_out_disjointness_and_metrics_computation_i6():
    """I-6: Verify that held-out tokens are disjoint from calibration tokens and held_out metrics differ from in-sample."""
    m, n = 128, 512
    rng = np.random.RandomState(42)
    W = rng.randn(m, n).astype(np.float32) * 0.02

    # In-sample tokens (seed 100) vs Held-out tokens (seed 200)
    X_cal = generate_synthetic_activations(n_samples=16, hidden_size=n, seed=100)
    X_ho = generate_synthetic_activations(n_samples=16, hidden_size=n, seed=200)

    # Assert token sets are strictly disjoint
    assert not np.allclose(X_cal, X_ho)
    min_dist = np.min(np.linalg.norm(X_cal[:, None, :] - X_ho[None, :, :], axis=-1))
    assert min_dist > 1.0, f"Token sets overlap or are too close: min_dist={min_dist}"

    H = (X_cal.T @ X_cal) / 16 + 0.01 * np.eye(n, dtype=np.float32)
    Y_cal = X_cal @ W.T
    Y_ho = X_ho @ W.T

    qmat = absmax_ternary_quantize(W, scaling_mode="block", block_size=128)
    w_hat = qmat.reconstruct()

    meta = {"layer_idx": 0, "expert_idx": 0, "proj": "test_proj"}
    result = evaluate_quantization(
        W_fp8=W,
        qmat=qmat,
        W_hat=w_hat,
        X=X_cal,
        H=H,
        Y_ground_truth=Y_cal,
        elapsed_s=0.05,
        meta=meta,
        X_held_out=X_ho,
        Y_held_out=Y_ho,
    )

    # In-sample drift and held-out drift are independently computed and not identical
    assert result.rel_output_drift != result.held_out_drift
    assert result.output_cosine_sim != result.held_out_cosine
