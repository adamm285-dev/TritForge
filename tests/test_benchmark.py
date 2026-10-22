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
    result = evaluate_quantization(
        W_fp8=W,
        qmat=qmat,
        W_hat=w_hat,
        X=X,
        H=H,
        Y_ground_truth=Y_gt,
        elapsed_s=0.05,
        meta=meta,
    )

    assert result.layer_idx == 0
    assert result.expert_idx == 0
    assert result.proj_type == "test_proj"
    assert 0.0 < result.rel_frobenius_err < 1.0
    assert 0.0 < result.weight_cosine_sim <= 1.0
    assert 0.0 < result.output_cosine_sim <= 1.0
    assert result.compression_ratio_base3 > 70.0
