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


def test_benchmark_pass_fail_uses_held_out_drift_d8a(tmp_path):
    """D8a: Benchmark PASS/FAIL must be determined by held_out_drift (< 0.45), not rel_output_drift.
    When in-sample drift passes (< 0.45) but held-out drift fails (>= 0.45), logged status must be FAIL.
    Fails if benchmark status reverts to rel_output_drift."""
    from tritforge.benchmark import BenchmarkRunResult, log_benchmark_to_evals

    evals_file = tmp_path / "evals.md"
    evals_file.write_text("# Test Evals\n", encoding="utf-8")

    # Construct result where in-sample rel_output_drift passes (< 0.45)
    # but held_out_drift fails (>= 0.45)
    res = BenchmarkRunResult(
        timestamp="2026-10-23 12:00:00",
        layer_idx=0,
        expert_idx=0,
        proj_type="gate_proj",
        matrix_shape=(128, 128),
        method_name="TritForge Hessian",
        scale_method="optimal",
        scaling_mode="block",
        rel_frobenius_err=0.20,
        weight_cosine_sim=0.98,
        hessian_loss=0.01,
        rel_output_drift=0.30,   # In-sample: PASSes threshold (< 0.45)
        output_cosine_sim=0.97,
        held_out_drift=0.55,     # Held-out: FAILS threshold (>= 0.45)
        held_out_cosine=0.92,
        output_mae=0.05,
        density=0.66,
        fp8_mb=1.0,
        packed_2bit_mb=0.3,
        packed_base3_mb=0.2,
        compression_ratio_base3=80.0,
        elapsed_time_s=0.05,
    )

    log_benchmark_to_evals([res], loss_reduction=50.0, evals_path=evals_file)

    logged_content = evals_file.read_text(encoding="utf-8")
    assert "| FAIL |" in logged_content, (
        f"Expected '| FAIL |' because held_out_drift (0.55) >= 0.45, "
        f"even though rel_output_drift (0.30) < 0.45.\nContent:\n{logged_content}"
    )
    assert "| PASS |" not in logged_content

