"""
Tests for tritforge.quantize
============================
Validates ternary kernels, Absmax/Mean-Abs scaling, second-order Hessian error
compensation, and lossless 2-bit & Base-3 bit-packing.
"""

import numpy as np
import pytest

from tritforge.quantize import (
    QuantizedTritMatrix,
    absmax_ternary_quantize,
    compute_cosine_similarity,
    compute_frobenius_error,
    compute_hessian_error,
    dequantize_trits,
    hessian_compensate_quantize,
    pack_trits_2bit,
    pack_trits_base3,
    unpack_trits_2bit,
    unpack_trits_base3,
)


@pytest.fixture
def sample_matrix():
    """Generates a reproducible synthetic weight matrix."""
    rng = np.random.RandomState(42)
    return rng.randn(128, 512).astype(np.float32) * 0.02


def test_absmax_ternary_quantize_invariants(sample_matrix):
    """Verify that all trits strictly belong to {-1, 0, +1}."""
    for mode in ["channel", "block", "tensor"]:
        for method in ["absmax", "mean_abs", "optimal"]:
            qmat = absmax_ternary_quantize(sample_matrix, scaling_mode=mode, scale_method=method)
            unique = set(np.unique(qmat.trits))
            assert unique.issubset({-1, 0, 1}), f"Failed on mode={mode}, method={method}: {unique}"
            assert qmat.original_shape == sample_matrix.shape
            assert 0.0 < qmat.density < 1.0


def test_dequantization_reconstruction(sample_matrix):
    """Verify that dequantized matrix matches expected shape and scale."""
    qmat = absmax_ternary_quantize(sample_matrix, scaling_mode="block", block_size=64)
    w_hat = qmat.reconstruct()
    assert w_hat.shape == sample_matrix.shape
    assert w_hat.dtype == np.float32
    # Verify non-trivial reconstruction
    assert not np.allclose(w_hat, 0.0)


def test_frobenius_and_cosine_metrics(sample_matrix):
    """Verify mathematical properties of Frobenius and Cosine similarity functions."""
    qmat = absmax_ternary_quantize(sample_matrix, scaling_mode="channel")
    w_hat = qmat.reconstruct()
    abs_err, rel_err = compute_frobenius_error(sample_matrix, w_hat)
    cos_sim = compute_cosine_similarity(sample_matrix, w_hat)

    assert abs_err > 0.0
    assert 0.0 < rel_err < 1.0
    assert 0.5 < cos_sim <= 1.0

    # Self-comparison should yield zero error and 1.0 cosine similarity
    zero_err, zero_rel = compute_frobenius_error(sample_matrix, sample_matrix)
    assert zero_err == 0.0
    assert zero_rel == 0.0
    assert np.isclose(compute_cosine_similarity(sample_matrix, sample_matrix), 1.0)


def test_hessian_compensation_error_reduction(sample_matrix):
    """Verify that second-order Hessian compensation strictly reduces Hessian error."""
    m, n = sample_matrix.shape
    rng = np.random.RandomState(42)
    X = rng.randn(64, n).astype(np.float32)
    H = (X.T @ X) / 64 + 0.01 * np.eye(n, dtype=np.float32)

    # 1. Naive Block Quantization
    q_naive = absmax_ternary_quantize(sample_matrix, scaling_mode="block", scale_method="absmax", block_size=64)
    w_naive = q_naive.reconstruct()
    loss_naive = compute_hessian_error(sample_matrix, w_naive, H)

    # 2. Hessian Compensated Quantization
    q_comp, w_comp = hessian_compensate_quantize(sample_matrix, H, block_size=64, scale_method="optimal", damp=0.01)
    loss_comp = compute_hessian_error(sample_matrix, w_comp, H)

    assert loss_comp < loss_naive, f"Hessian loss was not reduced: naive={loss_naive}, comp={loss_comp}"
    reduction = (1.0 - (loss_comp / loss_naive)) * 100.0
    assert reduction > 30.0, f"Expected >30% error reduction, got {reduction:.2f}%"


def test_bit_packing_2bit_roundtrip():
    """Verify lossless roundtrip of 2-bit packing (4 trits/byte)."""
    rng = np.random.RandomState(123)
    shape = (64, 128)
    trits = rng.choice([-1, 0, 1], size=shape).astype(np.int8)

    packed = pack_trits_2bit(trits)
    # Expected packed size: ceil(64 * 128 / 4) = 2048 bytes
    assert packed.nbytes == (64 * 128) // 4

    unpacked = unpack_trits_2bit(packed, shape)
    assert np.array_equal(trits, unpacked), "2-bit packing roundtrip failed"


def test_bit_packing_base3_roundtrip():
    """Verify lossless roundtrip of Base-3 packing (5 trits/byte, 1.6 bits/trit)."""
    rng = np.random.RandomState(456)
    shape = (50, 100)
    trits = rng.choice([-1, 0, 1], size=shape).astype(np.int8)

    packed = pack_trits_base3(trits)
    # Expected packed size: ceil(50 * 100 / 5) = 1000 bytes (20% of original int8)
    assert packed.nbytes == (50 * 100) // 5

    unpacked = unpack_trits_base3(packed, shape)
    assert np.array_equal(trits, unpacked), "Base-3 packing roundtrip failed"
