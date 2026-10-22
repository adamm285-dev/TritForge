"""
Tests for tritforge.kernel
==========================
Validates Zero-Multiply GEMM engine, sparse zero-skipping traversal,
mathematical identity with reference floating-point GEMM, and throughput benchmarking.
"""

import numpy as np
import pytest

from tritforge.kernel import (
    SparseTritIndex,
    ZeroMultiplyGEMMEngine,
    benchmark_gemm_throughput,
    build_sparse_trit_index,
    zero_multiply_sparse_gemm,
    zero_multiply_vectorized_gemm,
)
from tritforge.quantize import absmax_ternary_quantize


def test_sparse_trit_index_construction():
    """Verify CSR index construction omits zero-valued weights entirely."""
    # Create explicit test matrix with known entries
    # Row 0: [+1,  0, -1,  0]
    # Row 1: [ 0,  0,  0,  0]
    # Row 2: [-1, -1, +1, +1]
    trits = np.array([
        [ 1,  0, -1,  0],
        [ 0,  0,  0,  0],
        [-1, -1,  1,  1],
    ], dtype=np.int8)

    sparse_idx = build_sparse_trit_index(trits)

    assert sparse_idx.shape == (3, 4)
    # Row 0: 1 pos (col 0), 1 neg (col 2)
    assert sparse_idx.pos_row_ptrs[1] - sparse_idx.pos_row_ptrs[0] == 1
    assert sparse_idx.neg_row_ptrs[1] - sparse_idx.neg_row_ptrs[0] == 1
    # Row 1: 0 pos, 0 neg
    assert sparse_idx.pos_row_ptrs[2] - sparse_idx.pos_row_ptrs[1] == 0
    assert sparse_idx.neg_row_ptrs[2] - sparse_idx.neg_row_ptrs[1] == 0
    # Row 2: 2 pos (cols 2, 3), 2 neg (cols 0, 1)
    assert sparse_idx.pos_row_ptrs[3] - sparse_idx.pos_row_ptrs[2] == 2
    assert sparse_idx.neg_row_ptrs[3] - sparse_idx.neg_row_ptrs[2] == 2

    # Total non-zero entries = 2 + 0 + 4 = 6 out of 12 elements (density = 0.5)
    assert sparse_idx.nnz_total == 6
    assert sparse_idx.density == 0.5


def test_zero_multiply_sparse_gemm_exactness():
    """Verify sparse zero-skipping GEMM matches standard matrix multiplication across scaling modes."""
    rng = np.random.RandomState(42)
    b, m, n = 3, 8, 32
    block_size = 16
    x = rng.randn(b, n).astype(np.float32)
    w = rng.randn(m, n).astype(np.float32) * 0.02

    for mode in ["channel", "tensor", "block"]:
        qmat = absmax_ternary_quantize(w, scaling_mode=mode, block_size=block_size)
        w_hat = qmat.reconstruct()

        # Float GEMM reference
        y_ref = x @ w_hat.T

        # Sparse Zero-Multiply GEMM
        engine = ZeroMultiplyGEMMEngine(qmat, execution_mode="sparse", block_size=block_size)
        y_sparse = engine(x)

        assert y_sparse.shape == (b, m)
        diff = np.max(np.abs(y_ref - y_sparse))
        assert diff < 1e-5, f"Sparse GEMM drift in mode {mode} exceeded tolerance: {diff}"


def test_zero_multiply_vectorized_gemm_scaling_modes():
    """Verify vectorized zero-multiply GEMM across block, channel, and tensor scaling modes."""
    rng = np.random.RandomState(42)
    b, m, n = 4, 32, 128
    x = rng.randn(b, n).astype(np.float32)
    w = rng.randn(m, n).astype(np.float32) * 0.02

    for mode in ["block", "channel", "tensor"]:
        qmat = absmax_ternary_quantize(w, scaling_mode=mode, block_size=32)
        w_hat = qmat.reconstruct()
        y_ref = x @ w_hat.T

        engine = ZeroMultiplyGEMMEngine(qmat, execution_mode="vectorized", block_size=32)
        y_out = engine(x)

        assert y_out.shape == (b, m)
        diff = np.max(np.abs(y_ref - y_out))
        assert diff < 1e-5, f"Vectorized GEMM drift in mode {mode}: {diff}"


def test_benchmark_gemm_throughput_execution():
    """Verify benchmark harness executes and reports valid metrics."""
    res = benchmark_gemm_throughput(matrix_shape=(128, 512), batch_size=2, block_size=64, warmup=1, runs=2)

    assert "fp32_latency_ms" in res
    assert "vectorized_latency_ms" in res
    assert "vectorized_gops" in res
    assert "vectorized_speedup" in res
    assert res["vectorized_gops"] > 0.0
    assert res["cosine_sim_vs_ref"] > 0.999
    assert res["max_drift_vs_ref"] < 1e-4
