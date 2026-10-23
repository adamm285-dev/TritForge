from pathlib import Path
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


def test_quantized_trit_matrix_invalid_trits_raises_l8():
    """L-8: Verify that instantiating QuantizedTritMatrix containing trits not in {-1, 0, +1} raises ValueError."""
    scales = np.ones((2, 2), dtype=np.float32)

    # Valid trits pass
    valid_trits = np.array([[-1, 0], [1, 0]], dtype=np.int8)
    QuantizedTritMatrix(valid_trits, scales, scaling_mode="block", original_shape=(2, 2))

    # Invalid: +2
    invalid_pos = np.array([[-1, 2], [1, 0]], dtype=np.int8)
    with pytest.raises(ValueError, match="violat.*ternary invariant"):
        QuantizedTritMatrix(invalid_pos, scales, scaling_mode="block", original_shape=(2, 2))

    # Invalid: -2
    invalid_neg = np.array([[-2, 0], [1, 0]], dtype=np.int8)
    with pytest.raises(ValueError, match="violat.*ternary invariant"):
        QuantizedTritMatrix(invalid_neg, scales, scaling_mode="block", original_shape=(2, 2))


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



def test_ls_optimal_quantization_heavy_tail_optimality():
    """C-8 / Q-1: Verify that closed-form least-squares optimal ternary matches brute-force k."""
    bs = 128
    rng = np.random.RandomState(42)
    # Heavy-tailed Student-t distribution (df=2)
    w = rng.standard_t(df=2, size=bs).astype(np.float32)

    # 1. Closed-form via absmax_ternary_quantize with scale_method='ls_optimal'
    W_2d = w.reshape(1, bs)
    q_opt = absmax_ternary_quantize(W_2d, scaling_mode="block", scale_method="ls_optimal", block_size=bs)
    w_rec = q_opt.reconstruct().ravel()
    err_closed = float(np.sum((w - w_rec) ** 2))

    # 2. Brute-force over all possible k in 1..bs
    a = np.abs(w)
    s = -np.sort(-a)
    best_bf_err = float("inf")
    for k in range(1, bs + 1):
        thr = s[k - 1]
        t = np.where(a >= thr, np.sign(w), 0).astype(np.float32)
        n_nz = np.sum(t != 0)
        alpha = float(np.sum(w * t) / n_nz) if n_nz > 0 else 0.0
        err = float(np.sum((w - alpha * t) ** 2))
        if err < best_bf_err:
            best_bf_err = err

    # Assert closed-form matches optimal brute force within numerical precision (FP16 scale tolerance)
    assert np.isclose(err_closed, best_bf_err, rtol=1e-3, atol=1e-4)


@pytest.mark.skipif(not Path(r"H:\AI_Memory_Vault\models\Qwen3.8-2.4T-A95B-FP8").exists(), reason="FP8 source not found on H: drive")
def test_expert_229_down_proj_ls_optimal_fidelity():
    """C-8: Verify that LS optimal cuts expert 229 down_proj weight relative error < 0.45."""
    from tritforge.converter import ShardIndex
    from tritforge.slice import extract_tensor, read_safetensors_header

    src_dir = Path(r"H:\AI_Memory_Vault\models\Qwen3.8-2.4T-A95B-FP8")
    idx = ShardIndex(src_dir)
    t_name = "model.layers.0.mlp.experts.229.down_proj.weight"
    s, info, d = idx.tensor_to_shard[t_name]
    h, _ = read_safetensors_header(s)
    scale_name = f"{t_name}_scale_inv"
    h_sub = {t_name: h[t_name]}
    if scale_name in idx.tensor_to_shard:
        h_sub[scale_name] = idx.tensor_to_shard[scale_name][1]

    w = extract_tensor(s, t_name, header=h_sub, data_start_offset=d)

    q_ls = absmax_ternary_quantize(w, scaling_mode="block", scale_method="ls_optimal", block_size=128)
    w_ls = q_ls.reconstruct()
    rel_err = float(np.linalg.norm(w - w_ls) / np.linalg.norm(w))

    assert rel_err < 0.45, f"Expected relative error < 0.45, got {rel_err:.4f}"
