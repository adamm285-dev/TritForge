"""
TritForge Ternary Kernel (tritforge.quantize)
==============================================
Zero-dependency, clean-room 1.58-bit ternary quantization kernel implementing
Absmax quantization, second-order Hessian error compensation, and bit-packing.

Enforces:
- Invariant 1: Pure Python & NumPy tensor algebra.
- Invariant 3: Second-order exit drift compensation via activation Hessian.
- Invariant 5: Reconstructibility and mathematical determinism.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple, Union

import numpy as np


ScaleMethod = Literal["absmax", "mean_abs", "optimal", "ls_optimal"]
ScalingMode = Literal["channel", "block", "tensor"]


@dataclass
class QuantizedTritMatrix:
    """
    Encapsulates a 1.58-bit ternary quantized weight matrix.
    Trits are strictly {-1, 0, +1}.
    """
    trits: np.ndarray               # int8 array containing only {-1, 0, 1}
    scales: np.ndarray              # float32 scale factors (channel, block, or scalar)
    original_shape: Tuple[int, int] # Original (M, N) matrix shape
    scaling_mode: ScalingMode
    scale_method: ScaleMethod = "mean_abs"
    block_size: Optional[int] = None
    method: str = "absmax_naive"

    def __post_init__(self):
        # Invariant check (L-8): trits must strictly be in {-1, 0, 1}
        # Uses min()/max() check instead of costly np.unique on millions of elements
        if self.trits.size > 0:
            t_min = int(self.trits.min())
            t_max = int(self.trits.max())
            if t_min < -1 or t_max > 1:
                raise ValueError(f"Quantized trits violate ternary invariant: min={t_min}, max={t_max}")

    @property
    def density(self) -> float:
        """Fraction of non-zero ternary parameters."""
        return float(np.count_nonzero(self.trits) / self.trits.size)

    @property
    def sparsity(self) -> float:
        """Fraction of zero parameters (eliminated operations)."""
        return 1.0 - self.density

    @property
    def raw_memory_mb(self) -> float:
        """Memory size in MB if stored as 1-byte int8 + scales."""
        return (self.trits.nbytes + self.scales.nbytes) / (1024 * 1024)

    @property
    def packed_2bit_mb(self) -> float:
        """Memory size in MB under 2-bit packing (4 trits/byte) + scales."""
        packed_bytes = int(np.ceil(self.trits.size / 4))
        return (packed_bytes + self.scales.nbytes) / (1024 * 1024)

    @property
    def packed_base3_mb(self) -> float:
        """Memory size in MB under Base-3 packing (5 trits/byte = 1.6 bits/trit) + scales."""
        packed_bytes = int(np.ceil(self.trits.size / 5))
        return (packed_bytes + self.scales.nbytes) / (1024 * 1024)

    def reconstruct(self) -> np.ndarray:
        """Reconstructs the floating-point matrix approximation."""
        return dequantize_trits(self)


def _compute_scale(mat: np.ndarray, method: ScaleMethod, axis: Optional[int] = None, keepdims: bool = True) -> np.ndarray:
    """
    Computes scale factor according to method:
    - 'absmax': max(|mat|)
    - 'mean_abs': mean(|mat|) (BitNet b1.58 standard)
    - 'optimal': 1.5 * mean(|mat|) (Optimal ternary scale minimizing Frobenius reconstruction error)
    """
    abs_mat = np.abs(mat)
    if method == "absmax":
        sc = np.max(abs_mat, axis=axis, keepdims=keepdims)
    elif method == "mean_abs":
        sc = np.mean(abs_mat, axis=axis, keepdims=keepdims)
    elif method == "optimal":
        sc = np.mean(abs_mat, axis=axis, keepdims=keepdims) * 1.5
    else:
        raise ValueError(f"Unknown scale method: {method}")
    return np.maximum(sc, 1e-8).astype(np.float32)


def absmax_ternary_quantize(
    W: np.ndarray,
    scaling_mode: ScalingMode = "block",
    scale_method: ScaleMethod = "mean_abs",
    block_size: int = 128,
    threshold_ratio: float = 0.5,
) -> QuantizedTritMatrix:
    """
    Quantizes a floating-point matrix W to ternary trits {-1, 0, +1} using Absmax/Mean-Abs scaling.

    Args:
        W: 2D float32 weight matrix of shape (M, N).
        scaling_mode:
            - 'channel': per-row scale (M, 1)
            - 'block': block scale along columns of size `block_size` (M, N // block_size)
            - 'tensor': single scalar scale
        scale_method: 'absmax', 'mean_abs' (BitNet standard), or 'optimal' (1.5x mean_abs)
        block_size: column block size when scaling_mode == 'block'.
        threshold_ratio: threshold for mapping to {-1, 0, +1} (default 0.5 for rounding).

    Returns:
        QuantizedTritMatrix instance.
    """
    W = np.asarray(W, dtype=np.float32)
    m, n = W.shape

    if scaling_mode == "tensor":
        scale = _compute_scale(W, scale_method, axis=None, keepdims=False)
        scaled = W / scale
        trits = np.where(
            scaled > threshold_ratio,
            1,
            np.where(scaled < -threshold_ratio, -1, 0)
        ).astype(np.int8)
        scales = np.array([scale], dtype=np.float32)

    elif scaling_mode == "channel":
        scale = _compute_scale(W, scale_method, axis=1, keepdims=True)
        scaled = W / scale
        trits = np.where(
            scaled > threshold_ratio,
            1,
            np.where(scaled < -threshold_ratio, -1, 0)
        ).astype(np.int8)
        scales = scale

    elif scaling_mode == "block":
        n_blocks = int(np.ceil(n / block_size))
        scales = np.zeros((m, n_blocks), dtype=np.float32)
        trits = np.zeros((m, n), dtype=np.int8)

        if scale_method == "ls_optimal":
            # D6: Vectorize all full blocks at once (tensor 3D batching), loop only for ragged remainder
            bs = block_size
            n_full = (n // bs) * bs
            if n_full > 0:
                full_blocks = W[:, :n_full].reshape(m, n_full // bs, bs)
                a = np.abs(full_blocks)
                s = -np.sort(-a, axis=2)
                c = np.cumsum(s, axis=2)
                ks = np.arange(1, bs + 1, dtype=np.float32)
                k = np.argmax(c * c / ks, axis=2) + 1  # (m, n_full // bs)
                thr = np.take_along_axis(s, (k - 1)[..., None], 2)  # (m, n_full // bs, 1)
                alpha = (np.take_along_axis(c, (k - 1)[..., None], 2) / k[..., None]).astype(np.float16).astype(np.float32)
                trits[:, :n_full] = np.where(a >= thr, np.sign(full_blocks), 0).astype(np.int8).reshape(m, n_full)
                scales[:, :n_full // bs] = alpha.reshape(m, n_full // bs)

            if n > n_full:
                # Ragged trailing block
                k_cols = n - n_full
                w_rem = W[:, n_full:]
                a_rem = np.abs(w_rem)
                s_rem = -np.sort(-a_rem, axis=1)
                c_rem = np.cumsum(s_rem, axis=1)
                ks_rem = np.arange(1, k_cols + 1, dtype=np.float32)
                best_k = np.argmax(c_rem * c_rem / ks_rem, axis=1) + 1  # (m,)
                thr_rem = np.take_along_axis(s_rem, (best_k - 1)[:, None], 1)  # (m, 1)
                alpha_rem = (np.take_along_axis(c_rem, (best_k - 1)[:, None], 1) / best_k[:, None]).astype(np.float16).astype(np.float32)
                trits[:, n_full:] = np.where(a_rem >= thr_rem, np.sign(w_rem), 0).astype(np.int8)
                scales[:, -1:] = alpha_rem
        else:
            for b_idx in range(n_blocks):
                col_start = b_idx * block_size
                col_end = min(col_start + block_size, n)
                w_b = W[:, col_start:col_end]
                scale_b = _compute_scale(w_b, scale_method, axis=1, keepdims=True)
                scales[:, b_idx : b_idx + 1] = scale_b

                scaled_b = w_b / scale_b
                trits[:, col_start:col_end] = np.where(
                    scaled_b > threshold_ratio,
                    1,
                    np.where(scaled_b < -threshold_ratio, -1, 0)
                ).astype(np.int8)
    else:
        raise ValueError(f"Unknown scaling mode: {scaling_mode}")

    return QuantizedTritMatrix(
        trits=trits,
        scales=scales,
        original_shape=(m, n),
        scaling_mode=scaling_mode,
        scale_method=scale_method,
        block_size=block_size if scaling_mode == "block" else None,
        method="absmax_naive",
    )


def dequantize_trits(qmat: QuantizedTritMatrix) -> np.ndarray:
    """
    Reconstructs the floating-point approximation W_hat from a QuantizedTritMatrix.
    """
    trits = qmat.trits
    scales = qmat.scales
    m, n = qmat.original_shape

    if qmat.scaling_mode == "tensor":
        return trits.astype(np.float32) * float(scales[0])

    elif qmat.scaling_mode == "channel":
        return trits.astype(np.float32) * scales

    elif qmat.scaling_mode == "block":
        block_size = qmat.block_size or 128
        W_hat = np.zeros((m, n), dtype=np.float32)
        n_blocks = scales.shape[1]
        for b_idx in range(n_blocks):
            col_start = b_idx * block_size
            col_end = min(col_start + block_size, n)
            scale_b = scales[:, b_idx : b_idx + 1]
            W_hat[:, col_start:col_end] = trits[:, col_start:col_end].astype(np.float32) * scale_b
        return W_hat

    else:
        raise ValueError(f"Unknown scaling mode: {qmat.scaling_mode}")


def compute_frobenius_error(W: np.ndarray, W_hat: np.ndarray) -> Tuple[float, float]:
    """
    Calculates absolute and relative Frobenius norm errors between original and quantized weights:
    - abs_error = ||W - W_hat||_F
    - rel_error = ||W - W_hat||_F / ||W||_F
    """
    W = np.asarray(W, dtype=np.float32)
    W_hat = np.asarray(W_hat, dtype=np.float32)
    diff = W - W_hat
    if diff.ndim == 2:
        abs_err = float(np.linalg.norm(diff, ord="fro"))
        w_norm = float(np.linalg.norm(W, ord="fro"))
    else:
        abs_err = float(np.linalg.norm(diff.ravel()))
        w_norm = float(np.linalg.norm(W.ravel()))
    rel_err = abs_err / max(w_norm, 1e-12)
    return abs_err, rel_err


def compute_cosine_similarity(W: np.ndarray, W_hat: np.ndarray) -> float:
    """
    Calculates flattened cosine similarity: (W . W_hat) / (||W||_2 * ||W_hat||_2).
    """
    w_flat = np.asarray(W, dtype=np.float32).ravel()
    what_flat = np.asarray(W_hat, dtype=np.float32).ravel()
    dot = float(np.dot(w_flat, what_flat))
    norm_w = float(np.linalg.norm(w_flat))
    norm_what = float(np.linalg.norm(what_flat))
    denom = max(norm_w * norm_what, 1e-12)
    return dot / denom


def compute_hessian_error(W: np.ndarray, W_hat: np.ndarray, H: np.ndarray) -> float:
    """
    Calculates second-order reconstruction error:
    L_H = Tr( (W - W_hat) H (W - W_hat)^T ) = sum( (E @ H) * E )
    where E = W - W_hat.
    """
    E = np.asarray(W - W_hat, dtype=np.float32)
    H = np.asarray(H, dtype=np.float32)
    EH = E @ H
    loss = float(np.sum(EH * E))
    return loss


def hessian_compensate_quantize(
    W: np.ndarray,
    H: np.ndarray,
    block_size: int = 128,
    scale_method: ScaleMethod = "optimal",
    damp: float = 1e-4,
) -> Tuple[QuantizedTritMatrix, np.ndarray]:
    """
    Second-order Hessian-compensated ternary quantization.

    Minimizes:
        L_H = Tr( (W - W_hat) H (W - W_hat)^T )

    Algorithm:
    1. Regularize and invert Hessian: H_inv = (H + lambda * I)^(-1).
    2. Sequentially iterate through column blocks of size `block_size`:
       - Quantize block B to ternary trits and compute quantization residual E_B = W_B - W_hat_B.
       - Compensate remaining unquantized columns R using:
         W_R <- W_R - E_B @ (H_inv[B, B])^(-1) @ H_inv[B, R]
       - Freeze block B.
    """
    W = np.asarray(W, dtype=np.float32).copy()
    m, n = W.shape
    H = np.asarray(H, dtype=np.float32)

    if H.shape != (n, n):
        raise ValueError(f"Hessian shape {H.shape} must match columns of W ({n}, {n})")

    # Regularize Hessian with diagonal damping in-place (no 256MB eye allocation)
    diag_mean = float(np.mean(np.diag(H)))
    lambda_val = max(damp * diag_mean, 1e-6)
    H_damped = H.copy()
    np.fill_diagonal(H_damped, H_damped.diagonal() + lambda_val)

    # Invert Hessian
    H_inv = np.linalg.inv(H_damped)
    del H_damped

    n_blocks = int(np.ceil(n / block_size))
    scales = np.zeros((m, n_blocks), dtype=np.float32)
    trits = np.zeros((m, n), dtype=np.int8)
    W_hat = np.zeros((m, n), dtype=np.float32)

    for b_idx in range(n_blocks):
        col_start = b_idx * block_size
        col_end = min(col_start + block_size, n)
        b = slice(col_start, col_end)
        r = slice(col_end, n)

        # Current weights for block B
        w_b = W[:, b]

        # Scale for current block
        scale_b = _compute_scale(w_b, scale_method, axis=1, keepdims=True)
        scales[:, b_idx : b_idx + 1] = scale_b

        # Quantize to ternary {-1, 0, 1}
        scaled_b = w_b / scale_b
        q_trits_b = np.where(
            scaled_b > 0.5,
            1,
            np.where(scaled_b < -0.5, -1, 0)
        ).astype(np.int8)
        trits[:, b] = q_trits_b

        # Reconstructed block weights
        q_b = q_trits_b.astype(np.float32) * scale_b
        W_hat[:, b] = q_b

        # Quantization error for block B
        e_b = w_b - q_b

        # Compensate remaining columns R
        if col_end < n:
            H_bb = H_inv[b, b]
            H_br = H_inv[b, r]
            # Solve H_bb @ K = H_br -> K = inv(H_bb) @ H_br
            K = np.linalg.solve(H_bb, H_br)
            W[:, r] -= e_b @ K

    qmat = QuantizedTritMatrix(
        trits=trits,
        scales=scales,
        original_shape=(m, n),
        scaling_mode="block",
        scale_method=scale_method,
        block_size=block_size,
        method="hessian_compensated",
    )
    return qmat, W_hat


def pack_trits_2bit(trits: np.ndarray) -> np.ndarray:
    """
    Packs ternary trits {-1, 0, +1} into 2-bit bitfields (4 trits per uint8 byte).
    Encoding:
      -1 -> 2 (binary 10)
       0 -> 0 (binary 00)
      +1 -> 1 (binary 01)
    """
    trits_flat = np.asarray(trits, dtype=np.int8).ravel()
    n_elems = len(trits_flat)

    # Map to 2-bit unsigned code
    encoded = np.where(trits_flat == -1, 2, np.where(trits_flat == 1, 1, 0)).astype(np.uint8)

    pad_len = (4 - (n_elems % 4)) % 4
    if pad_len > 0:
        encoded = np.pad(encoded, (0, pad_len))

    blocks = encoded.reshape(-1, 4)
    packed = (
        blocks[:, 0]
        | (blocks[:, 1] << 2)
        | (blocks[:, 2] << 4)
        | (blocks[:, 3] << 6)
    )
    return packed


def unpack_trits_2bit(packed: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """
    Unpacks 2-bit packed uint8 byte array back into int8 ternary trits {-1, 0, +1}.
    """
    packed = np.asarray(packed, dtype=np.uint8).ravel()
    total_elements = int(np.prod(shape))

    t0 = packed & 3
    t1 = (packed >> 2) & 3
    t2 = (packed >> 4) & 3
    t3 = (packed >> 6) & 3

    unpacked_enc = np.stack([t0, t1, t2, t3], axis=1).reshape(-1)[:total_elements]
    trits = np.where(unpacked_enc == 2, -1, np.where(unpacked_enc == 1, 1, 0)).astype(np.int8)
    return trits.reshape(shape)


def pack_trits_base3(trits: np.ndarray) -> np.ndarray:
    """
    Packs ternary trits {-1, 0, +1} using Base-3 encoding (5 trits per uint8 byte).
    Achieves 1.6 bits/trit density (3^5 = 243 <= 256).
    Encoding:
      trit + 1 -> {0, 1, 2}
      value = t0 + 3*t1 + 9*t2 + 27*t3 + 81*t4
    """
    trits_flat = np.asarray(trits, dtype=np.int8).ravel()
    n_elems = len(trits_flat)

    encoded = (trits_flat + 1).astype(np.uint8)
    pad_len = (5 - (n_elems % 5)) % 5
    if pad_len > 0:
        encoded = np.pad(encoded, (0, pad_len))

    blocks = encoded.reshape(-1, 5)
    packed = (
        blocks[:, 0]
        + blocks[:, 1] * 3
        + blocks[:, 2] * 9
        + blocks[:, 3] * 27
        + blocks[:, 4] * 81
    ).astype(np.uint8)
    return packed


def unpack_trits_base3(packed: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """
    Unpacks Base-3 encoded uint8 array back into int8 ternary trits {-1, 0, +1}.
    """
    packed = np.asarray(packed, dtype=np.uint8).ravel()
    total_elements = int(np.prod(shape))

    rem = packed.astype(np.uint16)
    t0 = rem % 3; rem = rem // 3
    t1 = rem % 3; rem = rem // 3
    t2 = rem % 3; rem = rem // 3
    t3 = rem % 3; rem = rem // 3
    t4 = rem % 3

    unpacked_enc = np.stack([t0, t1, t2, t3, t4], axis=1).reshape(-1)[:total_elements]
    trits = (unpacked_enc.astype(np.int8) - 1)
    return trits.reshape(shape)
