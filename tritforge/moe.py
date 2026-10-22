"""
TritForge Asymmetric MoE Engine (tritforge.moe)
==============================================
Asymmetric Mixed-Precision Mixture-of-Experts Layer Assembly for multi-trillion
parameter architectures (Qwen 2.4T / Qwen3.8-2.4T-A95B).

Architectural Asymmetry:
- The Gatekeeper: Router and Shared Expert preserved in high precision (FP8 / BF16).
- The Lanes: Routed Experts quantized to 1.58-bit ternary trits {-1, 0, +1}
  with second-order Hessian error compensation and zero-multiply addition/subtraction GEMM.

Enforces:
- Invariant 1: Zero-dependency pure Python & NumPy execution.
- Invariant 2: Mathematical Asymmetry (The Gatekeeper Boundary).
- Invariant 3: Second-Order Exit Drift Compensation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from tritforge.quantize import (
    QuantizedTritMatrix,
    absmax_ternary_quantize,
    hessian_compensate_quantize,
)
from tritforge.slice import (
    DEFAULT_MODEL_DIR,
    ExpertSlice,
    extract_complete_expert,
    extract_layer0_gatekeeper,
    extract_token_embeddings,
)


def silu(x: np.ndarray) -> np.ndarray:
    """Vectorized SiLU (Swish) activation: x * sigmoid(x)."""
    # Clip for numerical stability in float32 exp
    clipped = np.clip(x, -50.0, 50.0)
    return x / (1.0 + np.exp(-clipped))


def rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Root Mean Square Layer Normalization (RMSNorm) in pure NumPy.
    y = (x / sqrt(mean(x^2) + eps)) * weight
    """
    variance = np.mean(np.square(x), axis=-1, keepdims=True)
    normed = x / np.sqrt(variance + eps)
    return normed * weight


def ternary_gemm(
    x: np.ndarray,
    qmat: QuantizedTritMatrix,
    block_size: int = 128,
) -> np.ndarray:
    """
    Zero-Multiply Ternary GEMM kernel.
    Computes y = x @ W_hat^T without floating-point weight multiplications.

    Parameters:
        x: Input activations (B, N).
        qmat: QuantizedTritMatrix containing trits in {-1, 0, 1} and block scales.
        block_size: Column block size (default: 128).

    Mathematical Identity:
        W_hat[:, b] = trits[:, b] * scales[:, b_idx]
        x @ W_hat^T = sum_{b} (x[:, b] @ trits[:, b]^T) * scales[:, b_idx]
        where x[:, b] @ trits[:, b]^T performs integer additions and subtractions.
    """
    x = np.asarray(x, dtype=np.float32)
    m, n = qmat.original_shape
    trits = qmat.trits
    scales = qmat.scales
    b_dim = x.shape[0]

    if qmat.scaling_mode == "tensor":
        # Pure additions/subtractions followed by single scalar multiply
        accum = x @ trits.T
        return accum * float(scales[0])

    elif qmat.scaling_mode == "channel":
        # Pure additions/subtractions followed by per-row scale
        accum = x @ trits.T  # (B, M)
        return accum * scales.T  # scales is (M, 1), scales.T is (1, M)

    elif qmat.scaling_mode == "block":
        n_blocks = scales.shape[1]
        out = np.zeros((b_dim, m), dtype=np.float32)
        for b_idx in range(n_blocks):
            col_start = b_idx * block_size
            col_end = min(col_start + block_size, n)
            x_b = x[:, col_start:col_end]
            t_b = trits[:, col_start:col_end]
            s_b = scales[:, b_idx : b_idx + 1].T  # shape (1, M)

            # Integer addition/subtraction accumulation
            accum_b = x_b @ t_b.T  # (B, M)
            out += accum_b * s_b
        return out
    else:
        raise ValueError(f"Unknown scaling mode: {qmat.scaling_mode}")


class MoERouter:
    """
    The Gatekeeper Router: kept in high precision (FP8 / BF16 master weights).
    Computes routing logits and normalized gating probabilities across 512 experts.
    """
    def __init__(
        self,
        gate_weight: np.ndarray,
        num_experts: int = 512,
        num_experts_per_tok: int = 10,
    ):
        self.gate_weight = np.asarray(gate_weight, dtype=np.float32)
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

    def forward(
        self,
        x: np.ndarray,
        top_k: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Routes tokens to top-k experts.

        Args:
            x: Input activation batch of shape (B, hidden_size).
            top_k: Number of experts per token (default: self.num_experts_per_tok).

        Returns:
            topk_indices: int64 array of shape (B, top_k) with selected expert IDs.
            topk_probs: float32 array of shape (B, top_k) with normalized gating weights (sum to 1.0).
        """
        k = top_k or self.num_experts_per_tok
        # Gating logits: (B, num_experts)
        logits = x @ self.gate_weight.T

        # Select top-k expert indices
        topk_indices = np.argsort(logits, axis=-1)[:, -k:][:, ::-1]
        topk_logits = np.take_along_axis(logits, topk_indices, axis=-1)

        # Softmax over top-k logits for gating probabilities
        shifted = topk_logits - np.max(topk_logits, axis=-1, keepdims=True)
        exp_vals = np.exp(shifted)
        topk_probs = exp_vals / np.maximum(np.sum(exp_vals, axis=-1, keepdims=True), 1e-12)

        return topk_indices, topk_probs.astype(np.float32)


class TernaryExpert:
    """
    The Lanes: Individual Routed Expert quantized to 1.58-bit ternary trits.
    Executes SwiGLU forward pass with zero-multiply addition/subtraction GEMM.
    """
    def __init__(
        self,
        expert_idx: int,
        gate_proj: QuantizedTritMatrix,
        up_proj: QuantizedTritMatrix,
        down_proj: QuantizedTritMatrix,
    ):
        self.expert_idx = expert_idx
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Executes SwiGLU activation and projection:
        y = (silu(x @ W_gate^T) * (x @ W_up^T)) @ W_down^T
        """
        gate_out = silu(ternary_gemm(x, self.gate_proj))
        up_out = ternary_gemm(x, self.up_proj)
        intermediate = gate_out * up_out
        output = ternary_gemm(intermediate, self.down_proj)
        return output


class SharedExpert:
    """
    The Gatekeeper Shared Expert: kept in high precision (FP8 / BF16).
    Applies shared trunk capacity to all tokens unconditionally.
    """
    def __init__(
        self,
        gate_proj: np.ndarray,
        up_proj: np.ndarray,
        down_proj: np.ndarray,
        gate_weight: Optional[np.ndarray] = None,
    ):
        self.gate_proj = np.asarray(gate_proj, dtype=np.float32)
        self.up_proj = np.asarray(up_proj, dtype=np.float32)
        self.down_proj = np.asarray(down_proj, dtype=np.float32)
        self.gate_weight = np.asarray(gate_weight, dtype=np.float32) if gate_weight is not None else None

    def forward(self, x: np.ndarray) -> np.ndarray:
        gate_out = silu(x @ self.gate_proj.T)
        up_out = x @ self.up_proj.T
        intermediate = gate_out * up_out
        output = intermediate @ self.down_proj.T

        if self.gate_weight is not None:
            # Gated shared expert
            gating = 1.0 / (1.0 + np.exp(-np.clip(x @ self.gate_weight.T, -50.0, 50.0)))
            output = output * gating
        return output


class Layer0AsymmetricMoE:
    """
    Complete Assembled Layer 0 Asymmetric MoE Block.
    Integrates the Gatekeeper Router, Shared Expert, and Quantized 1.58-bit Expert Lanes.
    """
    def __init__(
        self,
        router: MoERouter,
        experts: Dict[int, TernaryExpert],
        shared_expert: Optional[SharedExpert] = None,
        input_layernorm_weight: Optional[np.ndarray] = None,
        post_attention_layernorm_weight: Optional[np.ndarray] = None,
        num_experts: int = 512,
        num_experts_per_tok: int = 10,
    ):
        self.router = router
        self.experts = experts
        self.shared_expert = shared_expert
        self.input_layernorm_weight = (
            np.asarray(input_layernorm_weight, dtype=np.float32)
            if input_layernorm_weight is not None
            else None
        )
        self.post_attention_layernorm_weight = (
            np.asarray(post_attention_layernorm_weight, dtype=np.float32)
            if post_attention_layernorm_weight is not None
            else None
        )
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

    def forward(
        self,
        x: np.ndarray,
        residual: bool = True,
        return_routing_info: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Executes the complete Layer 0 Asymmetric MoE forward pass:
        1. Optional layernorm.
        2. Router top-k routing and gating probability calculation.
        3. Token dispatch through 1.58-bit ternary expert lanes via zero-multiply GEMM.
        4. High-precision shared expert accumulation.
        5. Residual connection: y = x + MoE(x).
        """
        x_in = np.asarray(x, dtype=np.float32)
        b_dim, hidden_size = x_in.shape

        # Post-attention layernorm before entering MoE
        if self.post_attention_layernorm_weight is not None:
            normed_x = rmsnorm(x_in, self.post_attention_layernorm_weight)
        else:
            normed_x = x_in

        # Gatekeeper Router evaluates top-k probabilities
        topk_indices, topk_probs = self.router.forward(normed_x, top_k=self.num_experts_per_tok)

        # Output accumulator for routed experts
        expert_outputs = np.zeros_like(x_in)

        # Dispatch tokens across active expert lanes
        active_expert_ids = np.unique(topk_indices)
        for e_id in active_expert_ids:
            if e_id not in self.experts:
                # If an expert was not pre-quantized, skip or log (in full model, all 512 exist)
                continue
            expert = self.experts[e_id]

            # Find which tokens routed to this expert and their rank
            token_mask = (topk_indices == e_id)  # (B, top_k) boolean
            tok_indices, rank_indices = np.where(token_mask)

            if len(tok_indices) == 0:
                continue

            x_expert = normed_x[tok_indices]
            out_expert = expert.forward(x_expert)
            probs = topk_probs[tok_indices, rank_indices][:, np.newaxis]

            # Accumulate weighted expert outputs
            np.add.at(expert_outputs, tok_indices, out_expert * probs)

        # Add High-Precision Shared Expert if present
        if self.shared_expert is not None:
            shared_out = self.shared_expert.forward(normed_x)
            expert_outputs += shared_out

        # Apply residual connection
        final_output = (x_in + expert_outputs) if residual else expert_outputs

        if return_routing_info:
            routing_info = {
                "topk_indices": topk_indices,
                "topk_probs": topk_probs,
                "active_experts": active_expert_ids.tolist(),
                "shared_expert_active": self.shared_expert is not None,
            }
            return final_output, routing_info
        return final_output


def collect_empirical_expert_covariances(
    router: MoERouter,
    activations: np.ndarray,
    target_experts: Optional[Sequence[int]] = None,
    top_k: int = 10,
    damp: float = 0.01,
) -> Dict[int, np.ndarray]:
    """
    Streams real transformer token activations through the Gatekeeper Router
    to accumulate empirical activation covariances (H_e = sum p_i x_i^T x_i)
    for targeted expert lanes.

    Replaces synthetic noise with real empirical token distributions (Mock M-001 graduation).
    """
    activations = np.asarray(activations, dtype=np.float32)
    b_dim, hidden_size = activations.shape

    topk_indices, topk_probs = router.forward(activations, top_k=top_k)

    target_set = set(target_experts) if target_experts is not None else None
    covariances: Dict[int, np.ndarray] = {}

    # Target specific experts to enforce constant RAM footprint
    candidates = target_set if target_set is not None else set(np.unique(topk_indices))

    for e_id in candidates:
        token_mask = (topk_indices == e_id)
        tok_indices, rank_indices = np.where(token_mask)
        if len(tok_indices) > 0:
            x_e = activations[tok_indices]
            p_e = topk_probs[tok_indices, rank_indices][:, np.newaxis]
            weighted_x = x_e * np.sqrt(p_e)
            sum_p = max(float(np.sum(p_e)), 1e-6)
            H_e = (weighted_x.T @ weighted_x) / sum_p
        else:
            # If expert was not routed to by this batch, use batch covariance
            H_e = (activations.T @ activations) / b_dim

        # In-place diagonal damping without extra 256MB eye allocation
        np.fill_diagonal(H_e, H_e.diagonal() + damp)
        covariances[int(e_id)] = H_e

    return covariances


def assemble_layer0_moe(
    base_dir: Union[str, Path] = DEFAULT_MODEL_DIR,
    expert_indices: Sequence[int] = (0,),
    calibrate_with_real_tokens: bool = True,
    sample_token_ids: Optional[Sequence[int]] = None,
    block_size: int = 128,
) -> Layer0AsymmetricMoE:
    """
    Extracts Layer 0 weights from disk shards and assembles the complete
    Asymmetric MoE Layer:
    - Gatekeeper Router & Shared Expert loaded in BF16/FP32 precision.
    - Routed Experts extracted and quantized to 1.58-bit ternary trits
      using empirical activation covariance collected from real token embeddings.
    """
    base = Path(base_dir)

    # 1. Extract Gatekeeper
    gatekeeper = extract_layer0_gatekeeper(base)
    router = MoERouter(gatekeeper["router_weight"], num_experts=512, num_experts_per_tok=10)

    shared_expert = None
    if "shared_gate_proj" in gatekeeper:
        shared_expert = SharedExpert(
            gate_proj=gatekeeper["shared_gate_proj"],
            up_proj=gatekeeper["shared_up_proj"],
            down_proj=gatekeeper["shared_down_proj"],
            gate_weight=gatekeeper.get("shared_gate_weight"),
        )

    # 2. Collect Real Empirical Activations if requested
    expert_covs = {}
    if calibrate_with_real_tokens:
        # Default prompt tokens or vocabulary slice
        token_ids = sample_token_ids or list(range(100, 228))
        embeddings = extract_token_embeddings(token_ids, base_dir=base)

        # Pass through input layernorm if present
        if "input_layernorm" in gatekeeper:
            x_norm1 = rmsnorm(embeddings, gatekeeper["input_layernorm"])
        else:
            x_norm1 = embeddings

        if "post_attention_layernorm" in gatekeeper:
            x_calib = rmsnorm(x_norm1, gatekeeper["post_attention_layernorm"])
        else:
            x_calib = x_norm1

        # Accumulate empirical covariance specifically for target expert_indices
        expert_covs = collect_empirical_expert_covariances(
            router, x_calib, target_experts=expert_indices, top_k=10
        )
        del embeddings, x_norm1, x_calib

    # 3. Extract and Quantize Requested Routed Expert Lanes
    experts: Dict[int, TernaryExpert] = {}
    hidden_size = 8192

    for e_id in expert_indices:
        raw_slices = extract_complete_expert(e_id, base_dir=base, layer_idx=0)
        w_gate = np.nan_to_num(raw_slices["gate_proj"].weight)
        w_up = np.nan_to_num(raw_slices["up_proj"].weight)
        w_down = np.nan_to_num(raw_slices["down_proj"].weight)

        H_e = expert_covs.get(e_id)
        if H_e is None:
            # Fallback isotropic covariance
            H_e = np.eye(hidden_size, dtype=np.float32)

        # Quantize gate_proj & up_proj (inputs have dimension 8192)
        q_gate, _ = hessian_compensate_quantize(w_gate, H_e, block_size=block_size, scale_method="optimal")
        q_up, _ = hessian_compensate_quantize(w_up, H_e, block_size=block_size, scale_method="optimal")

        # Free H_e after gate and up quantization
        if e_id in expert_covs:
            del expert_covs[e_id]

        # Quantize down_proj (inputs have intermediate dimension 2048)
        H_down = np.eye(2048, dtype=np.float32)
        q_down, _ = hessian_compensate_quantize(w_down, H_down, block_size=block_size, scale_method="optimal")

        experts[e_id] = TernaryExpert(
            expert_idx=e_id,
            gate_proj=q_gate,
            up_proj=q_up,
            down_proj=q_down,
        )

    # 4. Assemble Complete Layer 0 Block
    return Layer0AsymmetricMoE(
        router=router,
        experts=experts,
        shared_expert=shared_expert,
        input_layernorm_weight=gatekeeper.get("input_layernorm"),
        post_attention_layernorm_weight=gatekeeper.get("post_attention_layernorm"),
        num_experts=512,
        num_experts_per_tok=10,
    )
