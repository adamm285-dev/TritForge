"""
TritForge Multi-Expert Parallel Batch Dispatcher (tritforge.dispatcher)
=======================================================================
Efficient token routing and batch dispatch engine for Sparse Mixture-of-Experts.
Permutes tokens into contiguous expert batches, executes zero-multiply ternary
inference, and scatters outputs back to the original sequence order.

Core Architectural Properties:
- Permutation-Based Dispatch: Eliminates Python per-token loops using vectorized scatter-gather.
- Zero-Multiply Expert Execution: Feeds contiguous token batches through TernaryExpert SwiGLU.
- Load-Balanced Telemetry: Monitors expert capacity, active lanes, and routing distribution.
- Gatekeeper Boundary: Preserves high-precision Router and Shared Expert paths.

Enforces:
- Invariant 1: Pure Python & NumPy zero-dependency core.
- Invariant 2: Mathematical identity with sequential MoE forward pass.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from tritforge.moe import Layer0AsymmetricMoE, MoERouter, SharedExpert, TernaryExpert, rmsnorm


@dataclass
class DispatchMetrics:
    """Telemetry captured during a batch dispatch forward pass."""
    total_tokens: int
    top_k: int
    active_experts: int
    expert_token_counts: Dict[int, int]
    routing_latency_ms: float
    expert_latency_ms: float
    scatter_latency_ms: float
    total_latency_ms: float
    load_imbalance_ratio: float  # max_load / mean_load


class MoEBatchDispatcher:
    """
    High-performance vectorized token dispatcher for Sparse MoE layers.

    Implements the token permutation pattern:
    1. Flatten and sort (token_idx, rank) pairs by assigned expert ID.
    2. Segment tokens into contiguous expert batches.
    3. Execute expert forward passes on batched tokens using zero-multiply GEMMs.
    4. Scatter-accumulate weighted expert outputs into the original sequence tensor.
    """
    def __init__(
        self,
        router: MoERouter,
        experts: Dict[int, TernaryExpert],
        shared_expert: Optional[SharedExpert] = None,
        norm_weight: Optional[np.ndarray] = None,
        num_experts: int = 512,
        num_experts_per_tok: int = 10,
    ):
        self.router = router
        self.experts = experts
        self.shared_expert = shared_expert
        self.norm_weight = np.asarray(norm_weight, dtype=np.float32) if norm_weight is not None else None
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

    def dispatch_and_execute(
        self,
        x: np.ndarray,
        residual: bool = True,
        collect_metrics: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, DispatchMetrics]]:
        """
        Dispatches tokens across experts and accumulates outputs.

        Args:
            x: Input activation tensor of shape (B, hidden_size) or (B, SeqLen, hidden_size).
            residual: Whether to add the residual connection (x + MoE(x)).
            collect_metrics: If True, returns DispatchMetrics telemetry.

        Returns:
            Output tensor of shape matching x (float32).
            metrics (optional): DispatchMetrics if collect_metrics=True.
        """
        t_start = time.perf_counter()
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1]).astype(np.float32)
        n_tokens, hidden_size = x_2d.shape

        # 1. Pre-MoE Normalization
        if self.norm_weight is not None:
            normed_x = rmsnorm(x_2d, self.norm_weight, zero_centered=True)
        else:
            normed_x = x_2d

        # 2. Gatekeeper Router Top-K Routing
        t_route_0 = time.perf_counter()
        topk_indices, topk_probs = self.router.forward(normed_x, top_k=self.num_experts_per_tok)
        t_route_1 = time.perf_counter()

        # 3. Vectorized Token Permutation
        # Flatten routing assignments
        k = self.num_experts_per_tok
        flat_expert_ids = topk_indices.ravel()
        flat_token_ids = np.repeat(np.arange(n_tokens), k)
        flat_ranks = np.tile(np.arange(k), n_tokens)
        flat_probs = topk_probs.ravel()

        # Sort assignments by expert ID for contiguous memory access
        sort_order = np.argsort(flat_expert_ids, kind="stable")
        sorted_expert_ids = flat_expert_ids[sort_order]
        sorted_token_ids = flat_token_ids[sort_order]
        sorted_probs = flat_probs[sort_order]

        # Identify unique active experts and their partition boundaries
        unique_experts, split_indices = np.unique(sorted_expert_ids, return_index=True)
        split_boundaries = np.append(split_indices[1:], len(sorted_expert_ids))

        # Output accumulator
        expert_accum = np.zeros_like(x_2d)
        expert_counts: Dict[int, int] = {}

        # 4. Batched Expert Inference
        t_expert_0 = time.perf_counter()
        for idx, e_id in enumerate(unique_experts):
            start_i = split_indices[idx]
            end_i = split_boundaries[idx]
            token_count = end_i - start_i
            expert_counts[int(e_id)] = token_count

            if e_id not in self.experts:
                raise KeyError(
                    f"Expert {e_id} required for active batch was not loaded in memory (C-5)"
                )

            expert = self.experts[e_id]
            batch_tok_ids = sorted_token_ids[start_i:end_i]
            batch_probs = sorted_probs[start_i:end_i, np.newaxis]

            # Gather batched activations
            x_batch = normed_x[batch_tok_ids]

            # Execute Zero-Multiply Ternary Expert SwiGLU
            out_batch = expert.forward(x_batch)

            # Apply gating probabilities
            weighted_batch = out_batch * batch_probs

            # Scatter-accumulate into output tensor
            np.add.at(expert_accum, batch_tok_ids, weighted_batch)
        t_expert_1 = time.perf_counter()

        # 5. High-Precision Shared Expert Path
        t_scatter_0 = time.perf_counter()
        if self.shared_expert is not None:
            shared_out = self.shared_expert.forward(normed_x)
            expert_accum += shared_out

        # 6. Residual Connection
        final_out = (x_2d + expert_accum) if residual else expert_accum
        final_out = final_out.reshape(orig_shape)
        t_end = time.perf_counter()

        if not collect_metrics:
            return final_out

        # Telemetry computation
        counts_arr = np.array(list(expert_counts.values())) if expert_counts else np.array([1])
        load_imbalance = float(np.max(counts_arr) / np.maximum(np.mean(counts_arr), 1e-6))

        metrics = DispatchMetrics(
            total_tokens=n_tokens,
            top_k=k,
            active_experts=len(unique_experts),
            expert_token_counts=expert_counts,
            routing_latency_ms=(t_route_1 - t_route_0) * 1000.0,
            expert_latency_ms=(t_expert_1 - t_expert_0) * 1000.0,
            scatter_latency_ms=(t_end - t_scatter_0) * 1000.0,
            total_latency_ms=(t_end - t_start) * 1000.0,
            load_imbalance_ratio=load_imbalance,
        )
        return final_out, metrics


def create_batch_dispatcher_from_moe(
    moe_layer: Layer0AsymmetricMoE,
) -> MoEBatchDispatcher:
    """
    Constructs an optimized MoEBatchDispatcher directly from a Layer0AsymmetricMoE instance.
    """
    return MoEBatchDispatcher(
        router=moe_layer.router,
        experts=moe_layer.experts,
        shared_expert=moe_layer.shared_expert,
        norm_weight=moe_layer.post_attention_layernorm_weight,
        num_experts=moe_layer.num_experts,
        num_experts_per_tok=moe_layer.num_experts_per_tok,
    )
