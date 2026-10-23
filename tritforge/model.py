"""
TritForge Full Asymmetric Model Engine (tritforge.model)
========================================================
End-to-end forward pipeline and autoregressive generation engine for Sparse
Mixture-of-Experts architectures with Asymmetric Mixed-Precision.

Architectural Asymmetry:
- The Gatekeeper (Embedding, Attention, Router, Shared Experts, Norms, LM Head):
  Preserved in original high precision (FP8 / BF16 / FP32).
- The Lanes (Routed Experts across all layers):
  Quantized to 1.58-bit ternary trits {-1, 0, +1} stored in native .trit containers
  and executed via zero-multiply addition/subtraction GEMM.

Workstation Memory Architecture:
- Streaming Expert Execution: Layers can stream active expert lanes on-demand directly
  from seek-based .trit containers, bounding active resident RAM within single-workstation
  budgets (128 GB - 256 GB RAM profile).

Enforces:
- Invariant 1: Zero external dependencies (pure Python standard library + NumPy).
- Invariant 2: Strict Gatekeeper boundary preservation.
- Invariant 4: Constant-memory seek-based storage and streaming execution.
"""

from __future__ import annotations

from collections import OrderedDict

import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from tritforge.container import load_trit_layer, load_trit_tensor
from tritforge.dispatcher import DispatchMetrics, MoEBatchDispatcher
from tritforge.moe import (
    Layer0AsymmetricMoE,
    MoERouter,
    SharedExpert,
    TernaryExpert,
    rmsnorm,
    silu,
    ternary_gemm,
)
from tritforge.quantize import QuantizedTritMatrix


@dataclass
class TritForgeConfig:
    """Hyperparameter configuration for full Asymmetric MoE models."""
    vocab_size: int = 248320
    hidden_size: int = 8192
    intermediate_size: int = 2048
    num_hidden_layers: int = 92
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    num_experts: int = 512
    num_experts_per_tok: int = 10
    shared_expert_intermediate_size: int = 2048
    rms_norm_eps: float = 1e-6
    block_size: int = 128
    max_position_embeddings: int = 32768
    rope_theta: float = 1000000.0
    tie_word_embeddings: bool = False

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def kv_dim(self) -> int:
        return self.num_key_value_heads * self.head_dim

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TritForgeConfig:
        valid_fields = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


class AsymmetricAttention:
    """
    High-precision Gatekeeper Attention module (Grouped-Query Attention).
    Preserves exact FP8/BF16/FP32 precision for Q, K, V, and O projections.
    """
    def __init__(
        self,
        config: TritForgeConfig,
        q_proj: np.ndarray,
        k_proj: np.ndarray,
        v_proj: np.ndarray,
        o_proj: np.ndarray,
    ):
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = 1.0 / np.sqrt(float(self.head_dim))

        self.q_proj = np.asarray(q_proj, dtype=np.float32)
        self.k_proj = np.asarray(k_proj, dtype=np.float32)
        self.v_proj = np.asarray(v_proj, dtype=np.float32)
        self.o_proj = np.asarray(o_proj, dtype=np.float32)

    def forward(
        self,
        x: np.ndarray,
        causal_mask: bool = True,
        kv_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Forward GQA attention computation.

        Args:
            x: Input activation of shape (B, S, D).
            causal_mask: Whether to apply lower-triangular causal attention mask.
            kv_cache: Optional past (K, V) tensors of shapes (B, past_S, num_kv_heads, head_dim).

        Returns:
            output: Attention output tensor of shape (B, S, D).
            new_kv_cache: Updated (K, V) tuple.
        """
        b_sz, seq_len, _ = x.shape

        # Linear projections
        # x: (B, S, D) @ W.T -> (B, S, out_dim)
        q = np.matmul(x, self.q_proj.T)  # (B, S, num_heads * head_dim)
        k = np.matmul(x, self.k_proj.T)  # (B, S, num_kv_heads * head_dim)
        v = np.matmul(x, self.v_proj.T)  # (B, S, num_kv_heads * head_dim)

        # Reshape to (B, num_heads, S, head_dim)
        q = q.reshape(b_sz, seq_len, self.num_heads, self.head_dim).swapaxes(1, 2)
        k = k.reshape(b_sz, seq_len, self.num_kv_heads, self.head_dim).swapaxes(1, 2)
        v = v.reshape(b_sz, seq_len, self.num_kv_heads, self.head_dim).swapaxes(1, 2)

        # Handle KV cache for autoregressive decoding
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = np.concatenate([past_k, k], axis=2)
            v = np.concatenate([past_v, v], axis=2)
        new_kv_cache = (k, v)
        total_kv_len = k.shape[2]

        # Expand KV heads for Grouped-Query Attention (GQA) if num_kv_heads < num_heads
        if self.num_kv_heads < self.num_heads:
            repeats = self.num_heads // self.num_kv_heads
            k_expanded = np.repeat(k, repeats, axis=1)
            v_expanded = np.repeat(v, repeats, axis=1)
        else:
            k_expanded = k
            v_expanded = v

        # Scaled dot-product attention: (B, num_heads, S, total_kv_len)
        scores = np.matmul(q, k_expanded.swapaxes(-1, -2)) * self.scale

        if causal_mask and seq_len > 1:
            # Construct causal mask where query i can only attend to key j <= i + past_len
            past_len = total_kv_len - seq_len
            q_idx = np.arange(seq_len)[:, None] + past_len
            k_idx = np.arange(total_kv_len)[None, :]
            mask = q_idx < k_idx  # True where masked out
            scores = np.where(mask[None, None, :, :], -1e9, scores)

        # Stable Softmax
        max_scores = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - max_scores)
        attn_probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

        # Context accumulation: (B, num_heads, S, head_dim)
        context = np.matmul(attn_probs, v_expanded)

        # Reshape back to (B, S, hidden_size)
        context = context.swapaxes(1, 2).reshape(b_sz, seq_len, self.hidden_size)

        # Output projection
        out = np.matmul(context, self.o_proj.T)
        return out, new_kv_cache


class StreamingMoELayer:
    """
    On-Demand Streaming MoE Layer for Single-Workstation Execution.

    Preserves resident Gatekeeper parameters (Router + Shared Expert) in RAM,
    while seeking and streaming active expert lanes directly from .trit binary
    container files during inference. Bounds resident RAM to active working set.
    """
    def __init__(
        self,
        router: MoERouter,
        trit_path: Optional[Union[str, Path]] = None,
        experts: Optional[Dict[int, TernaryExpert]] = None,
        shared_expert: Optional[SharedExpert] = None,
        norm_weight: Optional[np.ndarray] = None,
        num_experts: int = 512,
        num_experts_per_tok: int = 10,
        max_cached_experts: int = 64,
        layer_idx: int = 0,
    ):
        self.router = router
        self.trit_path = Path(trit_path) if trit_path is not None else None
        self.shared_expert = shared_expert
        self.norm_weight = np.asarray(norm_weight, dtype=np.float32) if norm_weight is not None else None
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.max_cached_experts = max_cached_experts
        self.layer_idx = layer_idx

        # In-memory expert cache: expert_idx -> TernaryExpert (OrderedDict for LRU, C-5)
        self.experts: OrderedDict[int, TernaryExpert] = OrderedDict(experts) if experts is not None else OrderedDict()
        self.lru_order: List[int] = list(self.experts.keys())
        self._lock = threading.Lock()

    def _ensure_experts_loaded(self, active_expert_ids: Sequence[int]) -> Dict[int, TernaryExpert]:
        """
        Loads missing active experts from .trit file via seek-based extraction (C-5).
        Enforces true LRU eviction (OrderedDict) and NEVER evicts experts needed in the current active set.
        Thread-safe: protected by self._lock.
        """
        with self._lock:
            active_set = set(int(e) for e in active_expert_ids)
            needed = [e for e in active_set if e not in self.experts]

            if needed:
                if self.trit_path is None or not self.trit_path.exists():
                    raise FileNotFoundError(
                        f"Expert indices {needed} required for Layer {self.layer_idx} but no valid "
                        f".trit container provided at {self.trit_path}"
                    )

                for e_id in needed:
                    expert = load_trit_layer(self.trit_path, expert_idx=int(e_id))
                    self.experts[int(e_id)] = expert
                    if hasattr(self.experts, "move_to_end"):
                        self.experts.move_to_end(int(e_id))
                    else:
                        if int(e_id) in self.lru_order:
                            self.lru_order.remove(int(e_id))
                        self.lru_order.append(int(e_id))

            # Refresh LRU recency for active hits
            for e_id in active_set:
                if e_id in self.experts:
                    if hasattr(self.experts, "move_to_end"):
                        self.experts.move_to_end(e_id)
                    elif hasattr(self, "lru_order") and e_id in self.lru_order:
                        self.lru_order.remove(e_id)
                        self.lru_order.append(e_id)

            # Evict down to target cache size, BUT NEVER evict any expert in the current active_set (C-5)
            target_capacity = max(self.max_cached_experts, len(active_set))
            if hasattr(self.experts, "keys"):
                candidates = [k for k in list(self.experts.keys()) if k not in active_set]
                while len(self.experts) > target_capacity and candidates:
                    evict_id = candidates.pop(0)
                    if evict_id in self.experts:
                        del self.experts[evict_id]
                    if hasattr(self, "lru_order") and evict_id in self.lru_order:
                        self.lru_order.remove(evict_id)

            return {e: self.experts[e] for e in active_expert_ids if e in self.experts}

    def forward(
        self,
        x: np.ndarray,
        residual: bool = True,
        collect_metrics: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, DispatchMetrics]]:
        """
        Forward execution across streaming expert lanes.
        """
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1]).astype(np.float32)

        # 1. Normalization
        normed_x = rmsnorm(x_2d, self.norm_weight, zero_centered=True) if self.norm_weight is not None else x_2d

        # 2. Gatekeeper top-k routing
        topk_indices, topk_probs = self.router.forward(normed_x, top_k=self.num_experts_per_tok)
        active_ids = np.unique(topk_indices.ravel())

        # 3. Ensure active experts are available (seek-loaded if streaming)
        active_expert_dict = self._ensure_experts_loaded(active_ids)

        # 4. Dispatch through batch permutation engine
        dispatcher = MoEBatchDispatcher(
            router=self.router,
            experts=active_expert_dict,
            shared_expert=self.shared_expert,
            norm_weight=None,  # Already applied above
            num_experts=self.num_experts,
            num_experts_per_tok=self.num_experts_per_tok,
        )

        out, metrics = dispatcher.dispatch_and_execute(
            normed_x,
            residual=False,
            collect_metrics=True,
        )

        # Add residual connection
        final_out = (x_2d + out if residual else out).reshape(orig_shape)

        if collect_metrics:
            return final_out, metrics
        return final_out


class AsymmetricTransformerBlock:
    """
    Single Asymmetric Transformer Layer.
    Combines:
    1. Input RMSNorm + Gatekeeper Attention + Residual Connection.
    2. Post-Attention RMSNorm + Asymmetric MoE (Gatekeeper Router + Ternary Lanes) + Residual.
    """
    def __init__(
        self,
        layer_idx: int,
        config: TritForgeConfig,
        input_layernorm_weight: np.ndarray,
        self_attn: AsymmetricAttention,
        post_attention_layernorm_weight: np.ndarray,
        moe: Union[StreamingMoELayer, MoEBatchDispatcher, Layer0AsymmetricMoE],
    ):
        self.layer_idx = layer_idx
        self.config = config
        self.input_layernorm_weight = np.asarray(input_layernorm_weight, dtype=np.float32)
        self.self_attn = self_attn
        self.post_attention_layernorm_weight = np.asarray(post_attention_layernorm_weight, dtype=np.float32)
        self.moe = moe

    def forward(
        self,
        x: np.ndarray,
        causal_mask: bool = True,
        kv_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        collect_metrics: bool = False,
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray], Optional[Dict[str, Any]]]:
        """
        Forward pass through the transformer block.

        Returns:
            h: Output tensor of shape (B, S, D).
            new_kv_cache: Updated (K, V) cache for this layer.
            telemetry: Optional dictionary of routing/execution metrics.
        """
        # 1. Attention Block with Residual
        normed_attn_in = rmsnorm(x, self.input_layernorm_weight, eps=self.config.rms_norm_eps, zero_centered=True)
        attn_out, new_kv_cache = self.self_attn.forward(
            normed_attn_in,
            causal_mask=causal_mask,
            kv_cache=kv_cache,
        )
        h = x + attn_out

        # 2. MoE Block with Residual
        normed_moe_in = rmsnorm(h, self.post_attention_layernorm_weight, eps=self.config.rms_norm_eps, zero_centered=True)

        telemetry: Optional[Dict[str, Any]] = None
        if isinstance(self.moe, StreamingMoELayer):
            if collect_metrics:
                moe_out, metrics = self.moe.forward(normed_moe_in, residual=False, collect_metrics=True)
                telemetry = asdict(metrics)
            else:
                moe_out = self.moe.forward(normed_moe_in, residual=False, collect_metrics=False)
        elif isinstance(self.moe, MoEBatchDispatcher):
            if collect_metrics:
                moe_out, metrics = self.moe.dispatch_and_execute(normed_moe_in, residual=False, collect_metrics=True)
                telemetry = asdict(metrics)
            else:
                moe_out = self.moe.dispatch_and_execute(normed_moe_in, residual=False, collect_metrics=False)
        else:
            # Layer0AsymmetricMoE
            moe_out, r_info = self.moe.forward(normed_moe_in, residual=False, return_routing_info=True)
            if collect_metrics:
                telemetry = r_info

        out = h + moe_out
        return out, new_kv_cache, telemetry


@dataclass
class GenerationResult:
    """Telemetry and generated tokens resulting from autoregressive generation."""
    sequences: np.ndarray  # Shape (B, total_seq_len)
    generated_tokens: List[List[int]]  # Shape [B, generated_len]
    per_step_latencies_ms: List[float]
    total_time_ms: float
    tokens_per_second: float
    routing_telemetry: List[Dict[str, Any]]


class TritForgeModel:
    """
    End-to-End Asymmetric Mixed-Precision MoE Inference Engine.

    Pipelined structure:
    Embedding -> [Layer 0 ... Layer L-1] -> Final RMSNorm -> LM Head.
    Supports in-memory and streaming .trit container execution.
    """
    def __init__(
        self,
        config: TritForgeConfig,
        embed_tokens: np.ndarray,
        layers: List[AsymmetricTransformerBlock],
        norm: np.ndarray,
        lm_head: Optional[np.ndarray] = None,
    ):
        self.config = config
        self.embed_tokens = np.asarray(embed_tokens, dtype=np.float32)
        self.layers = layers
        self.norm = np.asarray(norm, dtype=np.float32)
        self.lm_head = np.asarray(lm_head, dtype=np.float32) if lm_head is not None else None

    def forward(
        self,
        input_ids: Union[np.ndarray, Sequence[int], Sequence[Sequence[int]]],
        kv_caches: Optional[List[Optional[Tuple[np.ndarray, np.ndarray]]]] = None,
        collect_metrics: bool = False,
    ) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]], List[Optional[Dict[str, Any]]]]:
        """
        Forward pass of the full model.

        Args:
            input_ids: Integer token array of shape (B, S) or (S,).
            kv_caches: Optional list of (K, V) tuples per layer for incremental decoding.
            collect_metrics: Whether to collect MoE routing telemetry across layers.

        Returns:
            logits: Logits tensor of shape (B, S, vocab_size).
            new_kv_caches: List of updated (K, V) cache tuples per layer.
            layer_telemetries: List of routing telemetry dictionaries per layer.
        """
        ids = np.asarray(input_ids, dtype=np.int64)
        if ids.ndim == 1:
            ids = ids[None, :]  # Shape (1, S)
        b_sz, seq_len = ids.shape

        # 1. Embedding Lookup: (B, S, hidden_size)
        h = self.embed_tokens[ids]

        if kv_caches is None:
            kv_caches = [None] * len(self.layers)

        new_kv_caches = []
        layer_telemetries = []

        # 2. Multi-Layer Transformer Stack
        for idx, layer in enumerate(self.layers):
            layer_cache = kv_caches[idx] if idx < len(kv_caches) else None
            h, updated_cache, telemetry = layer.forward(
                h,
                causal_mask=True,
                kv_cache=layer_cache,
                collect_metrics=collect_metrics,
            )
            new_kv_caches.append(updated_cache)
            layer_telemetries.append(telemetry)

        # 3. Final RMSNorm
        h = rmsnorm(h, self.norm, eps=self.config.rms_norm_eps, zero_centered=True)

        # 4. LM Head Projection: (B, S, vocab_size)
        if self.lm_head is not None:
            logits = np.matmul(h, self.lm_head.T)
        else:
            # Tied word embeddings
            logits = np.matmul(h, self.embed_tokens.T)

        return logits, new_kv_caches, layer_telemetries

    def generate(
        self,
        prompt_token_ids: Union[np.ndarray, Sequence[int], Sequence[Sequence[int]]],
        max_new_tokens: int = 20,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 50,
        eos_token_id: Optional[int] = None,
        seed: Optional[int] = 42,
    ) -> GenerationResult:
        """
        Autoregressive generation loop with greedy decoding or temperature/top-p sampling.

        Args:
            prompt_token_ids: Input prompt tokens of shape (B, S) or (S,).
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature. If <= 1e-5, performs deterministic greedy decoding.
            top_p: Nucleus sampling probability cutoff (0.0 to 1.0).
            top_k: Top-k filtering threshold.
            eos_token_id: Optional end-of-sequence token ID to halt generation early.
            seed: Optional random seed for reproducible sampling.

        Returns:
            GenerationResult containing generated sequences and execution telemetry.
        """
        rng = np.random.RandomState(seed)
        ids = np.asarray(prompt_token_ids, dtype=np.int64)
        if ids.ndim == 1:
            ids = ids[None, :]
        b_sz, prompt_len = ids.shape

        # Enforce security ceiling on generation token count to prevent unbounded memory allocation
        max_new_tokens = max(1, min(int(max_new_tokens), 4096))

        current_ids = ids.copy()
        generated_list: List[List[int]] = [[] for _ in range(b_sz)]
        step_latencies: List[float] = []
        all_telemetries: List[Dict[str, Any]] = []

        kv_caches: Optional[List[Optional[Tuple[np.ndarray, np.ndarray]]]] = None
        t_start_total = time.perf_counter()

        for step in range(max_new_tokens):
            t_step_0 = time.perf_counter()

            # For step 0, pass all prompt tokens to populate initial KV cache.
            # For step > 0, pass only the newly generated single token.
            step_input = current_ids if step == 0 else current_ids[:, -1:]

            logits, kv_caches, telemetries = self.forward(
                step_input,
                kv_caches=kv_caches,
                collect_metrics=(step == 0),
            )
            if step == 0 and telemetries:
                all_telemetries.extend([t for t in telemetries if t is not None])

            # Extract logits for the final token position: shape (B, V)
            next_token_logits = logits[:, -1, :].astype(np.float64)

            # Sampling / Greedy Selection
            if temperature <= 1e-5 or top_p <= 1e-5:
                # Deterministic Greedy Decoding
                next_tokens = np.argmax(next_token_logits, axis=-1)  # Shape (B,)
            else:
                # Temperature Scaling
                scaled_logits = next_token_logits / max(temperature, 1e-5)

                next_tokens = np.zeros(b_sz, dtype=np.int64)
                for b in range(b_sz):
                    row = scaled_logits[b].copy()

                    # Top-K Filtering
                    if 0 < top_k < len(row):
                        topk_threshold = np.partition(row, -top_k)[-top_k]
                        row[row < topk_threshold] = -np.inf

                    # Softmax with Numerical Stability
                    max_logit = np.max(row)
                    if np.isneginf(max_logit):
                        row = next_token_logits[b]
                        max_logit = np.max(row)
                    exp_logits = np.exp(row - max_logit)
                    probs = exp_logits / np.sum(exp_logits)

                    # Top-P (Nucleus) Filtering
                    if 0.0 < top_p < 1.0:
                        sorted_indices = np.argsort(probs)[::-1]
                        sorted_probs = probs[sorted_indices]
                        cumulative_probs = np.cumsum(sorted_probs)

                        # Remove tokens with cumulative probability above threshold (keep first above)
                        cutoff_mask = cumulative_probs > top_p
                        cutoff_mask[0] = False  # Always retain top-1
                        probs_filtered = sorted_probs.copy()
                        probs_filtered[cutoff_mask] = 0.0
                        norm_sum = np.sum(probs_filtered)
                        if norm_sum > 0:
                            probs_filtered /= norm_sum
                        else:
                            probs_filtered[0] = 1.0

                        # Sample from nucleus
                        sampled_idx = rng.choice(len(probs_filtered), p=probs_filtered)
                        next_tokens[b] = sorted_indices[sampled_idx]
                    else:
                        next_tokens[b] = rng.choice(len(probs), p=probs)

            # Append to generated sequence
            current_ids = np.concatenate([current_ids, next_tokens[:, None]], axis=1)
            for b in range(b_sz):
                generated_list[b].append(int(next_tokens[b]))

            t_step_1 = time.perf_counter()
            step_latencies.append((t_step_1 - t_step_0) * 1000.0)

            # Check EOS condition
            if eos_token_id is not None and all(tok == eos_token_id for tok in next_tokens):
                break

        t_end_total = time.perf_counter()
        total_time_ms = (t_end_total - t_start_total) * 1000.0
        total_gen_tokens = sum(len(g) for g in generated_list)
        tokens_per_sec = (total_gen_tokens / (total_time_ms / 1000.0)) if total_time_ms > 0 else 0.0

        return GenerationResult(
            sequences=current_ids,
            generated_tokens=generated_list,
            per_step_latencies_ms=step_latencies,
            total_time_ms=total_time_ms,
            tokens_per_second=tokens_per_sec,
            routing_telemetry=all_telemetries,
        )


def create_dummy_asymmetric_model(
    vocab_size: int = 1000,
    hidden_size: int = 128,
    intermediate_size: int = 64,
    num_layers: int = 2,
    num_heads: int = 4,
    num_kv_heads: int = 2,
    num_experts: int = 8,
    num_experts_per_tok: int = 2,
    block_size: int = 32,
    seed: int = 42,
) -> TritForgeModel:
    """
    Factory helper to create a verified, lightweight in-memory Asymmetric MoE model
    for zero-cost automated unit testing and numerical verification.
    """
    from tritforge.quantize import absmax_ternary_quantize

    rng = np.random.RandomState(seed)
    config = TritForgeConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        shared_expert_intermediate_size=intermediate_size,
        block_size=block_size,
    )

    embed_tokens = rng.randn(vocab_size, hidden_size).astype(np.float32) * 0.02
    norm = np.ones(hidden_size, dtype=np.float32)
    lm_head = rng.randn(vocab_size, hidden_size).astype(np.float32) * 0.02

    layers: List[AsymmetricTransformerBlock] = []
    head_dim = hidden_size // num_heads
    kv_dim = num_kv_heads * head_dim

    for layer_idx in range(num_layers):
        # Attention Projections
        q_p = rng.randn(hidden_size, hidden_size).astype(np.float32) * 0.02
        k_p = rng.randn(kv_dim, hidden_size).astype(np.float32) * 0.02
        v_p = rng.randn(kv_dim, hidden_size).astype(np.float32) * 0.02
        o_p = rng.randn(hidden_size, hidden_size).astype(np.float32) * 0.02
        attn = AsymmetricAttention(config, q_p, k_p, v_p, o_p)

        # Gatekeeper Router
        gate_w = rng.randn(num_experts, hidden_size).astype(np.float32) * 0.01
        router = MoERouter(gate_w, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok)

        # Shared Expert
        shared = SharedExpert(
            gate_proj=rng.randn(intermediate_size, hidden_size).astype(np.float32) * 0.01,
            up_proj=rng.randn(intermediate_size, hidden_size).astype(np.float32) * 0.01,
            down_proj=rng.randn(hidden_size, intermediate_size).astype(np.float32) * 0.01,
        )

        # Quantized Experts
        experts: Dict[int, TernaryExpert] = {}
        for e_id in range(num_experts):
            w_g = rng.randn(intermediate_size, hidden_size).astype(np.float32) * 0.01
            w_u = rng.randn(intermediate_size, hidden_size).astype(np.float32) * 0.01
            w_d = rng.randn(hidden_size, intermediate_size).astype(np.float32) * 0.01
            experts[e_id] = TernaryExpert(
                expert_idx=e_id,
                gate_proj=absmax_ternary_quantize(w_g, scaling_mode="block", block_size=block_size),
                up_proj=absmax_ternary_quantize(w_u, scaling_mode="block", block_size=block_size),
                down_proj=absmax_ternary_quantize(w_d, scaling_mode="block", block_size=block_size),
            )

        dispatcher = MoEBatchDispatcher(
            router=router,
            experts=experts,
            shared_expert=shared,
            norm_weight=None,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
        )

        block = AsymmetricTransformerBlock(
            layer_idx=layer_idx,
            config=config,
            input_layernorm_weight=np.ones(hidden_size, dtype=np.float32),
            self_attn=attn,
            post_attention_layernorm_weight=np.ones(hidden_size, dtype=np.float32),
            moe=dispatcher,
        )
        layers.append(block)

    return TritForgeModel(
        config=config,
        embed_tokens=embed_tokens,
        layers=layers,
        norm=norm,
        lm_head=lm_head,
    )
