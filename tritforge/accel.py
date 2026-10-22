"""
TritForge Hybrid Hardware Acceleration Engine (tritforge.accel)
================================================================
Hybrid execution backend designed for consumer workstations with moderate GPU VRAM
(e.g., 12 GB RTX 3080 Ti / RTX 4070) paired with high system RAM (64 GB - 256 GB):

Hybrid Offload Architecture:
1. Gatekeeper Attention on GPU:
   - Offloads Grouped-Query Attention (GQA), causal masking, and KV caches to GPU VRAM
     (utilizing PyTorch / CUDA when available) for maximum token generation throughput.
2. Streaming Routed Experts on CPU:
   - Keeps 1.58-bit ternary expert lanes in system CPU RAM, executing via our Zero-Multiply
     addition/subtraction GEMM engine without consuming GPU VRAM.
3. Zero-Failure CPU Fallback:
   - Automatically and gracefully falls back to 100% pure NumPy CPU execution if PyTorch
     or CUDA is not installed or detected.

Enforces:
- Invariant 1: Zero mandatory external dependencies (graceful optional acceleration).
- Invariant 2: Mathematical Asymmetry across heterogeneous hardware tiers.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from tritforge.dispatcher import DispatchMetrics, MoEBatchDispatcher
from tritforge.model import (
    AsymmetricAttention,
    AsymmetricTransformerBlock,
    GenerationResult,
    StreamingMoELayer,
    TritForgeConfig,
    TritForgeModel,
)
from tritforge.moe import rmsnorm


def is_torch_available() -> bool:
    """Returns True if PyTorch is installed in the current environment."""
    try:
        import torch
        return True
    except ImportError:
        return False


def is_cuda_available() -> bool:
    """Returns True if PyTorch is installed and at least one CUDA GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except (ImportError, Exception):
        return False


@dataclass
class DeviceConfig:
    """Hardware target and offload configuration for hybrid execution."""
    device: str = "auto"  # "auto", "cuda", "cpu", or "pure_numpy"
    vram_budget_gb: float = 12.0
    dtype: str = "float16"  # "float16", "bfloat16", or "float32"
    offload_attention: bool = True
    offload_router: bool = False

    def resolve_device(self) -> str:
        """Resolves target execution device based on environment telemetry."""
        if self.device == "pure_numpy":
            return "pure_numpy"
        if self.device == "cuda":
            return "cuda" if is_cuda_available() else "cpu"
        if self.device == "cpu":
            return "cpu" if is_torch_available() else "pure_numpy"
        # "auto" mode: prefer cuda if available, else cpu if torch available, else pure_numpy
        if is_cuda_available():
            return "cuda"
        elif is_torch_available():
            return "cpu"
        return "pure_numpy"


class HybridAttention:
    """
    Hybrid Gatekeeper Attention module supporting:
    - GPU offload (PyTorch CUDA / CPU tensors) when PyTorch is available.
    - 100% pure NumPy fallback when PyTorch is absent or pure_numpy mode is selected.
    """
    def __init__(
        self,
        config: TritForgeConfig,
        q_proj: np.ndarray,
        k_proj: np.ndarray,
        v_proj: np.ndarray,
        o_proj: np.ndarray,
        device_config: Optional[DeviceConfig] = None,
    ):
        self.config = config
        self.dev_cfg = device_config or DeviceConfig()
        self.resolved_device = self.dev_cfg.resolve_device()

        # Always maintain reference pure-NumPy attention as infallible baseline
        self.numpy_attn = AsymmetricAttention(config, q_proj, k_proj, v_proj, o_proj)
        self.torch_initialized = False

        if self.resolved_device in ("cuda", "cpu") and self.dev_cfg.offload_attention and is_torch_available():
            try:
                import torch

                t_dtype = (
                    torch.float16 if self.dev_cfg.dtype == "float16"
                    else (torch.bfloat16 if self.dev_cfg.dtype == "bfloat16" else torch.float32)
                )
                self.torch_device = torch.device(self.resolved_device)
                self.t_dtype = t_dtype

                self.q_proj_t = torch.from_numpy(q_proj).to(dtype=t_dtype, device=self.torch_device)
                self.k_proj_t = torch.from_numpy(k_proj).to(dtype=t_dtype, device=self.torch_device)
                self.v_proj_t = torch.from_numpy(v_proj).to(dtype=t_dtype, device=self.torch_device)
                self.o_proj_t = torch.from_numpy(o_proj).to(dtype=t_dtype, device=self.torch_device)
                self.scale = 1.0 / np.sqrt(float(config.head_dim))
                self.torch_initialized = True
            except Exception:
                # Graceful fallback to pure NumPy on any torch initialization error
                self.torch_initialized = False
                self.resolved_device = "pure_numpy"

    def forward(
        self,
        x: np.ndarray,
        causal_mask: bool = True,
        kv_cache: Optional[Any] = None,
    ) -> Tuple[np.ndarray, Any]:
        """
        Executes Attention on GPU / Torch CPU if active, or pure NumPy fallback.

        Args:
            x: Input activation NumPy array of shape (B, S, D).
            causal_mask: Whether to apply lower-triangular causal attention mask.
            kv_cache: Past key/value states.

        Returns:
            out: Attention output NumPy array of shape (B, S, D).
            new_kv_cache: Updated KV cache.
        """
        if not self.torch_initialized or self.resolved_device == "pure_numpy":
            return self.numpy_attn.forward(x, causal_mask=causal_mask, kv_cache=kv_cache)

        import torch

        b_sz, seq_len, _ = x.shape
        x_t = torch.from_numpy(x).to(dtype=self.t_dtype, device=self.torch_device)

        with torch.no_grad():
            # Linear projections
            q = torch.matmul(x_t, self.q_proj_t.t())
            k = torch.matmul(x_t, self.k_proj_t.t())
            v = torch.matmul(x_t, self.v_proj_t.t())

            # Reshape to (B, num_heads, S, head_dim)
            num_h = self.config.num_attention_heads
            num_kv_h = self.config.num_key_value_heads
            h_dim = self.config.head_dim

            q = q.view(b_sz, seq_len, num_h, h_dim).transpose(1, 2)
            k = k.view(b_sz, seq_len, num_kv_h, h_dim).transpose(1, 2)
            v = v.view(b_sz, seq_len, num_kv_h, h_dim).transpose(1, 2)

            if kv_cache is not None and isinstance(kv_cache, tuple) and len(kv_cache) == 2:
                past_k, past_v = kv_cache
                if isinstance(past_k, torch.Tensor) and isinstance(past_v, torch.Tensor):
                    k = torch.cat([past_k, k], dim=2)
                    v = torch.cat([past_v, v], dim=2)
                elif isinstance(past_k, np.ndarray) and isinstance(past_v, np.ndarray):
                    k_t = torch.from_numpy(past_k).to(dtype=self.t_dtype, device=self.torch_device)
                    v_t = torch.from_numpy(past_v).to(dtype=self.t_dtype, device=self.torch_device)
                    k = torch.cat([k_t, k], dim=2)
                    v = torch.cat([v_t, v], dim=2)

            new_kv_cache = (k, v)
            total_kv_len = k.shape[2]

            # GQA Head Expansion if num_kv_heads < num_heads
            if num_kv_h < num_h:
                repeats = num_h // num_kv_h
                k_expanded = k.repeat_interleave(repeats, dim=1)
                v_expanded = v.repeat_interleave(repeats, dim=1)
            else:
                k_expanded = k
                v_expanded = v

            # Scaled dot-product attention
            scores = torch.matmul(q, k_expanded.transpose(-1, -2)) * self.scale

            if causal_mask and seq_len > 1:
                past_len = total_kv_len - seq_len
                q_idx = torch.arange(seq_len, device=self.torch_device)[:, None] + past_len
                k_idx = torch.arange(total_kv_len, device=self.torch_device)[None, :]
                mask = q_idx < k_idx
                mask_val = -65000.0 if self.t_dtype == torch.float16 else -1e9
                scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), mask_val)

            attn_probs = torch.softmax(scores, dim=-1)
            context = torch.matmul(attn_probs, v_expanded)

            context = context.transpose(1, 2).contiguous().view(b_sz, seq_len, self.config.hidden_size)
            out_t = torch.matmul(context, self.o_proj_t.t())

            # Return NumPy array to preserve unified interface
            out_np = out_t.to(torch.float32).cpu().numpy()
            return out_np, new_kv_cache


class HybridTransformerBlock:
    """
    Hybrid Transformer Layer executing Gatekeeper Attention via HybridAttention
    (with GPU VRAM offload if available) while routing MoE Expert lanes in CPU RAM.
    """
    def __init__(
        self,
        layer_idx: int,
        config: TritForgeConfig,
        input_layernorm_weight: np.ndarray,
        hybrid_attn: HybridAttention,
        post_attention_layernorm_weight: np.ndarray,
        moe: Union[StreamingMoELayer, MoEBatchDispatcher],
    ):
        self.layer_idx = layer_idx
        self.config = config
        self.input_layernorm_weight = np.asarray(input_layernorm_weight, dtype=np.float32)
        self.hybrid_attn = hybrid_attn
        self.post_attention_layernorm_weight = np.asarray(post_attention_layernorm_weight, dtype=np.float32)
        self.moe = moe

    def forward(
        self,
        x: np.ndarray,
        causal_mask: bool = True,
        kv_cache: Optional[Any] = None,
        collect_metrics: bool = False,
    ) -> Tuple[np.ndarray, Any, Optional[Dict[str, Any]]]:
        """
        Executes hybrid transformer forward pass.
        """
        # 1. Attention pathway (GPU offloaded if available)
        normed_attn_in = rmsnorm(x, self.input_layernorm_weight, eps=self.config.rms_norm_eps)
        attn_out, new_kv_cache = self.hybrid_attn.forward(
            normed_attn_in,
            causal_mask=causal_mask,
            kv_cache=kv_cache,
        )
        h = x + attn_out

        # 2. MoE Expert pathway (CPU Zero-Multiply GEMM)
        normed_moe_in = rmsnorm(h, self.post_attention_layernorm_weight, eps=self.config.rms_norm_eps)

        telemetry: Optional[Dict[str, Any]] = None
        if isinstance(self.moe, StreamingMoELayer):
            if collect_metrics:
                moe_out, metrics = self.moe.forward(normed_moe_in, residual=False, collect_metrics=True)
                telemetry = asdict(metrics)
            else:
                moe_out = self.moe.forward(normed_moe_in, residual=False, collect_metrics=False)
        else:
            if collect_metrics:
                moe_out, metrics = self.moe.dispatch_and_execute(normed_moe_in, residual=False, collect_metrics=True)
                telemetry = asdict(metrics)
            else:
                moe_out = self.moe.dispatch_and_execute(normed_moe_in, residual=False, collect_metrics=False)

        out = h + moe_out
        return out, new_kv_cache, telemetry


class HybridTritForgeModel:
    """
    Hybrid End-to-End Inference Engine orchestrating GPU Attention with CPU Expert Lanes.
    """
    def __init__(
        self,
        config: TritForgeConfig,
        embed_tokens: np.ndarray,
        layers: List[HybridTransformerBlock],
        norm: np.ndarray,
        lm_head: Optional[np.ndarray] = None,
        device_config: Optional[DeviceConfig] = None,
    ):
        self.config = config
        self.embed_tokens = np.asarray(embed_tokens, dtype=np.float32)
        self.layers = layers
        self.norm = np.asarray(norm, dtype=np.float32)
        self.lm_head = np.asarray(lm_head, dtype=np.float32) if lm_head is not None else None
        self.dev_cfg = device_config or DeviceConfig()

    def forward(
        self,
        input_ids: Union[np.ndarray, Sequence[int], Sequence[Sequence[int]]],
        kv_caches: Optional[List[Any]] = None,
        collect_metrics: bool = False,
    ) -> Tuple[np.ndarray, List[Any], List[Optional[Dict[str, Any]]]]:
        ids = np.asarray(input_ids, dtype=np.int64)
        if ids.ndim == 1:
            ids = ids[None, :]

        h = self.embed_tokens[ids]
        if kv_caches is None:
            kv_caches = [None] * len(self.layers)

        new_kv_caches = []
        layer_telemetries = []

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

        h = rmsnorm(h, self.norm, eps=self.config.rms_norm_eps)

        if self.lm_head is not None:
            logits = np.matmul(h, self.lm_head.T)
        else:
            logits = np.matmul(h, self.embed_tokens.T)

        return logits, new_kv_caches, layer_telemetries

    def generate(
        self,
        prompt_token_ids: Sequence[int],
        max_new_tokens: int = 15,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 50,
    ) -> GenerationResult:
        """Autoregressive generation loop under hybrid acceleration."""
        rng = np.random.RandomState(42)
        ids = np.asarray(prompt_token_ids, dtype=np.int64)[None, :]
        current_ids = ids.copy()

        generated: List[int] = []
        latencies: List[float] = []
        kv_caches = None
        t_start = time.perf_counter()

        for step in range(max_new_tokens):
            t0 = time.perf_counter()
            step_in = current_ids if step == 0 else current_ids[:, -1:]
            logits, kv_caches, _ = self.forward(step_in, kv_caches=kv_caches)

            next_logits = logits[0, -1, :].astype(np.float64)

            if temperature <= 1e-5:
                next_tok = int(np.argmax(next_logits))
            else:
                scaled = next_logits / temperature
                probs = np.exp(scaled - np.max(scaled))
                probs /= np.sum(probs)
                next_tok = int(rng.choice(len(probs), p=probs))

            current_ids = np.concatenate([current_ids, np.array([[next_tok]], dtype=np.int64)], axis=1)
            generated.append(next_tok)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

        total_time_ms = (time.perf_counter() - t_start) * 1000.0
        tok_sec = (len(generated) / (total_time_ms / 1000.0)) if total_time_ms > 0 else 0.0

        return GenerationResult(
            sequences=current_ids,
            generated_tokens=[generated],
            per_step_latencies_ms=latencies,
            total_time_ms=total_time_ms,
            tokens_per_second=tok_sec,
            routing_telemetry=[],
        )


def create_hybrid_dummy_model(
    vocab_size: int = 500,
    hidden_size: int = 64,
    intermediate_size: int = 32,
    num_layers: int = 2,
    device_config: Optional[DeviceConfig] = None,
) -> HybridTritForgeModel:
    """Helper constructing verified lightweight Hybrid model for unit testing."""
    from tritforge.moe import MoERouter, SharedExpert, TernaryExpert
    from tritforge.quantize import absmax_ternary_quantize

    rng = np.random.RandomState(42)
    dev_cfg = device_config or DeviceConfig()

    cfg = TritForgeConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        block_size=16,
    )

    embed = rng.randn(vocab_size, hidden_size).astype(np.float32) * 0.02
    norm = np.ones(hidden_size, dtype=np.float32)
    lm_head = rng.randn(vocab_size, hidden_size).astype(np.float32) * 0.02

    layers = []
    for l_idx in range(num_layers):
        qp = rng.randn(hidden_size, hidden_size).astype(np.float32) * 0.02
        kp = rng.randn(cfg.kv_dim, hidden_size).astype(np.float32) * 0.02
        vp = rng.randn(cfg.kv_dim, hidden_size).astype(np.float32) * 0.02
        op = rng.randn(hidden_size, hidden_size).astype(np.float32) * 0.02

        hybrid_attn = HybridAttention(cfg, qp, kp, vp, op, device_config=dev_cfg)

        router = MoERouter(rng.randn(cfg.num_experts, hidden_size).astype(np.float32) * 0.01,
                           num_experts=cfg.num_experts, num_experts_per_tok=cfg.num_experts_per_tok)
        shared = SharedExpert(
            gate_proj=rng.randn(intermediate_size, hidden_size).astype(np.float32) * 0.01,
            up_proj=rng.randn(intermediate_size, hidden_size).astype(np.float32) * 0.01,
            down_proj=rng.randn(hidden_size, intermediate_size).astype(np.float32) * 0.01,
        )

        experts = {}
        for e_id in range(cfg.num_experts):
            wg = rng.randn(intermediate_size, hidden_size).astype(np.float32) * 0.01
            wu = rng.randn(intermediate_size, hidden_size).astype(np.float32) * 0.01
            wd = rng.randn(hidden_size, intermediate_size).astype(np.float32) * 0.01
            experts[e_id] = TernaryExpert(
                e_id,
                absmax_ternary_quantize(wg, scaling_mode="block", block_size=16),
                absmax_ternary_quantize(wu, scaling_mode="block", block_size=16),
                absmax_ternary_quantize(wd, scaling_mode="block", block_size=16),
            )

        disp = MoEBatchDispatcher(router, experts, shared, num_experts=cfg.num_experts, num_experts_per_tok=cfg.num_experts_per_tok)
        block = HybridTransformerBlock(l_idx, cfg, np.ones(hidden_size, dtype=np.float32),
                                       hybrid_attn, np.ones(hidden_size, dtype=np.float32), disp)
        layers.append(block)

    return HybridTritForgeModel(cfg, embed, layers, norm, lm_head, device_config=dev_cfg)
