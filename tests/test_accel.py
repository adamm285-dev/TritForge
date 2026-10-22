"""
Tests for tritforge.accel (Phase 6)
===================================
Validates hybrid execution backend: GPU Attention offload (PyTorch),
CPU Zero-Multiply Expert execution, and infallible pure NumPy fallback.
"""

import numpy as np
import pytest

from tritforge.accel import (
    DeviceConfig,
    HybridAttention,
    HybridTransformerBlock,
    HybridTritForgeModel,
    create_hybrid_dummy_model,
    is_cuda_available,
    is_torch_available,
)
from tritforge.model import AsymmetricAttention, TritForgeConfig


def test_device_config_resolution():
    """Verify hardware configuration resolves correctly based on runtime availability."""
    cfg_auto = DeviceConfig(device="auto")
    resolved = cfg_auto.resolve_device()
    assert resolved in ("cuda", "cpu", "pure_numpy")

    cfg_numpy = DeviceConfig(device="pure_numpy")
    assert cfg_numpy.resolve_device() == "pure_numpy"


def test_hybrid_attention_torch_and_numpy_identity():
    """Verify HybridAttention produces numerically identical outputs between Torch and pure NumPy."""
    rng = np.random.RandomState(42)
    b_sz, seq_len = 2, 4
    hidden_size = 64
    num_heads, num_kv_heads = 4, 2
    head_dim = hidden_size // num_heads
    kv_dim = num_kv_heads * head_dim

    cfg = TritForgeConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
    )

    qp = rng.randn(hidden_size, hidden_size).astype(np.float32) * 0.02
    kp = rng.randn(kv_dim, hidden_size).astype(np.float32) * 0.02
    vp = rng.randn(kv_dim, hidden_size).astype(np.float32) * 0.02
    op = rng.randn(hidden_size, hidden_size).astype(np.float32) * 0.02

    x = rng.randn(b_sz, seq_len, hidden_size).astype(np.float32)

    # 1. Pure NumPy Reference
    ref_attn = AsymmetricAttention(cfg, qp, kp, vp, op)
    ref_out, _ = ref_attn.forward(x, causal_mask=True)

    # 2. Hybrid Attention (Pure NumPy mode)
    hybrid_numpy = HybridAttention(cfg, qp, kp, vp, op, device_config=DeviceConfig(device="pure_numpy"))
    np_out, _ = hybrid_numpy.forward(x, causal_mask=True)
    assert np.allclose(ref_out, np_out, atol=1e-5)

    # 3. Hybrid Attention (Torch mode if torch available)
    if is_torch_available():
        hybrid_torch = HybridAttention(cfg, qp, kp, vp, op, device_config=DeviceConfig(device="cpu", dtype="float32"))
        torch_out, _ = hybrid_torch.forward(x, causal_mask=True)
        assert np.allclose(ref_out, torch_out, atol=1e-4)


def test_hybrid_transformer_block():
    """Verify HybridTransformerBlock forward pass with hybrid Attention + CPU MoE."""
    model = create_hybrid_dummy_model(
        vocab_size=100,
        hidden_size=32,
        intermediate_size=16,
        num_layers=1,
        device_config=DeviceConfig(device="auto"),
    )
    block = model.layers[0]

    x = np.random.randn(2, 3, 32).astype(np.float32)
    out, kv, telemetry = block.forward(x, causal_mask=True, collect_metrics=True)

    assert out.shape == (2, 3, 32)
    assert np.isfinite(out).all()
    assert kv is not None


def test_hybrid_model_generation():
    """Verify end-to-end autoregressive generation under hybrid execution."""
    dev_mode = "cpu" if is_torch_available() else "pure_numpy"
    model = create_hybrid_dummy_model(
        vocab_size=100,
        hidden_size=32,
        intermediate_size=16,
        num_layers=2,
        device_config=DeviceConfig(device=dev_mode),
    )

    prompt = [5, 10, 15]
    result = model.generate(prompt, max_new_tokens=4, temperature=0.0)

    assert result.sequences.shape == (1, 3 + 4)
    assert len(result.generated_tokens[0]) == 4
    assert result.tokens_per_second > 0
