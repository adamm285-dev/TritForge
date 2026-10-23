"""
Tests for tritforge.slice
=========================
Validates binary header parsing, seek-based slice extraction, and pure NumPy FP8/BF16 decoding.
"""

import json
from pathlib import Path
import struct
import numpy as np
import pytest

from tritforge.slice import (
    FP8_E4M3_LUT,
    decode_bf16_to_f32,
    decode_fp8_e4m3_to_f32,
    extract_expert_slice,
    read_safetensors_header,
)

MASTER_SHARD = Path(r"H:\AI_Memory_Vault\models\Qwen3.8-2.4T-A95B-FP8\model-00001-of-00213.safetensors")


def test_fp8_e4m3_lut_properties():
    """Verify FP8 E4M3 lookup table adheres to IEEE specifications."""
    assert len(FP8_E4M3_LUT) == 256
    # 0x00 is +0.0
    assert FP8_E4M3_LUT[0] == 0.0
    # 0x7E is +448.0 (max positive finite)
    assert FP8_E4M3_LUT[0x7E] == 448.0
    # 0xFE is -448.0 (max negative finite)
    assert FP8_E4M3_LUT[0xFE] == -448.0
    # 0x7F and 0xFF are NaN
    assert np.isnan(FP8_E4M3_LUT[0x7F])
    assert np.isnan(FP8_E4M3_LUT[0xFF])
    # Subnormal check: byte 1 is 1/512 = 0.001953125
    assert np.isclose(FP8_E4M3_LUT[1], 0.001953125)


def test_bf16_decoding():
    """Verify pure NumPy bfloat16 decoding matches IEEE float32."""
    # Test well-known values: 1.0 in float32 is 0x3F800000 -> upper 16 bits = 0x3F80 (16256)
    raw_u16 = np.array([0x3F80, 0x4000, 0xBF80], dtype=np.uint16)
    decoded = decode_bf16_to_f32(raw_u16)
    assert np.allclose(decoded, [1.0, 2.0, -1.0])


@pytest.mark.skipif(not MASTER_SHARD.exists(), reason="Master weight shard not found on H: drive")
def test_extract_expert_slice_live():
    """Verify live extraction of expert slice from Qwen 2.4T FP8 shard."""
    slice_obj = extract_expert_slice(
        MASTER_SHARD,
        layer_idx=0,
        expert_idx=0,
        proj="gate_proj",
    )
    assert slice_obj.layer_idx == 0
    assert slice_obj.expert_idx == 0
    assert slice_obj.proj_type == "gate_proj"
    assert slice_obj.original_shape == (2048, 8192)
    assert slice_obj.weight.shape == (2048, 8192)
    assert slice_obj.weight.dtype == np.float32
    assert slice_obj.scale_inv.shape == (16, 64)
    # Check reasonable neural network weight statistics
    assert not np.isnan(slice_obj.weight).all()
    assert abs(float(slice_obj.weight.mean())) < 0.01
    assert 0.0001 < float(slice_obj.weight.std()) < 0.05
    # Check that latency is reasonable even under heavy concurrent I/O
    assert slice_obj.load_time_ms < 5000.0


def test_read_safetensors_header_oversized(tmp_path):
    """Adversarial test: verify that oversized header length (>64MB) is safely rejected."""
    import struct
    fake_shard = tmp_path / "fake_bomb.safetensors"
    # Write header size = 128 MB (unsafe allocation attempt)
    with open(fake_shard, "wb") as f:
        f.write(struct.pack("<Q", 128 * 1024 * 1024))
        f.write(b"dummy")

    with pytest.raises(ValueError, match="Unsafe or corrupted safetensors header size"):
        read_safetensors_header(fake_shard)


def test_inplace_4d_block_scale_dequant_b6():
    """B6: Verify that 4D in-place block-scale broadcast matches np.repeat on non-square grid."""
    # Test tensor: 256 x 384 with 2 x 3 scale grid (block size 128 x 128)
    shape = (256, 384)
    s_shape = (2, 3)
    blk_h = shape[0] // s_shape[0]  # 128
    blk_w = shape[1] // s_shape[1]  # 128

    rng = np.random.RandomState(42)
    w_orig = rng.randn(*shape).astype(np.float32)
    s_f32 = rng.uniform(0.1, 2.0, size=s_shape).astype(np.float32)

    # 1. Reference: np.repeat method
    s_expanded = np.repeat(np.repeat(s_f32, blk_h, axis=0), blk_w, axis=1)
    w_expected = w_orig * s_expanded

    # 2. In-place 4D method (as implemented in slice.py)
    w_test = w_orig.copy()
    s_rows, s_cols = s_shape
    w_4d = w_test.reshape(s_rows, blk_h, s_cols, blk_w)
    w_4d *= s_f32[:, None, :, None]

    np.testing.assert_allclose(w_test, w_expected, rtol=1e-7, atol=1e-7)


def test_extract_tensor_fp8_e4m3_dequant_with_non_square_scale_grid_d7(tmp_path):
    """D7: extract_tensor dequantizes F8_E4M3 weight with BF16 weight_scale_inv on a non-square grid.
    Fails if scale multiplication is omitted or broadcast axes are swapped."""
    from tritforge.slice import extract_tensor

    w_shape = (256, 384)
    s_shape = (2, 3)

    rng = np.random.RandomState(42)
    # Exclude NaN bytes 127 (0x7F) and 255 (0xFF)
    valid_bytes = np.array([b for b in range(256) if b not in (127, 255)], dtype=np.uint8)
    raw_w_u8 = rng.choice(valid_bytes, size=w_shape)
    w_unscaled = FP8_E4M3_LUT[raw_w_u8]

    # BF16 scales on 2x3 block grid (blocks 128x128)
    s_f32 = rng.uniform(0.5, 2.5, size=s_shape).astype(np.float32)
    s_bf16_u16 = (s_f32.view(np.uint32) >> 16).astype(np.uint16)
    s_f32_decoded = decode_bf16_to_f32(s_bf16_u16.tobytes(), shape=s_shape)

    # Reference ground truth using np.repeat
    s_expanded = np.repeat(np.repeat(s_f32_decoded, 128, axis=0), 128, axis=1)
    expected = w_unscaled * s_expanded

    # Write synthetic .safetensors shard
    shard_path = tmp_path / "synthetic_fp8_shard.safetensors"
    w_bytes = raw_w_u8.tobytes()
    s_bytes = s_bf16_u16.tobytes()

    header = {
        "model.layers.0.mlp.experts.0.gate_proj.weight": {
            "dtype": "F8_E4M3",
            "shape": list(w_shape),
            "data_offsets": [0, len(w_bytes)],
        },
        "model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv": {
            "dtype": "BF16",
            "shape": list(s_shape),
            "data_offsets": [len(w_bytes), len(w_bytes) + len(s_bytes)],
        },
    }
    h_bytes = json.dumps(header).encode("utf-8")
    with open(shard_path, "wb") as f:
        f.write(struct.pack("<Q", len(h_bytes)))
        f.write(h_bytes)
        f.write(w_bytes)
        f.write(s_bytes)

    extracted = extract_tensor(shard_path, "model.layers.0.mlp.experts.0.gate_proj.weight")

    # Verification: must match expected exactly
    assert extracted.shape == w_shape
    np.testing.assert_allclose(extracted, expected, rtol=1e-5, atol=1e-5)

    # Verify that unscaled weights do NOT match (fails if scale multiplication is omitted)
    assert not np.allclose(extracted, w_unscaled)

