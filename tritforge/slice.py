"""
TritForge Slice Extractor (tritforge.slice)
===========================================
Isolated, clean-room weight loader that extracts individual expert projection
matrices from massive safetensors shards without reading the entire shard
into memory.

Enforces:
- Invariant 1: Zero external dependencies (Pure Python Standard Library + NumPy).
- Invariant 4: Constant-memory extraction via seek-based byte-range reads.
"""

from __future__ import annotations

import json
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


def _generate_fp8_e4m3fn_lut() -> np.ndarray:
    """
    Generates the exact 256-element IEEE float8_e4m3fn lookup table in pure NumPy.
    Format: 1 sign bit, 4 exponent bits, 3 mantissa bits, bias = 7.
    0x7F and 0xFF are NaN. Max finite value is 448.0.
    """
    lut = np.zeros(256, dtype=np.float32)
    for b in range(256):
        s = (b >> 7) & 1
        e = (b >> 3) & 0x0F
        m = b & 0x07
        sign = -1.0 if s else 1.0

        if e == 0:
            # Subnormal numbers: (-1)^s * 2^(-6) * (m / 8)
            lut[b] = sign * (2.0 ** -6) * (m / 8.0)
        elif e == 15 and m == 7:
            # 0x7F (127) and 0xFF (255) are NaN in E4M3FN
            lut[b] = np.nan
        else:
            # Normal numbers: (-1)^s * 2^(e - 7) * (1 + m / 8)
            lut[b] = sign * (2.0 ** (e - 7)) * (1.0 + m / 8.0)
    return lut


# Global 256-entry lookup table for O(1) vectorized FP8 E4M3 decoding
FP8_E4M3_LUT: np.ndarray = _generate_fp8_e4m3fn_lut()


def decode_bf16_to_f32(raw_bytes: Union[bytes, np.ndarray], shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
    """
    Decodes raw bfloat16 bytes into IEEE float32 using pure NumPy bit operations.
    bfloat16 represents the upper 16 bits of a 32-bit IEEE float.
    """
    if isinstance(raw_bytes, bytes):
        u16 = np.frombuffer(raw_bytes, dtype="<u2")
    elif isinstance(raw_bytes, np.ndarray):
        u16 = raw_bytes.view("<u2")
    else:
        raise TypeError(f"Unsupported input type for bf16 decoding: {type(raw_bytes)}")

    # Shift 16 bits left into float32 representation
    f32 = (u16.astype(np.uint32) << 16).view(np.float32)
    if shape is not None:
        f32 = f32.reshape(shape)
    return f32


def decode_fp8_e4m3_to_f32(raw_bytes: Union[bytes, np.ndarray], shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
    """
    Decodes raw FP8 E4M3 bytes into float32 using vectorized LUT indexing.
    """
    if isinstance(raw_bytes, bytes):
        u8 = np.frombuffer(raw_bytes, dtype=np.uint8)
    elif isinstance(raw_bytes, np.ndarray):
        u8 = raw_bytes.view(np.uint8)
    else:
        raise TypeError(f"Unsupported input type for fp8 decoding: {type(raw_bytes)}")

    f32 = FP8_E4M3_LUT[u8]
    if shape is not None:
        f32 = f32.reshape(shape)
    return f32


@dataclass
class ExpertSlice:
    """
    Represents an isolated extracted expert matrix slice.
    """
    layer_idx: int
    expert_idx: int
    proj_type: str
    tensor_name: str
    weight: np.ndarray              # Dequantized float32 master weight matrix
    raw_fp8: np.ndarray             # Raw FP8 uint8 bytes
    scale_inv: np.ndarray           # Block inverse scales (float32)
    original_shape: Tuple[int, int] # e.g. (2048, 8192)
    block_size: Tuple[int, int]     # e.g. (128, 128)
    load_time_ms: float

    @property
    def memory_mb(self) -> float:
        """Memory footprint of the dequantized float32 matrix in Megabytes."""
        return self.weight.nbytes / (1024 * 1024)

    @property
    def raw_fp8_mb(self) -> float:
        """Memory footprint of the raw FP8 matrix in Megabytes."""
        return self.raw_fp8.nbytes / (1024 * 1024)

    def __repr__(self) -> str:
        return (
            f"<ExpertSlice layer={self.layer_idx} expert={self.expert_idx} "
            f"proj='{self.proj_type}' shape={self.original_shape} "
            f"fp8_size={self.raw_fp8_mb:.2f}MB f32_size={self.memory_mb:.2f}MB "
            f"load_time={self.load_time_ms:.2f}ms>"
        )


def read_safetensors_header(file_path: Union[str, Path]) -> Tuple[Dict[str, dict], int]:
    """
    Reads only the 8-byte header size and JSON metadata header of a safetensors file.
    Does NOT read tensor payload bytes into memory.

    Returns:
        header: dict mapping tensor names to metadata {dtype, shape, data_offsets}
        data_start_offset: byte offset in file where raw tensor buffers begin
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Safetensors shard file not found: {path}")

    with open(path, "rb") as f:
        header_bytes = f.read(8)
        if len(header_bytes) < 8:
            raise ValueError(f"Corrupt or truncated safetensors file: {path}")
        header_size = struct.unpack("<Q", header_bytes)[0]
        if header_size == 0 or header_size > 64 * 1024 * 1024:
            raise ValueError(f"Unsafe or corrupted safetensors header size ({header_size} bytes): {path}")
        header_json = f.read(header_size).decode("utf-8")
        header = json.loads(header_json)

    data_start_offset = 8 + header_size
    return header, data_start_offset


def list_available_tensors(file_path: Union[str, Path]) -> List[str]:
    """
    Returns all tensor keys stored in the safetensors file.
    """
    header, _ = read_safetensors_header(file_path)
    return [k for k in header.keys() if k != "__metadata__"]


def extract_raw_tensor(
    file_path: Union[str, Path],
    tensor_name: str,
    header: Optional[Dict[str, dict]] = None,
    data_start_offset: Optional[int] = None,
) -> Tuple[bytes, dict]:
    """
    Directly seeks to and reads the exact byte slice of a specific tensor.
    Constant memory overhead: only allocates the bytes of the targeted tensor.
    """
    if header is None or data_start_offset is None:
        header, data_start_offset = read_safetensors_header(file_path)

    if tensor_name not in header:
        raise KeyError(
            f"Tensor '{tensor_name}' not found in {file_path}. "
            f"Available tensors sample: {list(header.keys())[:5]}"
        )

    t_info = header[tensor_name]
    start_rel, end_rel = t_info["data_offsets"]
    length = end_rel - start_rel

    with open(file_path, "rb") as f:
        f.seek(data_start_offset + start_rel)
        raw_bytes = f.read(length)

    if len(raw_bytes) != length:
        raise IOError(
            f"Failed to read complete tensor '{tensor_name}': expected {length} bytes, got {len(raw_bytes)}"
        )

    return raw_bytes, t_info


def extract_expert_slice(
    file_path: Union[str, Path],
    layer_idx: int = 0,
    expert_idx: int = 0,
    proj: str = "gate_proj",
    block_size: Tuple[int, int] = (128, 128),
) -> ExpertSlice:
    """
    Extracts a single expert projection matrix and its inverse block scales,
    dequantizing from FP8 E4M3 into float32 master weights.

    Args:
        file_path: Path to safetensors shard (e.g. model-00001-of-00213.safetensors)
        layer_idx: Transformer layer index (default: 0)
        expert_idx: Expert identifier (0 to 511)
        proj: Projection name ('gate_proj', 'up_proj', 'down_proj')
        block_size: Block quantization size (default: (128, 128))

    Returns:
        ExpertSlice containing dequantized float32 matrix and metadata.
    """
    t0 = time.perf_counter()
    header, data_start = read_safetensors_header(file_path)

    weight_name = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{proj}.weight"
    scale_name = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{proj}.weight_scale_inv"

    # Extract weight bytes
    raw_w, w_info = extract_raw_tensor(file_path, weight_name, header, data_start)
    w_shape = tuple(w_info["shape"])
    w_dtype = w_info["dtype"]

    # Extract scale bytes
    raw_s, s_info = extract_raw_tensor(file_path, scale_name, header, data_start)
    s_shape = tuple(s_info["shape"])
    s_dtype = s_info["dtype"]

    # Decode FP8 weights
    if w_dtype in ("F8_E4M3", "float8_e4m3fn"):
        w_f32 = decode_fp8_e4m3_to_f32(raw_w, w_shape)
        raw_fp8_u8 = np.frombuffer(raw_w, dtype=np.uint8).reshape(w_shape)
    else:
        raise NotImplementedError(f"Unsupported weight dtype: {w_dtype}")

    # Decode BF16 block scales
    if s_dtype in ("BF16", "bfloat16"):
        s_f32 = decode_bf16_to_f32(raw_s, s_shape)
    elif s_dtype in ("F32", "float32"):
        s_f32 = np.frombuffer(raw_s, dtype="<f4").reshape(s_shape)
    else:
        raise NotImplementedError(f"Unsupported scale dtype: {s_dtype}")

    # Expand 128x128 block scales to match weight shape
    blk_h, blk_w = block_size
    s_expanded = np.repeat(np.repeat(s_f32, blk_h, axis=0), blk_w, axis=1)

    # Dequantize master weight: W_dequant = W_fp8 * scale_inv
    w_dequant = w_f32 * s_expanded

    t1 = time.perf_counter()
    load_time_ms = (t1 - t0) * 1000.0

    return ExpertSlice(
        layer_idx=layer_idx,
        expert_idx=expert_idx,
        proj_type=proj,
        tensor_name=weight_name,
        weight=w_dequant,
        raw_fp8=raw_fp8_u8,
        scale_inv=s_f32,
        original_shape=w_shape,
        block_size=block_size,
        load_time_ms=load_time_ms,
    )


DEFAULT_MODEL_DIR = Path(r"H:\AI_Memory_Vault\models\Qwen3.8-2.4T-A95B-FP8")


def extract_tensor(
    file_path: Union[str, Path],
    tensor_name: str,
    header: Optional[Dict[str, dict]] = None,
    data_start_offset: Optional[int] = None,
) -> np.ndarray:
    """
    Extracts any tensor from a safetensors file and decodes it to float32.
    If a corresponding {tensor_name}_scale_inv exists, dequantizes using 128x128 block scaling.
    """
    if header is None or data_start_offset is None:
        header, data_start_offset = read_safetensors_header(file_path)

    raw_bytes, info = extract_raw_tensor(file_path, tensor_name, header, data_start_offset)
    shape = tuple(info["shape"])
    dtype = info["dtype"]

    if dtype in ("BF16", "bfloat16"):
        return decode_bf16_to_f32(raw_bytes, shape)
    elif dtype in ("F32", "float32"):
        return np.frombuffer(raw_bytes, dtype="<f4").reshape(shape)
    elif dtype in ("F8_E4M3", "float8_e4m3fn"):
        val = decode_fp8_e4m3_to_f32(raw_bytes, shape)
        scale_name = f"{tensor_name}_scale_inv"
        if scale_name not in header:
            # Fallback: re-read shard header to locate scale_inv if partial header was provided
            try:
                full_header, full_dstart = read_safetensors_header(file_path)
                if scale_name in full_header:
                    header = full_header
                    data_start_offset = full_dstart
            except Exception:
                pass
        if scale_name in header:
            scale_raw, s_info = extract_raw_tensor(file_path, scale_name, header, data_start_offset)
            scale_f32 = decode_bf16_to_f32(scale_raw, tuple(s_info["shape"]))
            blk_h = shape[0] // s_info["shape"][0]
            blk_w = shape[1] // s_info["shape"][1]
            scale_exp = np.repeat(np.repeat(scale_f32, blk_h, axis=0), blk_w, axis=1)
            val = val * scale_exp
        else:
            raise ValueError(f"FP8 tensor '{tensor_name}' requires '{scale_name}' but none found in shard header")
        return val
    elif dtype in ("I32", "int32"):
        return np.frombuffer(raw_bytes, dtype="<i4").reshape(shape)
    elif dtype in ("I64", "int64"):
        return np.frombuffer(raw_bytes, dtype="<i8").reshape(shape)
    else:
        raise NotImplementedError(f"Unsupported dtype '{dtype}' for tensor '{tensor_name}'")


def extract_complete_expert(
    expert_idx: int,
    base_dir: Union[str, Path] = DEFAULT_MODEL_DIR,
    layer_idx: int = 0,
) -> Dict[str, ExpertSlice]:
    """
    Extracts gate_proj, up_proj, and down_proj for an expert across shards.
    gate_proj & up_proj: shard 1 (model-00001-of-00213.safetensors)
    down_proj: shard 2 (model-00002-of-00213.safetensors)
    """
    base = Path(base_dir)
    shard1 = base / "model-00001-of-00213.safetensors"
    shard2 = base / "model-00002-of-00213.safetensors"

    gate_slice = extract_expert_slice(shard1, layer_idx=layer_idx, expert_idx=expert_idx, proj="gate_proj")
    up_slice = extract_expert_slice(shard1, layer_idx=layer_idx, expert_idx=expert_idx, proj="up_proj")
    down_slice = extract_expert_slice(shard2, layer_idx=layer_idx, expert_idx=expert_idx, proj="down_proj")

    return {
        "gate_proj": gate_slice,
        "up_proj": up_slice,
        "down_proj": down_slice,
    }


def extract_layer0_gatekeeper(
    base_dir: Union[str, Path] = DEFAULT_MODEL_DIR,
) -> Dict[str, np.ndarray]:
    """
    Extracts the Layer 0 Gatekeeper weights (Router, Norms, Shared Expert) from shard 11.
    All Gatekeeper weights are in BF16 master precision.
    """
    base = Path(base_dir)
    shard11 = base / "model-00011-of-00213.safetensors"
    header, dstart = read_safetensors_header(shard11)

    target_keys = {
        "router_weight": "model.layers.0.mlp.gate.weight",
        "input_layernorm": "model.layers.0.input_layernorm.weight",
        "post_attention_layernorm": "model.layers.0.post_attention_layernorm.weight",
        "shared_gate_proj": "model.layers.0.mlp.shared_expert.gate_proj.weight",
        "shared_up_proj": "model.layers.0.mlp.shared_expert.up_proj.weight",
        "shared_down_proj": "model.layers.0.mlp.shared_expert.down_proj.weight",
        "shared_gate_weight": "model.layers.0.mlp.shared_expert_gate.weight",
    }

    out = {}
    for key_name, tensor_name in target_keys.items():
        if tensor_name in header:
            out[key_name] = extract_tensor(shard11, tensor_name, header=header, data_start_offset=dstart)
    return out


def extract_token_embeddings(
    token_ids: Sequence[int],
    base_dir: Union[str, Path] = DEFAULT_MODEL_DIR,
) -> np.ndarray:
    """
    Extracts token embedding vectors for specific token IDs from shard 25.
    Shape: (len(token_ids), 8192).
    """
    base = Path(base_dir)
    shard25 = base / "model-00025-of-00213.safetensors"
    header, dstart = read_safetensors_header(shard25)
    t_info = header["model.embed_tokens.weight"]
    start_offset = dstart + t_info["data_offsets"][0]

    hidden_size = 8192
    row_bytes = hidden_size * 2  # BF16 = 2 bytes per float

    embeddings = np.zeros((len(token_ids), hidden_size), dtype=np.float32)
    with open(shard25, "rb") as f:
        for idx, tid in enumerate(token_ids):
            f.seek(start_offset + tid * row_bytes)
            raw = f.read(row_bytes)
            embeddings[idx] = decode_bf16_to_f32(raw, (hidden_size,))
    return embeddings
