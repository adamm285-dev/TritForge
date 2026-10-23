"""
TritForge Native Binary Container (tritforge.container)
=======================================================
Implements the high-density .trit binary container format for storing 1.58-bit
ternary weights and high-precision Gatekeeper parameters.

File Layout (.trit specification v1.0):
+-----------------------------------------------------------------------------+
| Magic Bytes (6B): b"TRIT\\x01\\x00"                                         |
| Metadata Length (4B, uint32 little-endian)                                  |
| JSON Metadata Block (UTF-8 encoded metadata dictionary)                     |
| Binary Payload Section (Byte-aligned packed trits and scale vectors)        |
+-----------------------------------------------------------------------------+

Enforces:
- Invariant 1: Pure Python & NumPy binary parsing (no third-party formats).
- Invariant 4: Constant-memory seek-based extraction without loading full file.
- Density: Base-3 packing (1.6 bits/weight, <= 3.70 MB per 2048x8192 slice).
"""

from __future__ import annotations

import json
import os
import struct
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from tritforge.quantize import (
    QuantizedTritMatrix,
    pack_trits_2bit,
    pack_trits_base3,
    unpack_trits_2bit,
    unpack_trits_base3,
)

TRIT_MAGIC = b"TRIT\x01\x00"
MAGIC_LEN = len(TRIT_MAGIC)


def _serialize_quantized_matrix(
    qmat: QuantizedTritMatrix,
    packing_mode: str = "base3",
    scales_dtype: str = "float16",
) -> Tuple[bytes, bytes, Dict[str, Any]]:
    """
    Serializes a QuantizedTritMatrix into packed payload bytes, scale bytes, and descriptor.
    """
    m, n = qmat.original_shape
    if packing_mode == "base3":
        packed_trits = pack_trits_base3(qmat.trits)
    elif packing_mode == "2bit":
        packed_trits = pack_trits_2bit(qmat.trits)
    else:
        raise ValueError(f"Unknown packing mode: {packing_mode}")

    trit_bytes = packed_trits.tobytes()

    # Scales serialization (stored as float16, bfloat16, or float32 bytes)
    scales_f32 = np.asarray(qmat.scales, dtype="<f4")
    if scales_dtype == "float16":
        scale_bytes = scales_f32.astype("<f2").tobytes()
    elif scales_dtype == "bfloat16":
        scale_bytes = (scales_f32.view(np.uint32) >> 16).astype("<u2").tobytes()
    elif scales_dtype == "float32":
        scale_bytes = scales_f32.tobytes()
    else:
        raise ValueError(f"Unsupported scales_dtype: {scales_dtype}")

    descriptor = {
        "type": "quantized_trit_matrix",
        "shape": [m, n],
        "scaling_mode": qmat.scaling_mode,
        "scale_method": qmat.scale_method,
        "block_size": qmat.block_size,
        "packing_mode": packing_mode,
        "density": qmat.density,
        "scales_shape": list(scales_f32.shape),
        "scales_dtype": scales_dtype,
    }
    return trit_bytes, scale_bytes, descriptor


def save_trit_file(
    file_path: Union[str, Path],
    tensors: Dict[str, Union[QuantizedTritMatrix, np.ndarray]],
    layer_idx: int = 0,
    packing_mode: str = "base3",
    scales_dtype: str = "float16",
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> int:
    """
    Saves quantized matrices and raw arrays into a self-contained .trit file.

    Returns:
        total_bytes_written: Total file size in bytes.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    metadata: Dict[str, Any] = {
        "format": "TRIT",
        "version": "1.0",
        "layer_idx": layer_idx,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tensors": {},
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    payload_chunks: List[bytes] = []
    current_offset = 0

    for name, item in tensors.items():
        if isinstance(item, QuantizedTritMatrix):
            trit_bytes, scale_bytes, desc = _serialize_quantized_matrix(
                item, packing_mode=packing_mode, scales_dtype=scales_dtype
            )

            trit_start = current_offset
            trit_end = trit_start + len(trit_bytes)
            current_offset = trit_end

            scale_start = current_offset
            scale_end = scale_start + len(scale_bytes)
            current_offset = scale_end

            desc["trit_offsets"] = [trit_start, trit_end]
            desc["scale_offsets"] = [scale_start, scale_end]
            metadata["tensors"][name] = desc

            payload_chunks.append(trit_bytes)
            payload_chunks.append(scale_bytes)

        elif isinstance(item, np.ndarray):
            arr_f32 = np.asarray(item, dtype="<f4")
            raw_bytes = arr_f32.tobytes()

            start_offset = current_offset
            end_offset = start_offset + len(raw_bytes)
            current_offset = end_offset

            metadata["tensors"][name] = {
                "type": "raw_array",
                "shape": list(arr_f32.shape),
                "dtype": "float32",
                "offsets": [start_offset, end_offset],
            }
            payload_chunks.append(raw_bytes)
        else:
            raise TypeError(f"Unsupported item type for tensor '{name}': {type(item)}")

    # Encode JSON header
    meta_json = json.dumps(metadata, indent=2).encode("utf-8")
    meta_len = len(meta_json)

    with open(path, "wb") as f:
        f.write(TRIT_MAGIC)
        f.write(struct.pack("<I", meta_len))
        f.write(meta_json)
        for chunk in payload_chunks:
            f.write(chunk)

    total_bytes = path.stat().st_size
    return total_bytes


class StreamingTritWriter:
    """
    Constant-memory streaming writer for massive multi-gigabyte .trit containers.
    Writes packed payloads incrementally to a temporary file on disk,
    keeping resident RAM strictly < 100 MB regardless of expert count.
    """
    def __init__(
        self,
        file_path: Union[str, Path],
        layer_idx: int = 0,
        packing_mode: str = "base3",
        scales_dtype: str = "float16",
        extra_metadata: Optional[Dict[str, Any]] = None,
    ):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.payload_tmp_path = self.file_path.with_suffix(".payload.tmp")
        self.layer_idx = layer_idx
        self.packing_mode = packing_mode
        self.scales_dtype = scales_dtype
        self.extra_metadata = extra_metadata or {}

        self.metadata: Dict[str, Any] = {
            "format": "TRIT",
            "version": "1.0",
            "layer_idx": layer_idx,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tensors": {},
        }
        self.metadata.update(self.extra_metadata)
        self.current_offset = 0
        self._payload_file = open(self.payload_tmp_path, "wb")

    def add_quantized_matrix(self, name: str, qmat: QuantizedTritMatrix) -> None:
        """Serializes and appends a QuantizedTritMatrix directly to disk."""
        trit_bytes, scale_bytes, desc = _serialize_quantized_matrix(
            qmat, packing_mode=self.packing_mode, scales_dtype=self.scales_dtype
        )
        trit_start = self.current_offset
        trit_end = trit_start + len(trit_bytes)
        self.current_offset = trit_end

        scale_start = self.current_offset
        scale_end = scale_start + len(scale_bytes)
        self.current_offset = scale_end

        desc["trit_offsets"] = [trit_start, trit_end]
        desc["scale_offsets"] = [scale_start, scale_end]
        self.metadata["tensors"][name] = desc

        self._payload_file.write(trit_bytes)
        self._payload_file.write(scale_bytes)

    def add_raw_array(self, name: str, arr: np.ndarray, dtype: str = "bfloat16") -> None:
        """Serializes and appends a raw array directly to disk (Q-2: BF16 by default)."""
        arr_f32 = np.asarray(arr, dtype=np.float32)
        shape_list = list(arr_f32.shape)

        if dtype == "bfloat16":
            # Lossless representation of BF16 source parameters, saving ~50% raw bytes vs FP32
            u32 = arr_f32.view(np.uint32)
            u16 = (u32 >> 16).astype("<u2")
            raw_bytes = u16.tobytes()
            storage_dtype = "bfloat16"
        else:
            raw_bytes = arr_f32.astype("<f4").tobytes()
            storage_dtype = "float32"

        start_offset = self.current_offset
        end_offset = start_offset + len(raw_bytes)
        self.current_offset = end_offset

        self.metadata["tensors"][name] = {
            "type": "raw_array",
            "shape": shape_list,
            "dtype": storage_dtype,
            "offsets": [start_offset, end_offset],
        }
        self._payload_file.write(raw_bytes)

    def close(self) -> int:
        """Finalizes the container atomically by writing header and prepending to payload."""
        self._payload_file.close()

        meta_json = json.dumps(self.metadata, indent=2).encode("utf-8")
        meta_len = len(meta_json)

        tmp_target = self.file_path.with_name(f"{self.file_path.name}.atomic_tmp")

        with open(tmp_target, "wb") as f_out:
            f_out.write(TRIT_MAGIC)
            f_out.write(struct.pack("<I", meta_len))
            f_out.write(meta_json)

            # Stream copy payload in 16 MB chunks
            with open(self.payload_tmp_path, "rb") as f_in:
                while True:
                    chunk = f_in.read(16 * 1024 * 1024)
                    if not chunk:
                        break
                    f_out.write(chunk)
            f_out.flush()
            os.fsync(f_out.fileno())

        if self.payload_tmp_path.exists():
            self.payload_tmp_path.unlink()

        # Atomic replacement guarantees zero corrupted/partially written files
        tmp_target.replace(self.file_path)

        return self.file_path.stat().st_size

    def abort(self) -> None:
        """Aborts container writing on failure without finalizing or creating destination file (D1, E1).
        Cleans up only temporary artifacts (.payload.tmp and .atomic_tmp) created by this writer.
        Never unlinks self.file_path, preserving any previous successful container (E1)."""
        try:
            if hasattr(self, "_payload_file") and self._payload_file and not self._payload_file.closed:
                self._payload_file.close()
        except Exception:
            pass

        if hasattr(self, "payload_tmp_path") and self.payload_tmp_path.exists():
            try:
                self.payload_tmp_path.unlink()
            except Exception:
                pass

        tmp_target = self.file_path.with_name(f"{self.file_path.name}.atomic_tmp")
        if tmp_target.exists():
            try:
                tmp_target.unlink()
            except Exception:
                pass



def read_trit_header(file_path: Union[str, Path]) -> Tuple[Dict[str, Any], int]:
    """
    Reads only the magic header and JSON metadata of a .trit container.
    Constant memory overhead: does NOT load tensor payloads into RAM.

    Returns:
        metadata: Dict containing container structure and tensor descriptors.
        payload_start_offset: Byte offset where raw binary data begins.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f".trit container not found: {path}")

    with open(path, "rb") as f:
        magic = f.read(MAGIC_LEN)
        if magic != TRIT_MAGIC:
            raise ValueError(f"Invalid .trit magic header: expected {TRIT_MAGIC}, got {magic}")

        meta_len_bytes = f.read(4)
        if len(meta_len_bytes) < 4:
            raise ValueError(f"Truncated .trit container: {path}")
        meta_len = struct.unpack("<I", meta_len_bytes)[0]
        if meta_len == 0 or meta_len > 64 * 1024 * 1024:
            raise ValueError(f"Unsafe or corrupted .trit metadata length ({meta_len} bytes): {path}")

        meta_json = f.read(meta_len).decode("utf-8")
        metadata = json.loads(meta_json)

    payload_start = MAGIC_LEN + 4 + meta_len
    return metadata, payload_start


def load_trit_tensor(
    file_path: Union[str, Path],
    tensor_name: str,
    header: Optional[Dict[str, Any]] = None,
    payload_start_offset: Optional[int] = None,
) -> Union[QuantizedTritMatrix, np.ndarray]:
    """
    Directly seeks to and reads a specific tensor from a .trit container.
    Allocates memory strictly for the requested tensor.
    """
    path = Path(file_path)
    if header is None or payload_start_offset is None:
        header, payload_start_offset = read_trit_header(path)

    tensors = header.get("tensors", {})
    if tensor_name not in tensors:
        raise KeyError(f"Tensor '{tensor_name}' not found in .trit file. Available: {list(tensors.keys())}")

    desc = tensors[tensor_name]
    t_type = desc.get("type")

    with open(path, "rb") as f:
        if t_type == "quantized_trit_matrix":
            shape = tuple(desc["shape"])
            packing_mode = desc.get("packing_mode", "base3")
            trit_start, trit_end = desc["trit_offsets"]
            scale_start, scale_end = desc["scale_offsets"]

            # Read packed trits
            f.seek(payload_start_offset + trit_start)
            trit_bytes = f.read(trit_end - trit_start)

            # Read scale vector
            f.seek(payload_start_offset + scale_start)
            scale_bytes = f.read(scale_end - scale_start)

            # Unpack trits
            packed_arr = np.frombuffer(trit_bytes, dtype=np.uint8)
            if packing_mode == "base3":
                trits = unpack_trits_base3(packed_arr, shape)
            elif packing_mode == "2bit":
                trits = unpack_trits_2bit(packed_arr, shape)
            else:
                raise ValueError(f"Unknown packing mode: {packing_mode}")

            # Decode scales
            scales_shape = tuple(desc["scales_shape"])
            s_dtype = desc.get("scales_dtype", "float32")
            if s_dtype == "float16":
                scales = np.frombuffer(scale_bytes, dtype="<f2").astype(np.float32).reshape(scales_shape)
            elif s_dtype == "bfloat16":
                u16 = np.frombuffer(scale_bytes, dtype="<u2")
                scales = ((u16.astype(np.uint32) << 16)).view(np.float32).reshape(scales_shape)
            else:
                scales = np.frombuffer(scale_bytes, dtype="<f4").reshape(scales_shape)

            return QuantizedTritMatrix(
                trits=trits,
                scales=scales,
                original_shape=shape,
                scaling_mode=desc.get("scaling_mode", "block"),
                scale_method=desc.get("scale_method", "optimal"),
                block_size=desc.get("block_size", 128),
                method="loaded_from_container",
            )

        elif t_type == "raw_array":
            shape = tuple(desc["shape"])
            start, end = desc["offsets"]
            arr_dtype = desc.get("dtype", "float32")
            f.seek(payload_start_offset + start)
            raw = f.read(end - start)
            if arr_dtype == "bfloat16":
                u16 = np.frombuffer(raw, dtype="<u2")
                arr = ((u16.astype(np.uint32) << 16)).view(np.float32).reshape(shape)
            else:
                arr = np.frombuffer(raw, dtype="<f4").reshape(shape)
            return arr.copy()
        else:
            raise NotImplementedError(f"Unsupported tensor type: {t_type}")


def save_trit_layer(
    file_path: Union[str, Path],
    layer_data: Optional[Union[Any, Dict[str, Any]]] = None,
    router_weight: Optional[np.ndarray] = None,
    experts: Optional[Dict[int, Any]] = None,
    shared_expert: Optional[Any] = None,
    input_norm: Optional[np.ndarray] = None,
    post_norm: Optional[np.ndarray] = None,
    layer_idx: int = 0,
    packing_mode: str = "base3",
    scales_dtype: str = "float16",
) -> int:
    """
    Saves an assembled Asymmetric MoE layer into a .trit container file.

    Accepts either:
    1. layer_data: An instance of Layer0AsymmetricMoE or dict of layer components.
    2. Explicit keyword arguments (router_weight, experts, etc.).
    """
    if layer_data is not None:
        if hasattr(layer_data, "router"):
            # Layer0AsymmetricMoE object
            router_weight = layer_data.router.gate_weight
            experts = layer_data.experts
            shared_expert = layer_data.shared_expert
            input_norm = getattr(layer_data, "input_layernorm_weight", None)
            post_norm = getattr(layer_data, "post_attention_layernorm_weight", None)
        elif isinstance(layer_data, dict):
            router_weight = layer_data.get("router_weight", router_weight)
            experts = layer_data.get("experts", experts)
            shared_expert = layer_data.get("shared_expert", shared_expert)
            input_norm = layer_data.get("input_norm", input_norm)
            post_norm = layer_data.get("post_norm", post_norm)
        elif isinstance(layer_data, np.ndarray):
            router_weight = layer_data

    tensors: Dict[str, Union[QuantizedTritMatrix, np.ndarray]] = {}

    if router_weight is not None:
        tensors["router.weight"] = np.asarray(router_weight, dtype=np.float32)
    if input_norm is not None:
        tensors["input_layernorm.weight"] = np.asarray(input_norm, dtype=np.float32)
    if post_norm is not None:
        tensors["post_attention_layernorm.weight"] = np.asarray(post_norm, dtype=np.float32)

    if shared_expert is not None:
        tensors["shared_expert.gate_proj"] = np.asarray(shared_expert.gate_proj, dtype=np.float32)
        tensors["shared_expert.up_proj"] = np.asarray(shared_expert.up_proj, dtype=np.float32)
        tensors["shared_expert.down_proj"] = np.asarray(shared_expert.down_proj, dtype=np.float32)
        if getattr(shared_expert, "gate_weight", None) is not None:
            tensors["shared_expert.gate_weight"] = np.asarray(shared_expert.gate_weight, dtype=np.float32)

    if experts is not None:
        for e_id, expert in experts.items():
            tensors[f"experts.{e_id}.gate_proj"] = expert.gate_proj
            tensors[f"experts.{e_id}.up_proj"] = expert.up_proj
            tensors[f"experts.{e_id}.down_proj"] = expert.down_proj

    extra_meta = {
        "num_experts": len(experts) if experts is not None else 0,
        "expert_indices": list(experts.keys()) if experts is not None else [],
    }
    return save_trit_file(
        file_path,
        tensors,
        layer_idx=layer_idx,
        packing_mode=packing_mode,
        scales_dtype=scales_dtype,
        extra_metadata=extra_meta,
    )


def load_trit_layer(
    file_path: Union[str, Path],
    expert_idx: Optional[int] = None,
    proj: Optional[str] = None,
) -> Any:
    """
    Seek-based loader for .trit containers with constant-memory partial extraction.

    Modes:
    1. load_trit_layer(path, expert_idx=0, proj="gate_proj"):
       Directly seeks and returns only the requested QuantizedTritMatrix.
    2. load_trit_layer(path, expert_idx=0):
       Directly seeks and returns a TernaryExpert with (gate_proj, up_proj, down_proj).
    3. load_trit_layer(path):
       Loads the complete Layer0AsymmetricMoE block.

    Guarantees:
    - Never allocates memory for unrequested experts or projections during targeted reads.
    """
    path = Path(file_path)
    header, payload_start = read_trit_header(path)
    tensors = header.get("tensors", {})

    # Mode 1: Targeted single projection query
    if expert_idx is not None and proj is not None:
        tensor_name = f"experts.{expert_idx}.{proj}"
        if tensor_name not in tensors:
            raise KeyError(f"Expert {expert_idx} projection '{proj}' not found in container: {list(tensors.keys())}")
        return load_trit_tensor(path, tensor_name, header=header, payload_start_offset=payload_start)

    # Mode 2: Targeted single expert query (gate_proj, up_proj, down_proj)
    from tritforge.moe import Layer0AsymmetricMoE, MoERouter, SharedExpert, TernaryExpert

    if expert_idx is not None and proj is None:
        g_name = f"experts.{expert_idx}.gate_proj"
        u_name = f"experts.{expert_idx}.up_proj"
        d_name = f"experts.{expert_idx}.down_proj"

        if g_name not in tensors or u_name not in tensors or d_name not in tensors:
            raise KeyError(f"Expert {expert_idx} not found in container {path}.")

        q_gate = load_trit_tensor(path, g_name, header=header, payload_start_offset=payload_start)
        q_up = load_trit_tensor(path, u_name, header=header, payload_start_offset=payload_start)
        q_down = load_trit_tensor(path, d_name, header=header, payload_start_offset=payload_start)

        return TernaryExpert(
            expert_idx=expert_idx,
            gate_proj=q_gate,
            up_proj=q_up,
            down_proj=q_down,
        )

    # Mode 3: Load full Layer0AsymmetricMoE (supports both HF and legacy key schemas, C-6)
    # 1. Load Router
    r_key = "mlp.gate.weight" if "mlp.gate.weight" in tensors else ("router.weight" if "router.weight" in tensors else None)
    if r_key is not None:
        router_w = load_trit_tensor(path, r_key, header=header, payload_start_offset=payload_start)
        router = MoERouter(gate_weight=router_w)
    else:
        router = None

    # 2. Load Layernorms
    in_key = "input_layernorm.weight" if "input_layernorm.weight" in tensors else None
    input_norm = load_trit_tensor(path, in_key, header=header, payload_start_offset=payload_start) if in_key else None

    post_key = "post_attention_layernorm.weight" if "post_attention_layernorm.weight" in tensors else None
    post_norm = load_trit_tensor(path, post_key, header=header, payload_start_offset=payload_start) if post_key else None

    # 3. Load Shared Expert
    shared_exp = None
    if "mlp.shared_expert.gate_proj.weight" in tensors:
        s_gate = load_trit_tensor(path, "mlp.shared_expert.gate_proj.weight", header=header, payload_start_offset=payload_start)
        s_up = load_trit_tensor(path, "mlp.shared_expert.up_proj.weight", header=header, payload_start_offset=payload_start)
        s_down = load_trit_tensor(path, "mlp.shared_expert.down_proj.weight", header=header, payload_start_offset=payload_start)
        s_gate_w = None
        if "mlp.shared_expert_gate.weight" in tensors:
            s_gate_w = load_trit_tensor(path, "mlp.shared_expert_gate.weight", header=header, payload_start_offset=payload_start)
        shared_exp = SharedExpert(gate_proj=s_gate, up_proj=s_up, down_proj=s_down, gate_weight=s_gate_w)
    elif "shared_expert.gate_proj" in tensors:
        s_gate = load_trit_tensor(path, "shared_expert.gate_proj", header=header, payload_start_offset=payload_start)
        s_up = load_trit_tensor(path, "shared_expert.up_proj", header=header, payload_start_offset=payload_start)
        s_down = load_trit_tensor(path, "shared_expert.down_proj", header=header, payload_start_offset=payload_start)
        s_gate_w = None
        if "shared_expert.gate_weight" in tensors:
            s_gate_w = load_trit_tensor(path, "shared_expert.gate_weight", header=header, payload_start_offset=payload_start)
        shared_exp = SharedExpert(gate_proj=s_gate, up_proj=s_up, down_proj=s_down, gate_weight=s_gate_w)

    # 4. Load all available experts
    expert_indices = header.get("expert_indices", [])
    if not expert_indices:
        # Infer from tensor keys
        expert_ids = set()
        for k in tensors.keys():
            if k.startswith("experts."):
                parts = k.split(".")
                if len(parts) >= 2 and parts[1].isdigit():
                    expert_ids.add(int(parts[1]))
        expert_indices = sorted(list(expert_ids))

    loaded_experts: Dict[int, TernaryExpert] = {}
    for e_id in expert_indices:
        g_name = f"experts.{e_id}.gate_proj"
        u_name = f"experts.{e_id}.up_proj"
        d_name = f"experts.{e_id}.down_proj"
        if g_name in tensors and u_name in tensors and d_name in tensors:
            loaded_experts[e_id] = TernaryExpert(
                expert_idx=e_id,
                gate_proj=load_trit_tensor(path, g_name, header=header, payload_start_offset=payload_start),
                up_proj=load_trit_tensor(path, u_name, header=header, payload_start_offset=payload_start),
                down_proj=load_trit_tensor(path, d_name, header=header, payload_start_offset=payload_start),
            )

    if router is None:
        # Return dict of loaded components
        return {
            "experts": loaded_experts,
            "shared_expert": shared_exp,
            "input_norm": input_norm,
            "post_norm": post_norm,
        }

    return Layer0AsymmetricMoE(
        router=router,
        experts=loaded_experts,
        shared_expert=shared_exp,
        input_layernorm_weight=input_norm,
        post_attention_layernorm_weight=post_norm,
        num_experts=router.num_experts,
        num_experts_per_tok=router.num_experts_per_tok,
    )

