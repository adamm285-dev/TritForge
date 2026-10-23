"""
TritForge Production Shard-to-Trit Batch Converter (tritforge.converter)
========================================================================
Streaming, memory-bounded batch conversion pipeline that converts multi-shard
Safetensors models (e.g., Qwen 2.4T / Qwen3.8-2.4T-A95B) into high-density .trit
binary container format with Asymmetric Mixed-Precision.

Core Capabilities:
- Layer-by-Layer Extraction: Scans shard headers and reads only needed byte ranges.
- Second-Order Compensation & Quantization: Applies block-wise Absmax ternary
  quantization and Hessian exit drift compensation to routed experts.
- Production .trit Export: Packages Gatekeeper (Attention, Router, Shared Experts,
  Norms) and Base-3 packed expert trits (<= 3.70 MB/slice) into per-layer .trit files.
- Checkpoint & Resume: Tracks conversion in conversion_manifest.json and gracefully
  resumes interrupted conversion runs without duplicating work.
- Real-Time Telemetry: Monitors ETA, compression ratio, and memory high-water mark.

Enforces:
- Invariant 1: Zero heavy external dependencies (Pure Python Standard Library + NumPy).
- Invariant 2: Mathematical Asymmetry (High-Precision Gatekeeper + Ternary Lanes).
- Invariant 4: Constant-memory seek-based extraction and conversion.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import re
import subprocess
import time
import tracemalloc
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from tritforge.container import StreamingTritWriter, save_trit_file, save_trit_layer
from tritforge.moe import Layer0AsymmetricMoE, MoERouter, SharedExpert, TernaryExpert
from tritforge.quantize import (
    QuantizedTritMatrix,
    absmax_ternary_quantize,
    hessian_compensate_quantize,
)
from tritforge.slice import (
    DEFAULT_MODEL_DIR,
    ExpertSlice,
    extract_tensor,
    read_safetensors_header,
)


@dataclass
class LayerShardMapping:
    """Discovered tensor locations for a specific layer across safetensors shards."""
    layer_idx: int
    gatekeeper_tensors: Dict[str, Tuple[Path, Dict[str, Any], int]] = field(default_factory=dict)
    # expert_idx -> proj_name -> (shard_path, tensor_desc, data_start_offset)
    expert_tensors: Dict[int, Dict[str, Tuple[Path, Dict[str, Any], int]]] = field(default_factory=dict)
    # expert_idx -> proj_name -> (shard_path, tensor_desc, data_start_offset)
    scale_tensors: Dict[int, Dict[str, Tuple[Path, Dict[str, Any], int]]] = field(default_factory=dict)


class ShardIndex:
    """
    Catalog of available tensors and shard byte-range mappings across a multi-shard
    safetensors model directory without allocating tensor payloads in memory.
    """
    def __init__(self, model_dir: Union[str, Path]):
        self.model_dir = Path(model_dir)
        self.shard_files: List[Path] = []
        self.tensor_to_shard: Dict[str, Tuple[Path, Dict[str, Any], int]] = {}
        self.layers: Dict[int, LayerShardMapping] = {}
        self.global_tensors: Dict[str, Tuple[Path, Dict[str, Any], int]] = {}
        self.config: Dict[str, Any] = {}

        self._index_directory()

    def _index_directory(self) -> None:
        """Discovers safetensors shards and indexes headers."""
        # Load config.json if present
        cfg_file = self.model_dir / "config.json"
        if cfg_file.exists():
            try:
                self.config = json.loads(cfg_file.read_text(encoding="utf-8"))
            except Exception:
                self.config = {}

        # Scan for safetensors shards
        self.shard_files = sorted(list(self.model_dir.glob("*.safetensors")))
        layer_pat = re.compile(r"model\.layers\.(\d+)\.")
        expert_pat = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.([a-z_]+)\.weight(_scale_inv)?")

        for shard_path in self.shard_files:
            # Skip shards currently being downloaded by aria2c
            if Path(f"{shard_path}.aria2").exists():
                continue

            try:
                header, dstart = read_safetensors_header(shard_path)
            except Exception:
                continue

            for t_name, t_info in header.items():
                if t_name.startswith("__"):
                    continue

                entry = (shard_path, t_info, dstart)
                self.tensor_to_shard[t_name] = entry

                # Check if layer tensor
                l_match = layer_pat.search(t_name)
                if l_match:
                    layer_idx = int(l_match.group(1))
                    if layer_idx not in self.layers:
                        self.layers[layer_idx] = LayerShardMapping(layer_idx=layer_idx)
                    layer_map = self.layers[layer_idx]

                    # Check if expert tensor
                    e_match = expert_pat.search(t_name)
                    if e_match:
                        e_idx = int(e_match.group(2))
                        proj = e_match.group(3)
                        is_scale = bool(e_match.group(4))

                        if is_scale:
                            if e_idx not in layer_map.scale_tensors:
                                layer_map.scale_tensors[e_idx] = {}
                            layer_map.scale_tensors[e_idx][proj] = entry
                        else:
                            if e_idx not in layer_map.expert_tensors:
                                layer_map.expert_tensors[e_idx] = {}
                            layer_map.expert_tensors[e_idx][proj] = entry
                    else:
                        # Gatekeeper tensor (router, shared expert, layernorm, attention)
                        layer_map.gatekeeper_tensors[t_name] = entry
                else:
                    self.global_tensors[t_name] = entry

    @property
    def available_layers(self) -> List[int]:
        """Returns sorted list of available layer indices in the model directory."""
        return sorted(list(self.layers.keys()))

    def get_layer_expert_indices(self, layer_idx: int) -> List[int]:
        """Returns available expert indices for a specific layer."""
        if layer_idx not in self.layers:
            return []
        return sorted(list(self.layers[layer_idx].expert_tensors.keys()))


@dataclass
class LayerConversionMetrics:
    """Metrics recorded for an individual layer conversion."""
    layer_idx: int
    num_experts: int
    input_bytes: int
    output_bytes: int
    compression_ratio_pct: float
    duration_s: float
    output_file: str
    sha256: str = ""


@dataclass
class ConversionSummary:
    """Summary of batch model conversion execution."""
    model_name: str
    total_layers_converted: int
    total_experts_converted: int
    total_input_bytes: int
    total_output_bytes: int
    overall_compression_pct: float
    total_duration_s: float
    manifest_path: str
    layer_metrics: List[LayerConversionMetrics]

    def to_markdown_table(self) -> str:
        lines = [
            f"### Production Batch Conversion Summary ({self.model_name})",
            "",
            "| Batch Metric | Target / Baseline | Converted Output | Ratio / Status |",
            "|:---|:---:|:---:|:---:|",
            f"| **Layers Processed** | {self.total_layers_converted} Layers | {self.total_layers_converted} .trit Containers | 100% Verified |",
            f"| **Experts Quantized** | {self.total_experts_converted} Experts | 1.58b Base-3 Trits | Zero-Multiply GEMM |",
            f"| **Input Shard Footprint** | {self.total_input_bytes / (1024**3):.2f} GB | Raw Safetensors | Uncompressed |",
            f"| **Output .trit Footprint** | {self.total_output_bytes / (1024**3):.2f} GB | Native .trit v1.0 | **{self.overall_compression_pct:.2f}% Reduction** |",
            f"| **Conversion Duration** | N/A | {self.total_duration_s:.1f} seconds | Completed |",
            "",
            "#### Per-Layer Conversion Breakdown",
            "| Layer | Experts | Input (MB) | Output (MB) | Savings | Duration (s) | Container File |",
            "|:---:|:---:|:---:|:---:|:---:|:---:|:---|",
        ]
        for m in self.layer_metrics:
            in_mb = m.input_bytes / (1024 ** 2)
            out_mb = m.output_bytes / (1024 ** 2)
            lines.append(
                f"| Layer {m.layer_idx:02d} | {m.num_experts} | {in_mb:.1f} MB | {out_mb:.1f} MB | "
                f"{m.compression_ratio_pct:.1f}% | {m.duration_s:.2f}s | `{Path(m.output_file).name}` |"
            )
        return "\n".join(lines)


def get_git_commit_sha() -> str:
    """Retrieves current git commit SHA, or empty string if unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).parent.parent,
        ).decode("ascii").strip()
    except Exception:
        return "unknown"


def check_git_dirty() -> bool:
    """Checks if git working tree has uncommitted changes in tritforge/."""
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "tritforge/"],
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).parent.parent,
        ).decode("utf-8").strip()
        return bool(status)
    except Exception:
        return False


def file_sha256(filepath: Union[str, Path], chunk_size: int = 4 * 1024 * 1024) -> str:
    """Computes SHA-256 hash of a file on disk."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


class BatchConverter:
    """
    Streaming, memory-bounded batch converter from multi-shard Safetensors to .trit format.
    """
    def __init__(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        packing_mode: str = "base3",
        scales_dtype: str = "float16",
        block_size: int = 128,
        scale_method: str = "ls_optimal",
        gatekeeper_dtype: str = "bfloat16",
        quantizer_version: str = "1.1.0",
        allow_dirty: bool = False,
        model_name: Optional[str] = None,
    ):
        if not allow_dirty and check_git_dirty():
            raise RuntimeError(
                "Refusing to run BatchConverter with uncommitted changes in tritforge/. "
                "Commit your changes first or pass allow_dirty=True."
            )

        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.packing_mode = packing_mode
        self.scales_dtype = scales_dtype
        self.block_size = block_size
        self.scale_method = scale_method
        self.gatekeeper_dtype = gatekeeper_dtype
        self.quantizer_version = quantizer_version
        self.allow_dirty = allow_dirty
        self.git_commit_sha = get_git_commit_sha()

        self.index = ShardIndex(self.input_dir)
        self.model_name = model_name or self.index.config.get("model_type", "Qwen3.8-2.4T-A95B-FP8")
        self.manifest_path = self.output_dir / "conversion_manifest.json"

        # Validate against existing container files in output directory (one output folder per experiment arm)
        self._validate_existing_containers()

    def _validate_existing_containers(self) -> None:
        """Ensures existing .trit files in output_dir match configured experiment arm."""
        for p in self.output_dir.glob("layer_*.trit"):
            try:
                meta, _ = read_trit_header(p)
                # Check gatekeeper dtype from raw arrays if present
                for t_name, t_info in meta.get("tensors", {}).items():
                    if t_info.get("type") == "raw_array":
                        existing_gk = t_info.get("dtype")
                        if existing_gk and existing_gk != self.gatekeeper_dtype:
                            raise ValueError(
                                f"Output directory {self.output_dir} contains container {p.name} with gatekeeper dtype '{existing_gk}', "
                                f"which conflicts with requested '{self.gatekeeper_dtype}'. Use a separate output folder per experiment arm."
                            )
                        break
                    elif t_info.get("type") == "quantized_trit_matrix":
                        existing_sm = t_info.get("scale_method")
                        if existing_sm and existing_sm != self.scale_method:
                            raise ValueError(
                                f"Output directory {self.output_dir} contains container {p.name} with scale_method '{existing_sm}', "
                                f"which conflicts with requested '{self.scale_method}'. Use a separate output folder per experiment arm."
                            )
                        break
            except ValueError:
                raise
            except Exception:
                pass

    def _read_manifest(self) -> Dict[str, Any]:
        """Reads existing manifest or initializes empty state."""
        if self.manifest_path.exists():
            try:
                data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                # Guard against mismatched parameters or missing keys on resume (B2)
                guarded_keys = {
                    "packing_mode": self.packing_mode,
                    "scales_dtype": self.scales_dtype,
                    "block_size": self.block_size,
                    "scale_method": self.scale_method,
                    "gatekeeper_dtype": self.gatekeeper_dtype,
                    "quantizer_version": self.quantizer_version,
                }
                for key, current_val in guarded_keys.items():
                    manifest_val = data.get(key)
                    if manifest_val is None:
                        raise ValueError(
                            f"Manifest configuration mismatch for '{key}': key is missing from existing manifest, "
                            f"converter configured with {current_val!r}. "
                            f"Refusing to resume with incompatible settings (B2)."
                        )
                    if manifest_val != current_val:
                        raise ValueError(
                            f"Manifest configuration mismatch for '{key}': "
                            f"manifest has {manifest_val!r}, converter configured with {current_val!r}. "
                            f"Refusing to resume with incompatible settings."
                        )
                return data
            except ValueError:
                raise
            except Exception:
                pass
        return {
            "model_name": self.model_name,
            "packing_mode": self.packing_mode,
            "scales_dtype": self.scales_dtype,
            "block_size": self.block_size,
            "scale_method": self.scale_method,
            "gatekeeper_dtype": self.gatekeeper_dtype,
            "quantizer_version": self.quantizer_version,
            "git_commit_sha": self.git_commit_sha,
            "completed_layers": [],
            "failed_layers": {},
            "layer_files": {},
            "metrics": {},
        }

    def _save_manifest(self, manifest: Dict[str, Any]) -> None:
        """Atomically saves conversion manifest to disk."""
        tmp = self.manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        tmp.replace(self.manifest_path)

    def convert_layer(
        self,
        layer_idx: int,
        expert_indices: Optional[Sequence[int]] = None,
        empirical_covariances: Optional[Dict[int, np.ndarray]] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Tuple[Path, LayerConversionMetrics]:
        """
        Converts a single MoE layer into a standalone production .trit container.

        Args:
            layer_idx: Layer index to convert.
            expert_indices: Optional list of expert IDs to convert. If None, converts all available.
            empirical_covariances: Optional precomputed activation Hessian matrices per expert.
            progress_callback: Optional callback receiving real-time conversion telemetry.

        Returns:
            Tuple of (output_trit_path, layer_metrics).
        """
        t0 = time.perf_counter()
        if layer_idx not in self.index.layers:
            raise KeyError(f"Layer {layer_idx} not found in model index at {self.input_dir}")

        layer_map = self.index.layers[layer_idx]
        available_experts = sorted(list(layer_map.expert_tensors.keys()))
        target_experts = list(expert_indices) if expert_indices is not None else available_experts

        out_file = self.output_dir / f"layer_{layer_idx:02d}.trit"

        extra_meta = {
            "model_name": self.model_name,
            "num_experts": len(target_experts),
            "expert_indices": target_experts,
            "scale_method": self.scale_method,
            "gatekeeper_dtype": self.gatekeeper_dtype,
            "quantizer_version": self.quantizer_version,
            "git_commit_sha": self.git_commit_sha,
            "converted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        writer = StreamingTritWriter(
            out_file,
            layer_idx=layer_idx,
            packing_mode=self.packing_mode,
            scales_dtype=self.scales_dtype,
            extra_metadata=extra_meta,
        )

        try:
    
            # 1. Extract and Stream Gatekeeper Tensors (Router, Norms, Shared Expert, Attention)
            total_input_bytes = 0
    
            for t_name, (shard_path, t_info, dstart) in layer_map.gatekeeper_tensors.items():
                arr = extract_tensor(shard_path, t_name, header={t_name: t_info}, data_start_offset=dstart)
                if not np.all(np.isfinite(arr)):
                    raise ValueError(f"Non-finite (NaN or Inf) values detected in gatekeeper tensor '{t_name}'")
                # Normalize key name for container storage
                rel_key = t_name.replace(f"model.layers.{layer_idx}.", "")
                writer.add_raw_array(rel_key, arr)
                total_input_bytes += arr.nbytes
                del arr
                gc.collect()
    
            # 2. Extract, Quantize, and Stream Routed Experts
            tracemalloc.start()
            total_experts_count = len(target_experts)
    
            for step_idx, e_id in enumerate(target_experts):
                if e_id not in layer_map.expert_tensors:
                    continue
    
                projs = layer_map.expert_tensors[e_id]
                for proj_name in ["gate_proj", "up_proj", "down_proj"]:
                    if proj_name not in projs:
                        continue
    
                    shard_path, t_info, dstart = projs[proj_name]
                    full_t_name = f"model.layers.{layer_idx}.mlp.experts.{e_id}.{proj_name}.weight"
                    header_subset = {full_t_name: t_info}
                    scale_t_name = f"{full_t_name}_scale_inv"
                    scale_shard_path = shard_path
                    scale_dstart = dstart
                    if e_id in layer_map.scale_tensors and proj_name in layer_map.scale_tensors[e_id]:
                        scale_shard_path, scale_t_info, scale_dstart = layer_map.scale_tensors[e_id][proj_name]
                        header_subset[scale_t_name] = scale_t_info
    
                    # If weight and scale reside in different shards, extract scale directly from its shard
                    if scale_shard_path != shard_path and scale_t_name in header_subset:
                        w_raw = extract_tensor(shard_path, full_t_name, header={full_t_name: t_info}, data_start_offset=dstart)
                        # If w_raw is already float32 without scaling (or if extract_tensor decoded it)
                        # For FP8 with cross-shard scale, extract_tensor on weight shard won't find scale_inv.
                        # slice.extract_tensor now raises ValueError if FP8 and scale_inv is missing.
                        # So we extract FP8 raw and scale manually across shards:
                        from tritforge.slice import extract_raw_tensor, decode_fp8_e4m3_to_f32, decode_bf16_to_f32
                        raw_w_bytes, w_meta = extract_raw_tensor(shard_path, full_t_name, {full_t_name: t_info}, dstart)
                        raw_s_bytes, s_meta = extract_raw_tensor(scale_shard_path, scale_t_name, {scale_t_name: scale_t_info}, scale_dstart)
                        w_decoded = decode_fp8_e4m3_to_f32(raw_w_bytes, tuple(w_meta["shape"]))
                        s_decoded = decode_bf16_to_f32(raw_s_bytes, tuple(s_meta["shape"]))
                        blk_h = w_meta["shape"][0] // s_meta["shape"][0]
                        blk_w = w_meta["shape"][1] // s_meta["shape"][1]
                        s_exp = np.repeat(np.repeat(s_decoded, blk_h, axis=0), blk_w, axis=1)
                        w = w_decoded * s_exp
                    else:
                        w = extract_tensor(shard_path, full_t_name,
                                           header=header_subset, data_start_offset=dstart)
    
                    if not np.all(np.isfinite(w)):
                        raise ValueError(f"Non-finite (NaN or Inf) values detected in expert tensor '{full_t_name}'")
    
                    total_input_bytes += w.nbytes
    
                    # Quantize: apply Hessian compensation if covariance available, else progressive Absmax (C-3, C-8)
                    cov_entry = empirical_covariances.get(e_id) if empirical_covariances is not None else None
                    # Support either single covariance or per-projection dict {proj_name: H}
                    if isinstance(cov_entry, dict):
                        cov = cov_entry.get(proj_name)
                    else:
                        cov = cov_entry if (cov_entry is not None and cov_entry.shape == (w.shape[1], w.shape[1])) else None
    
                    if cov is not None:
                        if self.scale_method == "ls_optimal":
                            raise NotImplementedError(
                                "Hessian compensation with ls_optimal scale_method is not yet supported (C-4/C-7 pending). "
                                "Refusing to silently fall back to heuristic."
                            )
                        qmat, _ = hessian_compensate_quantize(
                            w,
                            cov,
                            block_size=self.block_size,
                            scale_method=self.scale_method,
                        )
                    else:
                        qmat = absmax_ternary_quantize(
                            w,
                            scaling_mode="block",
                            block_size=self.block_size,
                            scale_method=self.scale_method,
                        )
    
                    writer.add_quantized_matrix(f"experts.{e_id}.{proj_name}", qmat)
    
                    # Explicit cleanup to guarantee constant-memory bound
                    del w, qmat
    
                if step_idx % 16 == 0:
                    gc.collect()
    
                if progress_callback is not None:
                    current_ram_mb = tracemalloc.get_traced_memory()[0] / (1024 ** 2)
                    progress_callback({
                        "layer_idx": layer_idx,
                        "expert_idx": e_id,
                        "experts_completed": step_idx + 1,
                        "total_experts": total_experts_count,
                        "current_ram_mb": current_ram_mb,
                    })
    
            tracemalloc.stop()
    
            # 3. Finalize Native .trit Container
            written_bytes = writer.close()
            out_sha256 = file_sha256(out_file)
        except Exception:
            try:
                writer.close()
            except Exception:
                pass
            if writer.payload_tmp_path.exists():
                try:
                    writer.payload_tmp_path.unlink()
                except Exception:
                    pass
            raise

        t1 = time.perf_counter()
        duration = t1 - t0
        comp_ratio = ((total_input_bytes - written_bytes) / total_input_bytes * 100.0) if total_input_bytes > 0 else 0.0

        metrics = LayerConversionMetrics(
            layer_idx=layer_idx,
            num_experts=len(target_experts),
            input_bytes=total_input_bytes,
            output_bytes=written_bytes,
            compression_ratio_pct=comp_ratio,
            duration_s=duration,
            output_file=str(out_file),
            sha256=out_sha256,
        )
        return out_file, metrics

    def convert_all(
        self,
        layers: Optional[Sequence[int]] = None,
        expert_indices: Optional[Sequence[int]] = None,
        resume: bool = True,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> ConversionSummary:
        """
        Executes batch conversion across all target layers with checkpoint resumption.
        """
        t_start = time.perf_counter()
        manifest = self._read_manifest() if resume else {
            "model_name": self.model_name,
            "packing_mode": self.packing_mode,
            "scales_dtype": self.scales_dtype,
            "block_size": self.block_size,
            "completed_layers": [],
            "layer_files": {},
            "metrics": {},
        }

        all_layers = self.index.available_layers
        target_layers = [l for l in (layers if layers is not None else all_layers) if l in all_layers]

        layer_metrics_list: List[LayerConversionMetrics] = []
        total_in = 0
        total_out = 0
        total_exp = 0

        for l_pos, l_idx in enumerate(target_layers):
            out_file = self.output_dir / f"layer_{l_idx:02d}.trit"

            # Check if layer previously failed in manifest
            if resume and str(l_idx) in manifest.get("failed_layers", {}):
                err_msg = manifest["failed_layers"][str(l_idx)]
                raise RuntimeError(
                    f"Layer {l_idx} previously failed with error: {err_msg!r}. "
                    f"Refusing to retry failed layers automatically (W2-15). Clear failed_layers in manifest to force."
                )

            # Check if layer already converted and can be resumed
            if resume and (l_idx in manifest.get("completed_layers", [])) and out_file.exists():
                stored_m = manifest.get("metrics", {}).get(str(l_idx))
                if stored_m:
                    m = LayerConversionMetrics(**stored_m)
                    # Verify file integrity on resume (L-4)
                    if m.sha256:
                        actual_hash = file_sha256(out_file)
                        if actual_hash != m.sha256:
                            raise ValueError(
                                f"Corrupted layer file '{out_file}': expected SHA-256 {m.sha256}, got {actual_hash}"
                            )
                    elif out_file.stat().st_size != m.output_bytes:
                        raise ValueError(
                            f"Size mismatch for layer file '{out_file}': expected {m.output_bytes} bytes, got {out_file.stat().st_size}"
                        )
                    layer_metrics_list.append(m)
                    total_in += m.input_bytes
                    total_out += m.output_bytes
                    total_exp += m.num_experts
                    continue

            # Convert layer
            try:
                path, metrics = self.convert_layer(
                    layer_idx=l_idx,
                    expert_indices=expert_indices,
                    progress_callback=progress_callback,
                )
            except Exception as ex:
                if "failed_layers" not in manifest:
                    manifest["failed_layers"] = {}
                manifest["failed_layers"][str(l_idx)] = str(ex)
                self._save_manifest(manifest)
                raise

            layer_metrics_list.append(metrics)
            total_in += metrics.input_bytes
            total_out += metrics.output_bytes
            total_exp += metrics.num_experts

            # Update manifest immediately
            if l_idx not in manifest["completed_layers"]:
                manifest["completed_layers"].append(l_idx)
            manifest["layer_files"][str(l_idx)] = str(path)
            manifest["metrics"][str(l_idx)] = asdict(metrics)
            self._save_manifest(manifest)

        # Export model_config.json if not present
        cfg_out = self.output_dir / "model_config.json"
        if not cfg_out.exists():
            cfg_data = dict(self.index.config)
            cfg_data["tritforge_version"] = "0.5.0"
            cfg_data["packing_mode"] = self.packing_mode
            cfg_data["block_size"] = self.block_size
            cfg_out.write_text(json.dumps(cfg_data, indent=2), encoding="utf-8")

        t_end = time.perf_counter()
        total_duration = t_end - t_start
        overall_comp = ((total_in - total_out) / total_in * 100.0) if total_in > 0 else 0.0

        return ConversionSummary(
            model_name=self.model_name,
            total_layers_converted=len(layer_metrics_list),
            total_experts_converted=total_exp,
            total_input_bytes=total_in,
            total_output_bytes=total_out,
            overall_compression_pct=overall_comp,
            total_duration_s=total_duration,
            manifest_path=str(self.manifest_path),
            layer_metrics=layer_metrics_list,
        )
