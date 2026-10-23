"""
TritForge Shard Integrity & Health Auditor (tritforge.verifier)
==============================================================
Fast, memory-bounded validator that audits multi-shard Safetensors archives
(e.g., Qwen 2.4T 213 shards) to guarantee zero file truncation, valid JSON headers,
and byte-offset bounds before batch conversion begins.

Enforces:
- Invariant 1: Zero heavy external dependencies (Pure Python Standard Library).
- Invariant 4: Constant-memory non-allocating scanning (< 50 MB RAM).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import re
import struct
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


@dataclass
class ShardHealth:
    """Diagnostic health state for a single safetensors shard."""
    file_name: str
    file_path: Path
    status: str  # "HEALTHY", "TRUNCATED", "CORRUPT_HEADER", "OFFSET_OUT_OF_BOUNDS", "INCOMPLETE_DOWNLOAD"
    physical_size_bytes: int
    header_length_bytes: int = 0
    declared_payload_bytes: int = 0
    tensor_count: int = 0
    error_message: Optional[str] = None

    @property
    def is_healthy(self) -> bool:
        return self.status == "HEALTHY"

    @property
    def physical_size_gb(self) -> float:
        return self.physical_size_bytes / (1024 ** 3)


@dataclass
class ArchiveHealthReport:
    """Comprehensive integrity report for all shards in a model directory."""
    model_dir: Path
    total_shards: int
    healthy_shards: int
    corrupted_shards: int
    incomplete_aria2_shards: int
    total_bytes: int
    total_tensors: int
    discovered_layers: List[int]
    shards: List[ShardHealth] = field(default_factory=list)
    duration_seconds: float = 0.0

    @property
    def all_healthy(self) -> bool:
        return self.corrupted_shards == 0 and self.incomplete_aria2_shards == 0 and self.healthy_shards > 0

    @property
    def total_gb(self) -> float:
        return self.total_bytes / (1024 ** 3)

    @property
    def total_tb(self) -> float:
        return self.total_bytes / (1024 ** 4)

    def to_markdown_table(self) -> str:
        lines = [
            "### Master Shard Archive Health & Integrity Audit",
            "",
            f"**Model Directory:** `{self.model_dir}`",
            f"**Audit Status:** {'✅ ALL SHARDS HEALTHY' if self.all_healthy else '⚠️ ANOMALIES DETECTED'}",
            f"**Audit Duration:** {self.duration_seconds:.2f} seconds",
            "",
            "| Audit Metric | Measured Value | Spec Requirement | Status |",
            "|:---|:---:|:---:|:---:|",
            f"| **Total Safetensors Shards** | {self.total_shards} Files | Discovered on disk | VERIFIED |",
            f"| **Healthy & Intact Shards** | {self.healthy_shards} / {self.total_shards} | 100% Header & Boundary Match | {'PASS' if self.healthy_shards == self.total_shards else 'WARN'} |",
            f"| **Corrupted / Truncated Shards** | {self.corrupted_shards} Files | Exactly 0 | {'PASS' if self.corrupted_shards == 0 else 'FAIL'} |",
            f"| **Incomplete Download Markers (.aria2)** | {self.incomplete_aria2_shards} Files | 0 Active Incomplete Markers | {'PASS' if self.incomplete_aria2_shards == 0 else 'DOWNLOADING'} |",
            f"| **Total Verified Data Volume** | {self.total_gb:.2f} GB ({self.total_tb:.2f} TB) | Master Volume on Disk | VERIFIED |",
            f"| **Total Indexed Tensors** | {self.total_tensors:,} Tensors | Valid Byte Offsets | VERIFIED |",
            f"| **Discovered Layer Coverage** | {len(self.discovered_layers)} Layers ({min(self.discovered_layers) if self.discovered_layers else 0}-{max(self.discovered_layers) if self.discovered_layers else 0}) | 92 Layers (0-91) | {'COMPLETE' if len(self.discovered_layers) == 92 else f'{len(self.discovered_layers)}/92 Ready'} |",
        ]

        if self.corrupted_shards > 0:
            lines.extend([
                "",
                "#### Corrupted Shard Details",
                "| Shard Name | Status | Error Reason |",
                "|:---|:---:|:---|",
            ])
            for s in self.shards:
                if not s.is_healthy:
                    lines.append(f"| `{s.file_name}` | {s.status} | {s.error_message} |")

        return "\n".join(lines)


def verify_single_shard(shard_path: Union[str, Path]) -> ShardHealth:
    """
    Validates a single safetensors file for header integrity, JSON validity,
    and byte-offset boundary compliance without loading tensor payloads into memory.
    """
    p = Path(shard_path)
    if not p.exists():
        return ShardHealth(
            file_name=p.name,
            file_path=p,
            status="MISSING",
            physical_size_bytes=0,
            error_message="File does not exist on disk",
        )

    # Check for accompanying .aria2 file (indicates active incomplete download)
    aria2_marker = p.with_name(p.name + ".aria2")
    if aria2_marker.exists():
        return ShardHealth(
            file_name=p.name,
            file_path=p,
            status="INCOMPLETE_DOWNLOAD",
            physical_size_bytes=p.stat().st_size,
            error_message="Active .aria2 incomplete download marker present",
        )

    file_size = p.stat().st_size
    if file_size < 8:
        return ShardHealth(
            file_name=p.name,
            file_path=p,
            status="TRUNCATED",
            physical_size_bytes=file_size,
            error_message="File smaller than 8 bytes (missing length prefix)",
        )

    try:
        with open(p, "rb") as f:
            header_len_bytes = f.read(8)
            header_len = struct.unpack("<Q", header_len_bytes)[0]

            if header_len <= 0 or header_len > 50 * 1024 * 1024:  # Max 50MB header guard
                return ShardHealth(
                    file_name=p.name,
                    file_path=p,
                    status="CORRUPT_HEADER",
                    physical_size_bytes=file_size,
                    header_length_bytes=header_len,
                    error_message=f"Invalid header length prefix: {header_len} bytes",
                )

            if file_size < 8 + header_len:
                return ShardHealth(
                    file_name=p.name,
                    file_path=p,
                    status="TRUNCATED",
                    physical_size_bytes=file_size,
                    header_length_bytes=header_len,
                    error_message=f"File truncated mid-header (size {file_size} < 8 + {header_len})",
                )

            raw_header = f.read(header_len)
            try:
                header = json.loads(raw_header.decode("utf-8"))
            except Exception as j_err:
                return ShardHealth(
                    file_name=p.name,
                    file_path=p,
                    status="CORRUPT_HEADER",
                    physical_size_bytes=file_size,
                    header_length_bytes=header_len,
                    error_message=f"Failed to parse JSON header: {j_err}",
                )

        payload_start = 8 + header_len
        max_offset = 0
        tensor_count = 0

        for t_name, t_meta in header.items():
            if t_name == "__metadata__":
                continue
            tensor_count += 1
            if not isinstance(t_meta, dict) or "data_offsets" not in t_meta:
                return ShardHealth(
                    file_name=p.name,
                    file_path=p,
                    status="CORRUPT_HEADER",
                    physical_size_bytes=file_size,
                    header_length_bytes=header_len,
                    error_message=f"Tensor '{t_name}' missing data_offsets",
                )

            d_offsets = t_meta["data_offsets"]
            if len(d_offsets) != 2 or d_offsets[1] < d_offsets[0]:
                return ShardHealth(
                    file_name=p.name,
                    file_path=p,
                    status="CORRUPT_HEADER",
                    physical_size_bytes=file_size,
                    header_length_bytes=header_len,
                    error_message=f"Invalid data_offsets {d_offsets} for '{t_name}'",
                )

            if d_offsets[1] > max_offset:
                max_offset = d_offsets[1]

        # Verify that max_offset does not exceed physical file size
        expected_min_size = payload_start + max_offset
        if file_size < expected_min_size:
            return ShardHealth(
                file_name=p.name,
                file_path=p,
                status="OFFSET_OUT_OF_BOUNDS",
                physical_size_bytes=file_size,
                header_length_bytes=header_len,
                declared_payload_bytes=max_offset,
                tensor_count=tensor_count,
                error_message=f"File truncated: payload requires {expected_min_size} bytes, got {file_size}",
            )

        return ShardHealth(
            file_name=p.name,
            file_path=p,
            status="HEALTHY",
            physical_size_bytes=file_size,
            header_length_bytes=header_len,
            declared_payload_bytes=max_offset,
            tensor_count=tensor_count,
        )

    except Exception as e:
        return ShardHealth(
            file_name=p.name,
            file_path=p,
            status="IO_ERROR",
            physical_size_bytes=file_size,
            error_message=str(e),
        )


def verify_shard_directory(
    model_dir: Union[str, Path],
    max_workers: int = 4,
) -> ArchiveHealthReport:
    """
    Audits all safetensors shards in a directory, checking headers, boundaries,
    and discovered layer indices.
    """
    t0 = time.time()
    p_dir = Path(model_dir)
    shard_paths = sorted(list(p_dir.glob("*.safetensors")))

    # Check for active .aria2 files
    aria2_files = list(p_dir.glob("*.aria2"))

    results: List[ShardHealth] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(verify_single_shard, shard_paths))

    healthy = sum(1 for s in results if s.is_healthy)
    corrupted = sum(1 for s in results if not s.is_healthy and s.status != "INCOMPLETE_DOWNLOAD")
    incompletes = sum(1 for s in results if s.status == "INCOMPLETE_DOWNLOAD") + len(aria2_files)
    total_bytes = sum(s.physical_size_bytes for s in results)
    total_tensors = sum(s.tensor_count for s in results if s.is_healthy)

    # Extract layer indices from healthy shard headers
    layer_pat = re.compile(r"model\.layers\.(\d+)\.")
    discovered_layers_set = set()

    for s in results:
        if s.is_healthy:
            try:
                with open(s.file_path, "rb") as f:
                    hlen = struct.unpack("<Q", f.read(8))[0]
                    hdr = json.loads(f.read(hlen).decode("utf-8"))
                    for k in hdr.keys():
                        m = layer_pat.search(k)
                        if m:
                            discovered_layers_set.add(int(m.group(1)))
            except Exception:
                pass

    duration = time.time() - t0
    return ArchiveHealthReport(
        model_dir=p_dir,
        total_shards=len(shard_paths),
        healthy_shards=healthy,
        corrupted_shards=corrupted,
        incomplete_aria2_shards=incompletes,
        total_bytes=total_bytes,
        total_tensors=total_tensors,
        discovered_layers=sorted(list(discovered_layers_set)),
        shards=results,
        duration_seconds=duration,
    )


@dataclass
class TritContainerHealth:
    file_name: str
    file_path: Path
    layer_idx: int
    size_bytes: int
    scale_method: str
    gatekeeper_dtype: str
    num_tensors: int
    sha256_match: Optional[bool] = None  # None if no hash in manifest, True if match, False if mismatch
    manifest_size_match: Optional[bool] = None
    manifest_error: Optional[str] = None


@dataclass
class TritVerificationReport:
    folder: Path
    containers: List[TritContainerHealth]
    manifest_found: bool
    manifest_mismatches: List[str]

    def render_markdown(self) -> str:
        lines = [
            f"# Trit Verification Report: `{self.folder}`",
            f"- Containers found: {len(self.containers)}",
            f"- Manifest present: {self.manifest_found}",
        ]
        if self.manifest_mismatches:
            lines.append("## Manifest Discrepancies / Mismatches")
            for m in self.manifest_mismatches:
                lines.append(f"- ⚠️ {m}")
        lines.extend([
            "",
            "| File | Layer | Size (MB) | Scale Method | GK Dtype | Tensors | Hash Check | Size Check |",
            "|---|---|---|---|---|---|---|---|",
        ])
        for c in self.containers:
            mb = c.size_bytes / (1024 ** 2)
            hash_str = "N/A" if c.sha256_match is None else ("PASS" if c.sha256_match else "FAIL")
            size_str = "N/A" if c.manifest_size_match is None else ("PASS" if c.manifest_size_match else "FAIL")
            lines.append(
                f"| `{c.file_name}` | {c.layer_idx} | {mb:.1f} | {c.scale_method} | {c.gatekeeper_dtype} | {c.num_tensors} | {hash_str} | {size_str} |"
            )
        return "\n".join(lines)


def verify_trit_directory(
    directory: Union[str, Path],
    no_hash: bool = False,
) -> TritVerificationReport:
    """
    Read-only verification of all layer_*.trit files in a directory against conversion_manifest.json (W2-11 + L-4).
    Reports file sizes, scale_method, gatekeeper dtype, tensor count, and SHA-256 integrity when recorded.
    """
    from tritforge.container import read_trit_header
    from tritforge.converter import file_sha256

    p_dir = Path(directory)
    manifest_path = p_dir / "conversion_manifest.json"
    manifest: Dict[str, Any] = {}
    manifest_found = manifest_path.exists()
    if manifest_found:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}

    manifest_metrics = manifest.get("metrics", {})
    manifest_layers = set(manifest.get("completed_layers", []))
    manifest_mismatches: List[str] = []

    containers: List[TritContainerHealth] = []
    found_layer_indices: set[int] = set()

    trit_files = sorted(list(p_dir.glob("layer_*.trit")))
    for p in trit_files:
        try:
            meta, _ = read_trit_header(p)
            l_idx = meta.get("layer_idx", -1)
            found_layer_indices.add(l_idx)
            tensors = meta.get("tensors", {})
            num_tensors = len(tensors)

            # Inspect scale_method from quantized matrices
            scale_method = "unknown"
            for t_info in tensors.values():
                if t_info.get("type") == "quantized_trit_matrix":
                    scale_method = t_info.get("scale_method", "unknown")
                    break

            # Inspect gatekeeper dtype from container metadata or raw arrays
            gk_dtype = meta.get("gatekeeper_dtype", "unknown")
            if gk_dtype == "unknown":
                for t_info in tensors.values():
                    if t_info.get("type") == "raw_array":
                        gk_dtype = t_info.get("dtype", "unknown")
                        break

            size_bytes = p.stat().st_size
            sha256_match: Optional[bool] = None
            size_match: Optional[bool] = None
            manifest_err: Optional[str] = None

            # Compare with manifest if available
            stored_m = manifest_metrics.get(str(l_idx))
            if stored_m:
                expected_size = stored_m.get("output_bytes")
                if expected_size is not None:
                    size_match = (size_bytes == expected_size)
                    if not size_match:
                        manifest_mismatches.append(
                            f"{p.name}: size mismatch (file is {size_bytes} bytes, manifest recorded {expected_size})"
                        )

                expected_sha = stored_m.get("sha256")
                if expected_sha and not no_hash:
                    actual_sha = file_sha256(p)
                    sha256_match = (actual_sha == expected_sha)
                    if not sha256_match:
                        manifest_mismatches.append(
                            f"{p.name}: SHA-256 hash mismatch (file is {actual_sha}, manifest recorded {expected_sha})"
                        )
            elif manifest_found:
                manifest_mismatches.append(f"{p.name}: layer {l_idx} found on disk but missing from manifest metrics")

            # Check expert completeness (D1)
            exp_indices = meta.get("expert_indices")
            num_experts = meta.get("num_experts")
            if exp_indices is not None:
                expected_exp_tensors = 3 * len(exp_indices)
                actual_exp_tensors = sum(
                    1 for t_name in tensors.keys() if t_name.startswith("experts.")
                )
                if actual_exp_tensors != expected_exp_tensors:
                    manifest_mismatches.append(
                        f"{p.name}: incomplete expert tensors ({actual_exp_tensors} found, expected {expected_exp_tensors} for {len(exp_indices)} experts)"
                    )
                if num_experts is not None and num_experts != len(exp_indices):
                    manifest_mismatches.append(
                        f"{p.name}: num_experts mismatch ({num_experts} in header, {len(exp_indices)} in expert_indices)"
                    )

            containers.append(
                TritContainerHealth(
                    file_name=p.name,
                    file_path=p,
                    layer_idx=l_idx,
                    size_bytes=size_bytes,
                    scale_method=scale_method,
                    gatekeeper_dtype=gk_dtype,
                    num_tensors=num_tensors,
                    sha256_match=sha256_match,
                    manifest_size_match=size_match,
                    manifest_error=manifest_err,
                )
            )
        except Exception as e:
            manifest_mismatches.append(f"{p.name}: header read error: {e}")

    # Check for layers in manifest but missing on disk
    for m_l in manifest_layers:
        if m_l not in found_layer_indices:
            manifest_mismatches.append(f"Manifest marks layer {m_l} as completed, but file layer_{m_l:02d}.trit is missing from disk")

    return TritVerificationReport(
        folder=p_dir,
        containers=containers,
        manifest_found=manifest_found,
        manifest_mismatches=manifest_mismatches,
    )
