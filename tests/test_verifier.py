"""
Tests for tritforge.verifier (Master Shard Integrity & Health Auditor)
=====================================================================
Validates header parsing, boundary checks, truncation detection, and CLI dispatch.
"""

from io import StringIO
import json
from pathlib import Path
import struct
import tempfile
import numpy as np
import pytest

from tritforge.cli import main
from tritforge.verifier import (
    ArchiveHealthReport,
    ShardHealth,
    verify_shard_directory,
    verify_single_shard,
)


def _write_mock_safetensors(
    file_path: Path,
    tensors: dict[str, np.ndarray],
    truncate_bytes: int = 0,
    corrupt_json: bool = False,
    overflow_offset: bool = False,
) -> None:
    """Helper to write test .safetensors files with deliberate fault injections."""
    header: dict[str, dict] = {}
    payload_parts: list[bytes] = []
    curr_offset = 0

    for name, arr in tensors.items():
        arr_bytes = arr.tobytes()
        arr_len = len(arr_bytes)
        dtype_str = "F32" if arr.dtype == np.float32 else "U8"
        end_offset = curr_offset + arr_len
        if overflow_offset:
            end_offset += 1000000  # Offset extends past EOF

        header[name] = {
            "dtype": dtype_str,
            "shape": list(arr.shape),
            "data_offsets": [curr_offset, end_offset],
        }
        payload_parts.append(arr_bytes)
        curr_offset += arr_len

    if corrupt_json:
        header_bytes = b"{invalid_json_header"
    else:
        header_bytes = json.dumps(header).encode("utf-8")

    with open(file_path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for part in payload_parts:
            f.write(part)

    if truncate_bytes > 0:
        total_len = file_path.stat().st_size
        with open(file_path, "r+b") as f:
            f.truncate(max(0, total_len - truncate_bytes))


def test_verify_healthy_shard():
    """Verify that a valid, well-formed shard reports HEALTHY."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        p = Path(tmp_dir) / "model-00001.safetensors"
        tensors = {"tensor_a": np.ones((10, 10), dtype=np.float32)}
        _write_mock_safetensors(p, tensors)

        health = verify_single_shard(p)
        assert health.is_healthy
        assert health.status == "HEALTHY"
        assert health.tensor_count == 1
        assert health.error_message is None


def test_verify_truncated_shard():
    """Verify that a file cut off mid-payload reports OFFSET_OUT_OF_BOUNDS."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        p = Path(tmp_dir) / "model-00002.safetensors"
        tensors = {"tensor_a": np.ones((100, 100), dtype=np.float32)}
        # Truncate 200 bytes from payload
        _write_mock_safetensors(p, tensors, truncate_bytes=200)

        health = verify_single_shard(p)
        assert not health.is_healthy
        assert health.status == "OFFSET_OUT_OF_BOUNDS"
        assert "payload requires" in health.error_message


def test_verify_corrupted_header():
    """Verify that invalid JSON header reports CORRUPT_HEADER."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        p = Path(tmp_dir) / "model-00003.safetensors"
        tensors = {"tensor_a": np.ones((10, 10), dtype=np.float32)}
        _write_mock_safetensors(p, tensors, corrupt_json=True)

        health = verify_single_shard(p)
        assert not health.is_healthy
        assert health.status == "CORRUPT_HEADER"


def test_verify_directory_audit():
    """Verify directory-wide health reporting with mixed healthy and corrupted shards."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        p_dir = Path(tmp_dir)
        tensors = {"model.layers.0.gate.weight": np.ones((10, 10), dtype=np.float32)}

        # 1 healthy
        _write_mock_safetensors(p_dir / "model-00001.safetensors", tensors)
        # 1 truncated
        _write_mock_safetensors(p_dir / "model-00002.safetensors", tensors, truncate_bytes=50)

        report = verify_shard_directory(p_dir)
        assert report.total_shards == 2
        assert report.healthy_shards == 1
        assert report.corrupted_shards == 1
        assert not report.all_healthy
        assert 0 in report.discovered_layers


def test_cli_verify_shards_command(capsys):
    """Verify CLI 'tritforge verify-shards' execution."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        p_dir = Path(tmp_dir)
        tensors = {"model.layers.0.weight": np.ones((5, 5), dtype=np.float32)}
        _write_mock_safetensors(p_dir / "model-00001.safetensors", tensors)

        code = main(["verify-shards", "-m", str(p_dir)])
        assert code == 0
        captured = capsys.readouterr()
        assert "Master Shard Archive Health & Integrity Audit" in captured.out
        assert "ALL SHARDS HEALTHY" in captured.out


def test_verify_trit_flags_manifest_size_mismatch_d8b():
    """D8b: verify_trit_directory reports manifest size mismatch when container size != manifest output_bytes.
    Fails if size_match check is forced to True."""
    from tritforge.verifier import verify_trit_directory
    from tritforge.container import StreamingTritWriter

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = Path(tmp_dir)

        # 1. Create a minimal valid layer_00.trit container
        writer = StreamingTritWriter(
            out_dir / "layer_00.trit",
            layer_idx=0,
            extra_metadata={"num_experts": 0, "expert_indices": []},
        )
        writer.add_raw_array("dummy", np.zeros((4, 4), dtype=np.float32))
        writer.close()

        container_file = out_dir / "layer_00.trit"
        actual_size = container_file.stat().st_size
        wrong_size = actual_size + 1024  # Deliberate size mismatch

        # 2. Write manifest with wrong output_bytes
        manifest_data = {
            "completed_layers": [0],
            "metrics": {
                "0": {
                    "output_bytes": wrong_size,
                    "sha256": "",
                }
            },
        }
        (out_dir / "conversion_manifest.json").write_text(json.dumps(manifest_data), encoding="utf-8")

        # 3. Verify directory
        report = verify_trit_directory(out_dir, no_hash=True)

        assert len(report.containers) == 1
        c = report.containers[0]
        # manifest_size_match must be False
        assert c.manifest_size_match is False, f"Expected manifest_size_match=False for size {actual_size} vs {wrong_size}"
        # manifest_mismatches list must contain the size mismatch error
        assert any("size mismatch" in m and f"file is {actual_size} bytes" in m for m in report.manifest_mismatches), (
            f"Expected size mismatch recorded in manifest_mismatches: {report.manifest_mismatches}"
        )

