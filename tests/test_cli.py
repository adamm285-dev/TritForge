"""
Tests for tritforge.cli (Phase 5 & Wave 1 Review Fixes)
======================================================
Validates argument parsing, subcommand dispatch, range parsing, audit outputs,
conversion CLI execution, failure of 'run' for unimplemented architectures,
and explicit 'demo' execution.
"""

from io import StringIO
import json
from pathlib import Path
import struct
import sys
import tempfile
import numpy as np
import pytest

from tritforge.cli import _parse_int_ranges, build_parser, main


def _create_mock_safetensors(file_path: Path, tensors: dict[str, np.ndarray]) -> None:
    """Helper to write a valid minimal .safetensors file."""
    header: dict[str, dict] = {}
    payload_parts: list[bytes] = []
    curr_offset = 0

    for name, arr in tensors.items():
        arr_bytes = arr.tobytes()
        arr_len = len(arr_bytes)
        dtype_str = "F32" if arr.dtype == np.float32 else "U8"
        header[name] = {
            "dtype": dtype_str,
            "shape": list(arr.shape),
            "data_offsets": [curr_offset, curr_offset + arr_len],
        }
        payload_parts.append(arr_bytes)
        curr_offset += arr_len

    header_bytes = json.dumps(header).encode("utf-8")
    with open(file_path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for part in payload_parts:
            f.write(part)


def test_parse_int_ranges():
    """Verify range string parser handles singles, commas, and hyphenated ranges."""
    assert _parse_int_ranges("0") == [0]
    assert _parse_int_ranges("0,1,2") == [0, 1, 2]
    assert _parse_int_ranges("0-3") == [0, 1, 2, 3]
    assert _parse_int_ranges("0, 2-4, 7") == [0, 2, 3, 4, 7]
    assert _parse_int_ranges("") == []


def test_cli_help():
    """Verify CLI prints help and exits with code 0."""
    code = main([])
    assert code == 0


def test_cli_audit_command(capsys):
    """Verify 'tritforge audit' runs and produces markdown table and json output."""
    # Table mode
    code = main(["audit"])
    assert code == 0
    captured = capsys.readouterr()
    assert "Full Model Hardware Footprint & Compression Audit" in captured.out
    assert "77." in captured.out

    # JSON mode
    code_json = main(["audit", "--format", "json"])
    assert code_json == 0
    captured_json = capsys.readouterr()
    data = json.loads(captured_json.out)
    assert "total_parameters" in data
    assert data["memory_reduction_vs_fp8_pct"] >= 75.0


def test_cli_run_command_fails(capsys):
    """I-4: Verify 'tritforge run' exits non-zero with not implemented message."""
    code = main(["run", "--prompt", "Hello TritForge"])
    assert code != 0
    captured = capsys.readouterr()
    assert "Inference not implemented for qwen3_5_moe yet (I-4)" in captured.err


def test_cli_demo_command(capsys):
    """I-4: Verify 'tritforge demo' runs toy demo model explicitly labeled as random."""
    code = main(["demo", "--prompt", "Hello TritForge", "--max-new-tokens", "5", "--temperature", "0.0"])
    assert code == 0
    captured = capsys.readouterr()
    assert "random toy model" in captured.out
    assert "Generated 5 tokens" in captured.out
    assert "Output Sequence Token IDs" in captured.out


def test_cli_convert_command():
    """Verify 'tritforge convert' executes batch conversion via CLI arguments."""
    with tempfile.TemporaryDirectory() as tmp_in, tempfile.TemporaryDirectory() as tmp_out:
        in_p = Path(tmp_in)
        out_p = Path(tmp_out)

        tensors = {
            "model.layers.0.mlp.gate.weight": np.random.randn(2, 32).astype(np.float32) * 0.01,
            "model.layers.0.mlp.experts.0.gate_proj.weight": np.random.randn(16, 32).astype(np.float32) * 0.02,
            "model.layers.0.mlp.experts.0.up_proj.weight": np.random.randn(16, 32).astype(np.float32) * 0.02,
            "model.layers.0.mlp.experts.0.down_proj.weight": np.random.randn(32, 16).astype(np.float32) * 0.02,
        }
        _create_mock_safetensors(in_p / "model-00001.safetensors", tensors)

        exit_code = main([
            "convert",
            "-i", str(in_p),
            "-o", str(out_p),
            "-l", "0",
            "-e", "0",
            "-p", "base3",
            "-b", "16",
        ])
        assert exit_code == 0
        assert (out_p / "layer_00.trit").exists()
        assert (out_p / "conversion_manifest.json").exists()
