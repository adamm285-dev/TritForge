"""
Tests for tritforge.converter (Phase 5)
========================================
Validates ShardIndex cataloging, layer-by-layer conversion, Base-3 packing export,
checkpointing, and resumption from partial conversion states.
"""

import json
from pathlib import Path
import struct
import tempfile
import numpy as np
import pytest

from tritforge.container import load_trit_layer, read_trit_header
from tritforge.converter import (
    BatchConverter,
    ConversionSummary,
    LayerConversionMetrics,
    ShardIndex,
)
from tritforge.slice import DEFAULT_MODEL_DIR


def _create_mock_safetensors(file_path: Path, tensors: dict[str, np.ndarray]) -> None:
    """Helper to write a valid minimal .safetensors file."""
    header: dict[str, dict] = {}
    payload_parts: list[bytes] = []
    curr_offset = 0

    for name, arr in tensors.items():
        arr_bytes = arr.tobytes()
        arr_len = len(arr_bytes)
        dtype_str = "F32" if arr.dtype == np.float32 else ("U8" if arr.dtype == np.uint8 else "F16")
        header[name] = {
            "dtype": dtype_str,
            "shape": list(arr.shape),
            "data_offsets": [curr_offset, curr_offset + arr_len],
        }
        payload_parts.append(arr_bytes)
        curr_offset += arr_len

    header_bytes = json.dumps(header).encode("utf-8")
    header_len = len(header_bytes)

    with open(file_path, "wb") as f:
        f.write(struct.pack("<Q", header_len))
        f.write(header_bytes)
        for part in payload_parts:
            f.write(part)


def test_shard_index_synthetic():
    """Verify ShardIndex correctly catalogs layers, experts, and global tensors."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create mock config
        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "test_moe",
            "num_hidden_layers": 2,
            "num_experts": 2,
        }), encoding="utf-8")

        # Shard 1: Layer 0 gatekeeper + Expert 0 & 1 gate_proj, up_proj
        tensors_shard1 = {
            "model.layers.0.mlp.gate.weight": np.ones((2, 64), dtype=np.float32),
            "model.layers.0.input_layernorm.weight": np.ones(64, dtype=np.float32),
            "model.layers.0.mlp.experts.0.gate_proj.weight": np.ones((32, 64), dtype=np.float32) * 0.1,
            "model.layers.0.mlp.experts.0.up_proj.weight": np.ones((32, 64), dtype=np.float32) * 0.2,
            "model.layers.0.mlp.experts.1.gate_proj.weight": np.ones((32, 64), dtype=np.float32) * 0.3,
            "model.layers.0.mlp.experts.1.up_proj.weight": np.ones((32, 64), dtype=np.float32) * 0.4,
        }
        _create_mock_safetensors(tmp_path / "model-00001.safetensors", tensors_shard1)

        # Shard 2: Layer 0 Expert 0 & 1 down_proj + Embeddings
        tensors_shard2 = {
            "model.embed_tokens.weight": np.ones((100, 64), dtype=np.float32),
            "model.layers.0.mlp.experts.0.down_proj.weight": np.ones((64, 32), dtype=np.float32) * 0.5,
            "model.layers.0.mlp.experts.1.down_proj.weight": np.ones((64, 32), dtype=np.float32) * 0.6,
        }
        _create_mock_safetensors(tmp_path / "model-00002.safetensors", tensors_shard2)

        idx = ShardIndex(tmp_path)
        assert 0 in idx.available_layers
        assert len(idx.available_layers) == 1
        assert "model.embed_tokens.weight" in idx.global_tensors

        exp_ids = idx.get_layer_expert_indices(0)
        assert exp_ids == [0, 1]

        layer_map = idx.layers[0]
        assert "model.layers.0.mlp.gate.weight" in layer_map.gatekeeper_tensors
        assert "gate_proj" in layer_map.expert_tensors[0]
        assert "up_proj" in layer_map.expert_tensors[0]
        assert "down_proj" in layer_map.expert_tensors[0]


def test_batch_converter_layer_conversion():
    """Verify BatchConverter converts a synthetic layer into a valid .trit container."""
    with tempfile.TemporaryDirectory() as tmp_in_dir, tempfile.TemporaryDirectory() as tmp_out_dir:
        in_path = Path(tmp_in_dir)
        out_path = Path(tmp_out_dir)

        # Populate synthetic model
        tensors_shard1 = {
            "model.layers.0.mlp.gate.weight": np.random.randn(2, 64).astype(np.float32) * 0.01,
            "model.layers.0.input_layernorm.weight": np.ones(64, dtype=np.float32),
            "model.layers.0.mlp.experts.0.gate_proj.weight": np.random.randn(32, 64).astype(np.float32) * 0.02,
            "model.layers.0.mlp.experts.0.up_proj.weight": np.random.randn(32, 64).astype(np.float32) * 0.02,
            "model.layers.0.mlp.experts.0.down_proj.weight": np.random.randn(64, 32).astype(np.float32) * 0.02,
        }
        _create_mock_safetensors(in_path / "model-00001.safetensors", tensors_shard1)

        converter = BatchConverter(
            input_dir=in_path,
            output_dir=out_path,
            packing_mode="base3",
            block_size=16,
        )

        trit_file, metrics = converter.convert_layer(0, expert_indices=[0])
        assert trit_file.exists()
        assert metrics.num_experts == 1
        assert metrics.compression_ratio_pct > 0.0

        # Validate that the converted container is readable by load_trit_layer
        loaded = load_trit_layer(trit_file, expert_idx=0)
        assert loaded.expert_idx == 0
        assert loaded.gate_proj.original_shape == (32, 64)
        assert loaded.up_proj.original_shape == (32, 64)
        assert loaded.down_proj.original_shape == (64, 32)


def test_batch_converter_checkpoint_and_resume():
    """Verify BatchConverter writes conversion_manifest.json and resumes without recomputing."""
    with tempfile.TemporaryDirectory() as tmp_in_dir, tempfile.TemporaryDirectory() as tmp_out_dir:
        in_path = Path(tmp_in_dir)
        out_path = Path(tmp_out_dir)

        tensors = {
            "model.layers.0.mlp.gate.weight": np.random.randn(2, 64).astype(np.float32) * 0.01,
            "model.layers.0.mlp.experts.0.gate_proj.weight": np.random.randn(16, 64).astype(np.float32) * 0.02,
            "model.layers.0.mlp.experts.0.up_proj.weight": np.random.randn(16, 64).astype(np.float32) * 0.02,
            "model.layers.0.mlp.experts.0.down_proj.weight": np.random.randn(64, 16).astype(np.float32) * 0.02,
        }
        _create_mock_safetensors(in_path / "model-00001.safetensors", tensors)

        converter = BatchConverter(
            input_dir=in_path,
            output_dir=out_path,
            packing_mode="base3",
            block_size=16,
        )

        # Pass 1: Convert all
        summary1 = converter.convert_all(layers=[0], resume=True)
        assert summary1.total_layers_converted == 1
        assert (out_path / "conversion_manifest.json").exists()
        assert (out_path / "layer_00.trit").exists()

        mtime_initial = (out_path / "layer_00.trit").stat().st_mtime

        # Pass 2: Resume (should detect layer 0 already converted and skip)
        summary2 = converter.convert_all(layers=[0], resume=True)
        assert summary2.total_layers_converted == 1
        mtime_resumed = (out_path / "layer_00.trit").stat().st_mtime

        assert mtime_initial == mtime_resumed  # File was not re-written


@pytest.mark.skipif(not (DEFAULT_MODEL_DIR / "model-00001-of-00213.safetensors").exists(),
                    reason="Master model shards not found on H: drive")
def test_live_master_shard_index():
    """Verify live cataloging of master Qwen 2.4T shards on H: drive."""
    idx = ShardIndex(DEFAULT_MODEL_DIR)
    assert len(idx.shard_files) > 0
    assert 0 in idx.available_layers
    assert len(idx.get_layer_expert_indices(0)) > 0



def test_converter_hessian_compensation_integration():
    """C-3: Verify convert_layer executes successfully with precomputed covariances."""
    with tempfile.TemporaryDirectory() as tmp_in_dir, tempfile.TemporaryDirectory() as tmp_out_dir:
        in_path = Path(tmp_in_dir)
        out_path = Path(tmp_out_dir)

        tensors_shard = {
            "model.layers.0.mlp.gate.weight": np.random.randn(2, 64).astype(np.float32) * 0.01,
            "model.layers.0.input_layernorm.weight": np.ones(64, dtype=np.float32),
            "model.layers.0.mlp.experts.0.gate_proj.weight": np.random.randn(32, 64).astype(np.float32) * 0.02,
            "model.layers.0.mlp.experts.0.up_proj.weight": np.random.randn(32, 64).astype(np.float32) * 0.02,
            "model.layers.0.mlp.experts.0.down_proj.weight": np.random.randn(64, 32).astype(np.float32) * 0.02,
        }
        _create_mock_safetensors(in_path / "model-00001.safetensors", tensors_shard)

        converter = BatchConverter(
            input_dir=in_path,
            output_dir=out_path,
            packing_mode="base3",
            block_size=16,
        )

        # Pass per-projection covariance dict
        cov_64 = np.eye(64, dtype=np.float32)
        cov_32 = np.eye(32, dtype=np.float32)
        empirical_covs = {
            0: {
                "gate_proj": cov_64,
                "up_proj": cov_64,
                "down_proj": cov_32,
            }
        }

        trit_file, metrics = converter.convert_layer(0, expert_indices=[0], empirical_covariances=empirical_covs)
        assert trit_file.exists()
        assert metrics.num_experts == 1

        loaded = load_trit_layer(trit_file, expert_idx=0)
        assert loaded.expert_idx == 0
        assert loaded.gate_proj.original_shape == (32, 64)
        assert loaded.down_proj.original_shape == (64, 32)


def test_converter_fp8_missing_scale_raises_l1():
    """L-1: FP8 tensor without scale_inv must raise ValueError."""
    with tempfile.TemporaryDirectory() as tmp_in_dir, tempfile.TemporaryDirectory() as tmp_out_dir:
        in_path = Path(tmp_in_dir)
        out_path = Path(tmp_out_dir)

        # Create raw FP8 tensor without matching _scale_inv
        raw_fp8 = np.ones((16, 16), dtype=np.uint8)
        header = {
            "model.layers.0.mlp.experts.0.gate_proj.weight": {
                "dtype": "F8_E4M3",
                "shape": [16, 16],
                "data_offsets": [0, raw_fp8.nbytes],
            }
        }
        h_bytes = json.dumps(header).encode("utf-8")
        with open(in_path / "model-00001.safetensors", "wb") as f:
            f.write(struct.pack("<Q", len(h_bytes)))
            f.write(h_bytes)
            f.write(raw_fp8.tobytes())

        converter = BatchConverter(in_path, out_path, block_size=16)
        with pytest.raises(ValueError, match="requires.*scale_inv"):
            converter.convert_layer(0, expert_indices=[0])


def test_converter_nan_guard_l3():
    """L-3: Non-finite (NaN or Inf) weights must raise ValueError."""
    with tempfile.TemporaryDirectory() as tmp_in_dir, tempfile.TemporaryDirectory() as tmp_out_dir:
        in_path = Path(tmp_in_dir)
        out_path = Path(tmp_out_dir)

        nan_arr = np.ones((16, 16), dtype=np.float32)
        nan_arr[0, 0] = np.nan
        tensors = {
            "model.layers.0.mlp.experts.0.gate_proj.weight": nan_arr,
            "model.layers.0.mlp.experts.0.up_proj.weight": np.ones((16, 16), dtype=np.float32),
            "model.layers.0.mlp.experts.0.down_proj.weight": np.ones((16, 16), dtype=np.float32),
        }
        _create_mock_safetensors(in_path / "model-00001.safetensors", tensors)

        converter = BatchConverter(in_path, out_path, block_size=16)
        with pytest.raises(ValueError, match="Non-finite.*detected"):
            converter.convert_layer(0, expert_indices=[0])


def test_converter_manifest_sha256_and_tamper_detection_l4():
    """L-4: Manifest records SHA-256 and detects corrupted/tampered files on resume."""
    with tempfile.TemporaryDirectory() as tmp_in_dir, tempfile.TemporaryDirectory() as tmp_out_dir:
        in_path = Path(tmp_in_dir)
        out_path = Path(tmp_out_dir)

        tensors = {
            "model.layers.0.mlp.experts.0.gate_proj.weight": np.ones((16, 16), dtype=np.float32),
            "model.layers.0.mlp.experts.0.up_proj.weight": np.ones((16, 16), dtype=np.float32),
            "model.layers.0.mlp.experts.0.down_proj.weight": np.ones((16, 16), dtype=np.float32),
        }
        _create_mock_safetensors(in_path / "model-00001.safetensors", tensors)

        converter = BatchConverter(in_path, out_path, block_size=16)
        converter.convert_all(layers=[0], expert_indices=[0], resume=True)

        manifest = converter._read_manifest()
        assert "0" in manifest["metrics"]
        sha = manifest["metrics"]["0"]["sha256"]
        assert len(sha) == 64

        # Tamper with output file
        out_file = out_path / "layer_00.trit"
        with open(out_file, "r+b") as f:
            f.seek(10)
            f.write(b"\xFF\xFF")

        # Resume should catch corruption and raise ValueError
        with pytest.raises(ValueError, match="Corrupted layer file"):
            converter.convert_all(layers=[0], expert_indices=[0], resume=True)


def test_converter_manifest_parameter_mismatch_guard_l6():
    """L-6: Resuming with mismatched conversion parameters must raise ValueError."""
    with tempfile.TemporaryDirectory() as tmp_in_dir, tempfile.TemporaryDirectory() as tmp_out_dir:
        in_path = Path(tmp_in_dir)
        out_path = Path(tmp_out_dir)

        tensors = {
            "model.layers.0.mlp.experts.0.gate_proj.weight": np.ones((16, 16), dtype=np.float32),
            "model.layers.0.mlp.experts.0.up_proj.weight": np.ones((16, 16), dtype=np.float32),
            "model.layers.0.mlp.experts.0.down_proj.weight": np.ones((16, 16), dtype=np.float32),
        }
        _create_mock_safetensors(in_path / "model-00001.safetensors", tensors)

        c1 = BatchConverter(in_path, out_path, block_size=16, scale_method="ls_optimal")
        c1.convert_all(layers=[0], expert_indices=[0], resume=True)

        # Attempt to resume with different scale_method
        c2 = BatchConverter(in_path, out_path, block_size=16, scale_method="absmax")
        with pytest.raises(ValueError, match="Manifest configuration mismatch for 'scale_method'"):
            c2.convert_all(layers=[0], expert_indices=[0], resume=True)
