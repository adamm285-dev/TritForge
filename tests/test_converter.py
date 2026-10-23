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


@pytest.mark.vault
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
            scale_method="absmax",
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

        # If container file exists, init rejects conflicting container (D2)
        with pytest.raises(ValueError, match="conflicts with requested 'absmax'"):
            BatchConverter(in_path, out_path, block_size=16, scale_method="absmax")

        # If container file is removed but manifest remains, manifest check rejects mismatch on resume (L-6)
        for p in out_path.glob("layer_*.trit"):
            p.unlink()
        c2 = BatchConverter(in_path, out_path, block_size=16, scale_method="absmax")
        with pytest.raises(ValueError, match="Manifest configuration mismatch for 'scale_method'"):
            c2.convert_all(layers=[0], expert_indices=[0], resume=True)


def test_converter_manifest_missing_key_rejected_b2():
    """B2: Resuming against a manifest with missing keys (like baseline without scale_method) must raise ValueError."""
    with tempfile.TemporaryDirectory() as tmp_in_dir, tempfile.TemporaryDirectory() as tmp_out_dir:
        in_path = Path(tmp_in_dir)
        out_path = Path(tmp_out_dir)

        tensors = {
            "model.layers.0.mlp.experts.0.gate_proj.weight": np.ones((16, 16), dtype=np.float32),
            "model.layers.0.mlp.experts.0.up_proj.weight": np.ones((16, 16), dtype=np.float32),
            "model.layers.0.mlp.experts.0.down_proj.weight": np.ones((16, 16), dtype=np.float32),
        }
        _create_mock_safetensors(in_path / "model-00001.safetensors", tensors)

        # Write baseline-shaped manifest without scale_method or gatekeeper_dtype
        legacy_manifest = {
            "model_name": "test_model",
            "packing_mode": "base3",
            "scales_dtype": "float16",
            "block_size": 16,
            "completed_layers": [0],
            "layer_files": {"0": str(out_path / "layer_00.trit")},
            "metrics": {},
        }
        (out_path / "conversion_manifest.json").write_text(json.dumps(legacy_manifest), encoding="utf-8")

        # Creating BatchConverter and trying to resume must fail because scale_method is missing
        converter = BatchConverter(in_path, out_path, block_size=16)
        with pytest.raises(ValueError, match="key is missing from existing manifest"):
            converter.convert_all(layers=[0], expert_indices=[0], resume=True)


def test_converter_failed_layer_recorded_and_not_retried_w2_15():
    """W2-15: Failed layer is recorded in manifest and not retried automatically."""
    with tempfile.TemporaryDirectory() as tmp_in_dir, tempfile.TemporaryDirectory() as tmp_out_dir:
        in_path = Path(tmp_in_dir)
        out_path = Path(tmp_out_dir)

        # Non-finite weight to force conversion failure
        nan_arr = np.ones((16, 16), dtype=np.float32)
        nan_arr[0, 0] = np.nan
        tensors = {
            "model.layers.0.mlp.experts.0.gate_proj.weight": nan_arr,
            "model.layers.0.mlp.experts.0.up_proj.weight": np.ones((16, 16), dtype=np.float32),
            "model.layers.0.mlp.experts.0.down_proj.weight": np.ones((16, 16), dtype=np.float32),
        }
        _create_mock_safetensors(in_path / "model-00001.safetensors", tensors)

        converter = BatchConverter(in_path, out_path, block_size=16)
        with pytest.raises(ValueError, match="Non-finite"):
            converter.convert_all(layers=[0], expert_indices=[0], resume=True)

        manifest = json.loads((out_path / "conversion_manifest.json").read_text(encoding="utf-8"))
        assert "0" in manifest["failed_layers"]
        assert "Non-finite" in manifest["failed_layers"]["0"]

        # Subsequent resume attempt must refuse to retry automatically
        with pytest.raises(RuntimeError, match="previously failed with error"):
            converter.convert_all(layers=[0], expert_indices=[0], resume=True)


def test_verify_trit_directory_reports_integrity_w2_11():
    """W2-11 + L-4: verify-trit reports container metadata and checks against manifest."""
    from tritforge.verifier import verify_trit_directory

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

        report = verify_trit_directory(out_path, no_hash=False)
        assert report.manifest_found is True
        assert len(report.containers) == 1
        c = report.containers[0]
        assert c.layer_idx == 0
        assert c.scale_method == "ls_optimal"
        assert c.gatekeeper_dtype == "bfloat16"
        assert c.sha256_match is True
        assert c.manifest_size_match is True
        assert not report.manifest_mismatches


def test_converter_cross_shard_fp8_scale_resolution_l1():
    """L-1: Two-shard fixture where weight is in shard 1 and weight_scale_inv is in shard 2.
    Asserts dequantized weight equals expected product, and fails if scale is sought in shard 1."""
    with tempfile.TemporaryDirectory() as tmp_in_dir, tempfile.TemporaryDirectory() as tmp_out_dir:
        in_path = Path(tmp_in_dir)
        out_path = Path(tmp_out_dir)

        # 256x256 FP8 weight in shard 1 (2x2 scale grid, block_size 128)
        shape = (256, 256)
        s_shape = (2, 2)
        raw_fp8 = np.full(shape, 32, dtype=np.uint8)  # constant uint8 code
        # In E4M3, byte 32 (0x20): s=0, e=4, m=0 -> 2^(4-7)*(1+0) = 2^(-3) = 0.125
        w_f32_decoded = np.full(shape, 0.125, dtype=np.float32)

        # BF16 scales in shard 2
        # Say scale = 2.0. BF16 representation for 2.0 is 0x4000
        scale_bf16 = np.full(s_shape, 0x4000, dtype=np.uint16)
        scale_f32 = np.full(s_shape, 2.0, dtype=np.float32)
        expected_w = w_f32_decoded * 2.0

        t_name = "model.layers.0.mlp.experts.0.gate_proj.weight"
        scale_name = f"{t_name}_scale_inv"

        # Shard 1: Gatekeeper + expert weight only (NO scale_inv in shard 1)
        h1 = {
            "model.layers.0.mlp.gate.weight": {
                "dtype": "F32", "shape": [1, 256], "data_offsets": [0, 1 * 256 * 4]
            },
            t_name: {
                "dtype": "F8_E4M3", "shape": list(shape), "data_offsets": [1024, 1024 + raw_fp8.nbytes]
            }
        }
        h1_bytes = json.dumps(h1).encode("utf-8")
        with open(in_path / "model-00001-of-00002.safetensors", "wb") as f:
            f.write(struct.pack("<Q", len(h1_bytes)))
            f.write(h1_bytes)
            f.write(np.zeros((1, 256), dtype=np.float32).tobytes())
            f.write(raw_fp8.tobytes())

        # Shard 2: Scale inv tensor only
        h2 = {
            scale_name: {
                "dtype": "BF16", "shape": list(s_shape), "data_offsets": [0, scale_bf16.nbytes]
            }
        }
        h2_bytes = json.dumps(h2).encode("utf-8")
        with open(in_path / "model-00002-of-00002.safetensors", "wb") as f:
            f.write(struct.pack("<Q", len(h2_bytes)))
            f.write(h2_bytes)
            f.write(scale_bf16.tobytes())

        # Build converter and convert
        converter = BatchConverter(in_path, out_path, block_size=128, scale_method="ls_optimal")
        trit_path, metrics = converter.convert_layer(0, expert_indices=[0])
        assert trit_path.exists()

        # Load quantized matrix and check scale/weights
        from tritforge.container import load_trit_layer
        loaded = load_trit_layer(trit_path, expert_idx=0, proj="gate_proj")
        # Reconstruction: Q * scales should approximate expected_w (which is constant 0.25)
        # ls_optimal ternary on constant 0.25 will produce trits +1 and scale 0.25
        recon = loaded.reconstruct()
        np.testing.assert_allclose(recon, expected_w, rtol=1e-3, atol=1e-3)


def test_failed_conversion_leaves_no_layer_file_or_tmp_d1():
    """D1 (a): A failed conversion due to NaN in expert must call abort() and leave no layer file or tmp files."""
    with tempfile.TemporaryDirectory() as tmp_in_dir, tempfile.TemporaryDirectory() as tmp_out_dir:
        in_path = Path(tmp_in_dir)
        out_path = Path(tmp_out_dir)

        # Expert 0 is normal, Expert 1 has NaN in down_proj
        tensors = {
            "model.layers.0.mlp.experts.0.gate_proj.weight": np.ones((16, 16), dtype=np.float32),
            "model.layers.0.mlp.experts.0.up_proj.weight": np.ones((16, 16), dtype=np.float32),
            "model.layers.0.mlp.experts.0.down_proj.weight": np.ones((16, 16), dtype=np.float32),
            "model.layers.0.mlp.experts.1.gate_proj.weight": np.ones((16, 16), dtype=np.float32),
            "model.layers.0.mlp.experts.1.up_proj.weight": np.ones((16, 16), dtype=np.float32),
            "model.layers.0.mlp.experts.1.down_proj.weight": np.full((16, 16), np.nan, dtype=np.float32),
        }
        _create_mock_safetensors(in_path / "model-00001.safetensors", tensors)

        converter = BatchConverter(in_path, out_path, block_size=16)
        with pytest.raises(ValueError, match="Non-finite.*values detected"):
            converter.convert_layer(0, expert_indices=[0, 1])

        # Assert no layer_00.trit, .payload.tmp, or .atomic_tmp exist
        layer_file = out_path / "layer_00.trit"
        payload_tmp = out_path / "layer_00.payload.tmp"
        atomic_tmp = out_path / "layer_00.trit.atomic_tmp"

        assert not layer_file.exists(), f"layer_00.trit should not exist after failure: {layer_file}"
        assert not payload_tmp.exists(), f"payload_tmp should not exist after failure: {payload_tmp}"
        assert not atomic_tmp.exists(), f"atomic_tmp should not exist after failure: {atomic_tmp}"


def test_failed_conversion_preserves_previous_good_container_e1(tmp_path):
    """E1: StreamingTritWriter.abort() must not delete the previous good container.
    1. Convert layer 0 in temp folder, record SHA-256 of layer_00.trit.
    2. Rewrite source with NaN in expert 1 down_proj.
    3. Converting layer 0 again raises.
    4. layer_00.trit still exists with same SHA-256, no .payload.tmp or .atomic_tmp remains.
    5. Manifest records layer 0 in failed_layers and verified_done_layers doesn't count it as done."""
    from tritforge.overnight import verified_done_layers
    from tritforge.converter import file_sha256

    in_path = tmp_path / "in"
    out_path = tmp_path / "out"
    in_path.mkdir()
    out_path.mkdir()

    # Step 1: Successful conversion of layer 0 with experts 0 and 1
    tensors_good = {
        "model.layers.0.mlp.experts.0.gate_proj.weight": np.ones((16, 16), dtype=np.float32),
        "model.layers.0.mlp.experts.0.up_proj.weight": np.ones((16, 16), dtype=np.float32),
        "model.layers.0.mlp.experts.0.down_proj.weight": np.ones((16, 16), dtype=np.float32),
        "model.layers.0.mlp.experts.1.gate_proj.weight": np.ones((16, 16), dtype=np.float32),
        "model.layers.0.mlp.experts.1.up_proj.weight": np.ones((16, 16), dtype=np.float32),
        "model.layers.0.mlp.experts.1.down_proj.weight": np.ones((16, 16), dtype=np.float32),
    }
    _create_mock_safetensors(in_path / "model-00001.safetensors", tensors_good)

    converter = BatchConverter(in_path, out_path, block_size=16)
    converter.convert_all(layers=[0], expert_indices=[0, 1], resume=False)

    layer_file = out_path / "layer_00.trit"
    assert layer_file.exists()
    good_sha = file_sha256(layer_file)
    assert verified_done_layers(out_path) == {0}

    # Step 2: Corrupt source by rewriting with NaN in expert 1 down_proj
    tensors_corrupt = dict(tensors_good)
    tensors_corrupt["model.layers.0.mlp.experts.1.down_proj.weight"] = np.full((16, 16), np.nan, dtype=np.float32)
    _create_mock_safetensors(in_path / "model-00001.safetensors", tensors_corrupt)

    # Step 3: Re-converting layer 0 raises
    c2 = BatchConverter(in_path, out_path, block_size=16)
    with pytest.raises(ValueError, match="Non-finite.*values detected"):
        c2.convert_all(layers=[0], expert_indices=[0, 1], resume=False)

    # Step 4: layer_00.trit still exists with same SHA-256, no tmp files left
    assert layer_file.exists(), "Previous good layer_00.trit was deleted by abort()!"
    assert file_sha256(layer_file) == good_sha, "layer_00.trit was modified or corrupted!"

    payload_tmp = out_path / "layer_00.payload.tmp"
    atomic_tmp = out_path / "layer_00.trit.atomic_tmp"
    assert not payload_tmp.exists()
    assert not atomic_tmp.exists()

    # Step 5: Manifest recorded failure and verified_done_layers rejects it
    manifest = json.loads((out_path / "conversion_manifest.json").read_text(encoding="utf-8"))
    assert "0" in manifest["failed_layers"]
    assert verified_done_layers(out_path) == set()



def test_verified_done_layers_criteria_e2(tmp_path):
    """E2: Tests verified_done_layers on:
      a) header-valid container not in completed_layers: not done
      b) in completed_layers and in failed_layers: not done
      c) size mismatch: not done
      d) sha256 mismatch (same size, one byte changed): not done
      e) fully consistent layer: done (positive control)."""
    from tritforge.overnight import verified_done_layers
    from tritforge.container import StreamingTritWriter
    from tritforge.converter import file_sha256

    # Create 5 test containers using StreamingTritWriter
    # layer 0: valid, will be fully consistent (e)
    # layer 1: valid, will NOT be in completed_layers (a)
    # layer 2: valid, will be in completed_layers AND failed_layers (b)
    # layer 3: valid, will have size mismatch (c)
    # layer 4: valid, will have SHA-256 mismatch (d)
    for l_idx in range(5):
        w = StreamingTritWriter(tmp_path / f"layer_0{l_idx}.trit", layer_idx=l_idx)
        w.add_raw_array("data", np.ones((8, 8), dtype=np.float32) * (l_idx + 1))
        w.close()

    # Precompute hashes and sizes
    sizes = {l_idx: (tmp_path / f"layer_0{l_idx}.trit").stat().st_size for l_idx in range(5)}
    shas = {l_idx: file_sha256(tmp_path / f"layer_0{l_idx}.trit") for l_idx in range(5)}

    # Modify layer 4 on disk by altering one byte in place to create SHA-256 mismatch with same size
    with open(tmp_path / "layer_04.trit", "r+b") as f:
        # Seek near end of file, read byte, flip it
        f.seek(sizes[4] - 10)
        orig_byte = f.read(1)
        flipped_byte = bytes([(orig_byte[0] ^ 0xFF)])
        f.seek(sizes[4] - 10)
        f.write(flipped_byte)
    assert (tmp_path / "layer_04.trit").stat().st_size == sizes[4]
    assert file_sha256(tmp_path / "layer_04.trit") != shas[4]

    # Build manifest
    manifest_data = {
        "completed_layers": [0, 2, 3, 4],  # layer 1 omitted (case a)
        "failed_layers": {"2": "Synthetic failure"},  # layer 2 failed (case b)
        "metrics": {
            "0": {"output_bytes": sizes[0], "sha256": shas[0]},  # case e: fully consistent
            "1": {"output_bytes": sizes[1], "sha256": shas[1]},  # case a: not in completed_layers
            "2": {"output_bytes": sizes[2], "sha256": shas[2]},  # case b: in failed_layers
            "3": {"output_bytes": sizes[3] + 128, "sha256": shas[3]},  # case c: size mismatch
            "4": {"output_bytes": sizes[4], "sha256": shas[4]},  # case d: sha mismatch
        },
    }
    (tmp_path / "conversion_manifest.json").write_text(json.dumps(manifest_data), encoding="utf-8")

    done = verified_done_layers(tmp_path)
    assert done == {0}, f"Expected only layer 0 to be verified done, got {done}"



def test_verify_trit_flags_missing_expert_tensors_d1():
    """D1 (c): verify-trit reports manifest mismatches if expert tensors are missing or incomplete."""
    from tritforge.verifier import verify_trit_directory
    from tritforge.container import StreamingTritWriter

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = Path(tmp_dir)

        # Write container with expert_indices=[0, 1] (expecting 6 expert tensors) but add only 3
        writer = StreamingTritWriter(
            out_dir / "layer_00.trit",
            layer_idx=0,
            extra_metadata={"num_experts": 2, "expert_indices": [0, 1]},
        )
        # Add 3 tensors for expert 0, none for expert 1
        from tritforge.quantize import absmax_ternary_quantize
        q = absmax_ternary_quantize(np.ones((4, 4), dtype=np.float32), block_size=4)
        writer.add_quantized_matrix("experts.0.gate_proj", q)
        writer.add_quantized_matrix("experts.0.up_proj", q)
        writer.add_quantized_matrix("experts.0.down_proj", q)
        writer.close()

        report = verify_trit_directory(out_dir, no_hash=True)
        assert len(report.containers) == 1
        assert any("incomplete expert tensors" in m for m in report.manifest_mismatches)


def test_converter_rejects_conflicting_scale_method_in_same_folder_d2():
    """D2: A folder holding a BF16 + 'optimal' container makes a BF16 + 'ls_optimal' converter raise."""
    with tempfile.TemporaryDirectory() as tmp_in_dir, tempfile.TemporaryDirectory() as tmp_out_dir:
        in_path = Path(tmp_in_dir)
        out_path = Path(tmp_out_dir)

        tensors = {
            "model.layers.0.mlp.gate.weight": np.ones((2, 16), dtype=np.float32),
            "model.layers.0.mlp.experts.0.gate_proj.weight": np.ones((16, 16), dtype=np.float32),
            "model.layers.0.mlp.experts.0.up_proj.weight": np.ones((16, 16), dtype=np.float32),
            "model.layers.0.mlp.experts.0.down_proj.weight": np.ones((16, 16), dtype=np.float32),
        }
        _create_mock_safetensors(in_path / "model-00001.safetensors", tensors)

        # Convert layer 0 with BF16 + optimal
        c_opt = BatchConverter(
            in_path, out_path, block_size=16,
            gatekeeper_dtype="bfloat16", scale_method="optimal"
        )
        c_opt.convert_all(layers=[0], expert_indices=[0], resume=False)
        assert (out_path / "layer_00.trit").exists()

        # Attempt to instantiate or run BatchConverter with BF16 + ls_optimal in same out_path
        with pytest.raises(ValueError, match="scale_method 'optimal'.*conflicts with requested 'ls_optimal'"):
            BatchConverter(
                in_path, out_path, block_size=16,
                gatekeeper_dtype="bfloat16", scale_method="ls_optimal"
            )


def test_converter_dirty_tree_guard_raises_and_allow_dirty_bypasses_d3(monkeypatch):
    """D3: BatchConverter raises when tritforge/ has uncommitted changes, unless allow_dirty=True."""
    with tempfile.TemporaryDirectory() as tmp_in_dir, tempfile.TemporaryDirectory() as tmp_out_dir:
        in_path = Path(tmp_in_dir)
        out_path = Path(tmp_out_dir)

        # Mock check_git_dirty to return True (simulating dirty working tree)
        import tritforge.converter
        monkeypatch.setattr(tritforge.converter, "check_git_dirty", lambda: True)

        with pytest.raises(RuntimeError, match="Refusing to run BatchConverter with uncommitted changes"):
            BatchConverter(in_path, out_path)

        # allow_dirty=True bypasses the check successfully
        c = BatchConverter(in_path, out_path, allow_dirty=True)
        assert c.allow_dirty is True


def test_overnight_preflight_dirty_tree_exits_nonzero_e3(monkeypatch):
    """E3: tritforge.overnight.preflight() exits non-zero when check_git_dirty returns True."""
    from tritforge import overnight
    monkeypatch.setattr(overnight, "check_git_dirty", lambda: True)

    with pytest.raises(SystemExit) as exc_info:
        overnight.preflight()
    assert exc_info.value.code != 0


def test_overnight_run_calls_preflight_first_e3(monkeypatch, tmp_path):
    """E3: tritforge.overnight.run() calls preflight() before anything else.
    When preflight is patched to raise a sentinel exception, run() raises it and leaves tmp_out empty.
    time.sleep is patched to raise so regressions fail fast."""
    from tritforge import overnight

    class PreflightSentinel(Exception):
        pass

    monkeypatch.setattr(overnight, "preflight", lambda: (_ for _ in ()).throw(PreflightSentinel("preflight called")))
    monkeypatch.setattr(overnight.time, "sleep", lambda s: (_ for _ in ()).throw(RuntimeError("Regression: overnight looped")))

    tmp_in = tmp_path / "in"
    tmp_out = tmp_path / "out"
    tmp_in.mkdir()

    with pytest.raises(PreflightSentinel, match="preflight called"):
        overnight.run(tmp_in, tmp_out, total_layers=1)

    assert not tmp_out.exists() or list(tmp_out.iterdir()) == []




