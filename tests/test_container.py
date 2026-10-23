"""
Tests for tritforge.container
=============================
Validates .trit binary format, header serialization, seek-based constant-memory reads,
physical disk size constraint (<= 3.70 MB for 2048x8192 slice, >= 75% reduction),
and Asymmetric MoE layer round-trip.
"""

from pathlib import Path
import numpy as np
import pytest

from tritforge.container import (
    TRIT_MAGIC,
    load_trit_layer,
    load_trit_tensor,
    read_trit_header,
    save_trit_file,
    save_trit_layer,
)
from tritforge.moe import Layer0AsymmetricMoE, MoERouter, SharedExpert, TernaryExpert
from tritforge.quantize import QuantizedTritMatrix, absmax_ternary_quantize


@pytest.fixture
def temp_trit_file(tmp_path):
    return tmp_path / "test_model.trit"


def test_trit_magic_header_validation(tmp_path):
    """Verify that corrupt or non-.trit files are rejected immediately."""
    corrupt_file = tmp_path / "corrupt.trit"
    corrupt_file.write_bytes(b"BADMAG\x00\x00\x00\x00")
    with pytest.raises(ValueError, match="Invalid .trit magic header"):
        read_trit_header(corrupt_file)


def test_raw_array_save_and_seek_load(temp_trit_file):
    """Verify raw numpy float32 tensors save and load accurately via seek."""
    arr = np.linspace(-5.0, 5.0, 1024, dtype=np.float32).reshape(32, 32)
    save_trit_file(temp_trit_file, {"test.tensor": arr})

    header, offset = read_trit_header(temp_trit_file)
    assert "test.tensor" in header["tensors"]
    assert header["tensors"]["test.tensor"]["shape"] == [32, 32]

    loaded = load_trit_tensor(temp_trit_file, "test.tensor")
    assert isinstance(loaded, np.ndarray)
    assert np.array_equal(arr, loaded)


def test_quantized_matrix_base3_and_2bit_roundtrip(temp_trit_file):
    """Verify QuantizedTritMatrix saves and loads losslessly in both packing modes."""
    rng = np.random.RandomState(42)
    w = rng.randn(64, 256).astype(np.float32) * 0.02

    for mode in ["base3", "2bit"]:
        qmat = absmax_ternary_quantize(w, scaling_mode="block", block_size=64)
        save_trit_file(temp_trit_file, {"qmat": qmat}, packing_mode=mode, scales_dtype="float16")

        loaded = load_trit_tensor(temp_trit_file, "qmat")
        assert isinstance(loaded, QuantizedTritMatrix)
        assert np.array_equal(qmat.trits, loaded.trits)
        assert np.allclose(qmat.scales, loaded.scales, atol=1e-3)
        assert loaded.original_shape == (64, 256)


def test_physical_disk_size_constraint(tmp_path):
    """
    CRITICAL SPECIFICATION CHECK:
    A saved expert slice (2048 x 8192 = 16,777,216 weights) must physically
    occupy <= 3.70 MB on disk (>= 75% reduction vs 16.78 MB FP8 / 33.55 MB FP16).
    """
    rng = np.random.RandomState(42)
    w_slice = rng.randn(2048, 8192).astype(np.float32) * 0.02
    qmat = absmax_ternary_quantize(w_slice, scaling_mode="block", block_size=128)

    file_path = tmp_path / "expert_slice.trit"
    total_bytes = save_trit_file(
        file_path,
        {"experts.0.gate_proj": qmat},
        layer_idx=0,
        packing_mode="base3",
        scales_dtype="float16",
    )

    size_mib = total_bytes / (1024 * 1024)
    size_mb = total_bytes / 1e6

    # Verify physical file size constraint
    assert total_bytes <= 3.70 * 1024 * 1024, f"File size {size_mib:.3f} MiB exceeds 3.70 MiB limit!"
    assert total_bytes <= 3.70 * 1e6, f"File size {size_mb:.3f} MB exceeds 3.70 MB limit!"

    # Verify reduction percentage vs 16.78 MB FP8 reference
    fp8_bytes = 2048 * 8192
    reduction_pct = (1.0 - total_bytes / fp8_bytes) * 100.0
    assert reduction_pct >= 75.0, f"Achieved reduction {reduction_pct:.2f}% is below 75% target!"


def test_save_and_seek_load_trit_layer(temp_trit_file):
    """Verify save_trit_layer and all 3 seek modes of load_trit_layer."""
    rng = np.random.RandomState(42)
    hidden_size, inter_size, num_experts, top_k = 64, 32, 4, 2
    block_size = 16

    router = MoERouter(rng.randn(num_experts, hidden_size).astype(np.float32) * 0.01)

    experts = {}
    for e in range(num_experts):
        experts[e] = TernaryExpert(
            expert_idx=e,
            gate_proj=absmax_ternary_quantize(rng.randn(inter_size, hidden_size).astype(np.float32) * 0.01, block_size=block_size),
            up_proj=absmax_ternary_quantize(rng.randn(inter_size, hidden_size).astype(np.float32) * 0.01, block_size=block_size),
            down_proj=absmax_ternary_quantize(rng.randn(hidden_size, inter_size).astype(np.float32) * 0.01, block_size=block_size),
        )

    shared = SharedExpert(
        gate_proj=rng.randn(inter_size, hidden_size).astype(np.float32) * 0.01,
        up_proj=rng.randn(inter_size, hidden_size).astype(np.float32) * 0.01,
        down_proj=rng.randn(hidden_size, inter_size).astype(np.float32) * 0.01,
    )

    layer = Layer0AsymmetricMoE(
        router=router,
        experts=experts,
        shared_expert=shared,
        post_attention_layernorm_weight=np.ones(hidden_size, dtype=np.float32),
        num_experts=num_experts,
        num_experts_per_tok=top_k,
    )

    # Save layer to .trit container
    save_trit_layer(temp_trit_file, layer_data=layer, packing_mode="base3", scales_dtype="float16")

    # Mode 1: Targeted single projection query (constant-memory seek)
    q_gate = load_trit_layer(temp_trit_file, expert_idx=2, proj="gate_proj")
    assert isinstance(q_gate, QuantizedTritMatrix)
    assert np.array_equal(q_gate.trits, experts[2].gate_proj.trits)

    # Mode 2: Targeted single expert query
    exp2 = load_trit_layer(temp_trit_file, expert_idx=2)
    assert isinstance(exp2, TernaryExpert)
    assert exp2.expert_idx == 2
    assert np.array_equal(exp2.up_proj.trits, experts[2].up_proj.trits)
    assert np.array_equal(exp2.down_proj.trits, experts[2].down_proj.trits)

    # Mode 3: Load complete layer
    loaded_layer = load_trit_layer(temp_trit_file)
    assert isinstance(loaded_layer, Layer0AsymmetricMoE)
    assert len(loaded_layer.experts) == num_experts

    # Forward pass equivalence check
    x = rng.randn(4, hidden_size).astype(np.float32)
    out_orig = layer.forward(x)
    out_loaded = loaded_layer.forward(x)
    assert np.allclose(out_orig, out_loaded, atol=1e-3)



def test_load_trit_layer_with_converter_hf_keys():
    """C-6: Verify load_trit_layer correctly loads container using converter's HF key format."""
    import tempfile
    from tritforge.container import StreamingTritWriter, load_trit_layer
    from tritforge.quantize import absmax_ternary_quantize
    from tritforge.moe import Layer0AsymmetricMoE

    hid, inter, n_exp = 64, 32, 4
    rng = np.random.RandomState(42)

    with tempfile.TemporaryDirectory() as tmp_dir:
        p = Path(tmp_dir) / "layer_hf_keys.trit"
        w = StreamingTritWriter(p, layer_idx=0)
        w.add_raw_array("mlp.gate.weight", rng.randn(n_exp, hid).astype(np.float32))
        w.add_raw_array("post_attention_layernorm.weight", np.ones(hid, np.float32))
        w.add_raw_array("mlp.shared_expert.gate_proj.weight", rng.randn(inter, hid).astype(np.float32))
        w.add_raw_array("mlp.shared_expert.up_proj.weight", rng.randn(inter, hid).astype(np.float32))
        w.add_raw_array("mlp.shared_expert.down_proj.weight", rng.randn(hid, inter).astype(np.float32))

        for e in range(n_exp):
            for proj, shp in (("gate_proj", (inter, hid)), ("up_proj", (inter, hid)), ("down_proj", (hid, inter))):
                w.add_quantized_matrix(f"experts.{e}.{proj}", absmax_ternary_quantize(rng.randn(*shp).astype(np.float32), block_size=16))
        w.close()

        # Mode 3 load
        layer = load_trit_layer(p)
        assert isinstance(layer, Layer0AsymmetricMoE)
        assert len(layer.experts) == n_exp
        assert layer.router is not None
        assert layer.shared_expert is not None
