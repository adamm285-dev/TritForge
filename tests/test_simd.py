"""Unit tests for the TritForge SIMD Vector Acceleration Subsystem."""

import pytest
import numpy as np
from tritforge.simd import SimdZeroMultiplyKernel


def test_simd_hardware_capabilities():
    caps = SimdZeroMultiplyKernel.get_hardware_capabilities()
    assert caps["has_avx2"] is True
    assert caps["has_avx512f"] is True
    assert caps["vector_width_bits"] >= 256
    assert caps["l3_cache_bytes"] == 96 * 1024 * 1024


def test_simd_lut_decoding_identity():
    # Test all possible trit values in a single packed byte
    # 0 = [-1, -1, -1, -1, -1]
    byte_zero = np.array([[0]], dtype=np.uint8)
    decoded_zero = SimdZeroMultiplyKernel.decode_packed_bytes(byte_zero)
    np.testing.assert_array_equal(decoded_zero[0], [-1, -1, -1, -1, -1])

    # 121 = [0, 0, 0, 0, 0] (since 1 + 3 + 9 + 27 + 81 = 121 in offset ternary)
    byte_mid = np.array([[121]], dtype=np.uint8)
    decoded_mid = SimdZeroMultiplyKernel.decode_packed_bytes(byte_mid)
    np.testing.assert_array_equal(decoded_mid[0], [0, 0, 0, 0, 0])

    # 242 = [+1, +1, +1, +1, +1]
    byte_max = np.array([[242]], dtype=np.uint8)
    decoded_max = SimdZeroMultiplyKernel.decode_packed_bytes(byte_max)
    np.testing.assert_array_equal(decoded_max[0], [1, 1, 1, 1, 1])


def test_simd_gemm_forward_correctness():
    # Verify mathematical equivalence against naive dot product
    batch_size = 2
    in_features = 10
    out_features = 4

    activations = np.random.randn(batch_size, in_features).astype(np.float32)

    # 10 features requires 2 packed bytes (5 trits each)
    packed_weights = np.random.randint(0, 243, size=(out_features, 2), dtype=np.uint8)
    scales = np.random.uniform(0.1, 2.0, size=out_features).astype(np.float32)

    # Reference calculation using decoded trits
    decoded_trits = SimdZeroMultiplyKernel.decode_packed_bytes(packed_weights)[:, :in_features]
    expected = (activations @ decoded_trits.T) * scales[np.newaxis, :]

    # Kernel execution
    result = SimdZeroMultiplyKernel.gemm_forward_vectorized(
        activations=activations,
        packed_weights=packed_weights,
        scales=scales,
        target_features=in_features,
    )

    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)
