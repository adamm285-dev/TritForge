"""
TritForge: Clean-room, zero-dependency 1.58-Bit Post-Training Quantization (PTQ)
and Inference Engine for Sparse Mixture-of-Experts (MoE) architectures.
"""

__version__ = "0.3.0"
__author__ = "TritForge Core Team"

from tritforge.slice import (
    ExpertSlice,
    extract_expert_slice,
    extract_complete_expert,
    extract_layer0_gatekeeper,
    extract_token_embeddings,
    extract_tensor,
    read_safetensors_header,
    list_available_tensors,
)
from tritforge.quantize import (
    QuantizedTritMatrix,
    absmax_ternary_quantize,
    hessian_compensate_quantize,
    compute_frobenius_error,
    compute_cosine_similarity,
    compute_hessian_error,
    dequantize_trits,
    pack_trits_2bit,
    unpack_trits_2bit,
    pack_trits_base3,
    unpack_trits_base3,
)
from tritforge.moe import (
    Layer0AsymmetricMoE,
    MoERouter,
    TernaryExpert,
    SharedExpert,
    assemble_layer0_moe,
    collect_empirical_expert_covariances,
    rmsnorm,
    silu,
    ternary_gemm,
)
from tritforge.container import (
    TRIT_MAGIC,
    load_trit_layer,
    load_trit_tensor,
    read_trit_header,
    save_trit_file,
    save_trit_layer,
)
from tritforge.kernel import (
    SparseTritIndex,
    ZeroMultiplyGEMMEngine,
    benchmark_gemm_throughput,
    build_sparse_trit_index,
    zero_multiply_sparse_gemm,
    zero_multiply_vectorized_gemm,
)
from tritforge.dispatcher import (
    DispatchMetrics,
    MoEBatchDispatcher,
    create_batch_dispatcher_from_moe,
)
from tritforge.model import (
    AsymmetricAttention,
    AsymmetricTransformerBlock,
    GenerationResult,
    StreamingMoELayer,
    TritForgeConfig,
    TritForgeModel,
    create_dummy_asymmetric_model,
)
from tritforge.eval import (
    FootprintAuditReport,
    FootprintBreakdown,
    GenerationFidelityEvaluator,
    GenerationFidelityMetrics,
    ModelFootprintAuditor,
    log_phase4_evals_to_disk,
)
from tritforge.converter import (
    BatchConverter,
    ConversionSummary,
    LayerConversionMetrics,
    ShardIndex,
)


def cli_main():
    from tritforge.cli import main
    return main()

__all__ = [
    "ExpertSlice",
    "extract_expert_slice",
    "extract_complete_expert",
    "extract_layer0_gatekeeper",
    "extract_token_embeddings",
    "extract_tensor",
    "read_safetensors_header",
    "list_available_tensors",
    "QuantizedTritMatrix",
    "absmax_ternary_quantize",
    "hessian_compensate_quantize",
    "compute_frobenius_error",
    "compute_cosine_similarity",
    "compute_hessian_error",
    "dequantize_trits",
    "pack_trits_2bit",
    "unpack_trits_2bit",
    "pack_trits_base3",
    "unpack_trits_base3",
    "Layer0AsymmetricMoE",
    "MoERouter",
    "TernaryExpert",
    "SharedExpert",
    "assemble_layer0_moe",
    "collect_empirical_expert_covariances",
    "rmsnorm",
    "silu",
    "ternary_gemm",
    "TRIT_MAGIC",
    "load_trit_layer",
    "load_trit_tensor",
    "read_trit_header",
    "save_trit_file",
    "save_trit_layer",
    "SparseTritIndex",
    "ZeroMultiplyGEMMEngine",
    "benchmark_gemm_throughput",
    "build_sparse_trit_index",
    "zero_multiply_sparse_gemm",
    "zero_multiply_vectorized_gemm",
    "DispatchMetrics",
    "MoEBatchDispatcher",
    "create_batch_dispatcher_from_moe",
    "AsymmetricAttention",
    "AsymmetricTransformerBlock",
    "GenerationResult",
    "StreamingMoELayer",
    "TritForgeConfig",
    "TritForgeModel",
    "create_dummy_asymmetric_model",
    "FootprintAuditReport",
    "FootprintBreakdown",
    "GenerationFidelityEvaluator",
    "GenerationFidelityMetrics",
    "ModelFootprintAuditor",
    "log_phase4_evals_to_disk",
    "BatchConverter",
    "ConversionSummary",
    "LayerConversionMetrics",
    "ShardIndex",
    "cli_main",
]
