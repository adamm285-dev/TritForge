"""
TritForge Unified Command-Line Interface (tritforge.cli)
=========================================================
Production command-line entry point for 1.58-bit Post-Training Quantization,
native .trit binary container conversion, streaming inference, and hardware audits.

Commands:
- tritforge convert: Converts raw multi-shard Safetensors into production .trit layers.
- tritforge run: Executes streaming inference & autoregressive generation.
- tritforge audit: Prints hardware footprint, parameter counts, and Spark reduction metrics.
- tritforge benchmark: Runs empirical drift and throughput benchmarks.

Enforces:
- Invariant 1: Zero external dependencies (pure Python standard library + NumPy).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from tritforge.container import load_trit_layer, read_trit_header
from tritforge.converter import BatchConverter, ShardIndex
from tritforge.eval import ModelFootprintAuditor
from tritforge.model import (
    AsymmetricAttention,
    AsymmetricTransformerBlock,
    StreamingMoELayer,
    TritForgeConfig,
    TritForgeModel,
    create_dummy_asymmetric_model,
)
from tritforge.slice import DEFAULT_MODEL_DIR


def _parse_int_ranges(spec: str) -> List[int]:
    """
    Parses string specifications like "0,1,2", "0-3", or "0,2-4" into sorted integer lists.
    """
    result: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            result.update(range(int(start_s), int(end_s) + 1))
        else:
            result.add(int(part))
    return sorted(list(result))


def handle_convert(args: argparse.Namespace) -> int:
    """Handles 'tritforge convert' subcommand."""
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"[ERROR] Input model directory does not exist: {input_dir}", file=sys.stderr)
        return 1

    layers = _parse_int_ranges(args.layers) if args.layers else None
    experts = _parse_int_ranges(args.experts) if args.experts else None

    print("=" * 70)
    print(" TritForge Production Batch Converter")
    print("=" * 70)
    print(f" Input Shards:   {input_dir}")
    print(f" Output Target:  {output_dir}")
    print(f" Packing Mode:   {args.packing} (1.58b Trits)")
    print(f" Block Size:     {args.block_size}")
    print(f" Layers:         {layers if layers else 'All Available'}")
    print(f" Experts/Layer:  {experts if experts else 'All Available'}")
    print(f" Resume Mode:    {not args.no_resume}")
    print("-" * 70)

    converter = BatchConverter(
        input_dir=input_dir,
        output_dir=output_dir,
        packing_mode=args.packing,
        block_size=args.block_size,
    )

    def progress_callback(info: dict) -> None:
        l_idx = info["layer_idx"]
        done = info["experts_completed"]
        tot = info["total_experts"]
        ram = info.get("current_ram_mb", 0.0)
        pct = (done / tot) * 100.0 if tot > 0 else 0.0
        sys.stdout.write(f"\r >> Layer {l_idx:02d} | Expert {done}/{tot} ({pct:.1f}%) | RAM: {ram:.1f} MB  ")
        sys.stdout.flush()

    summary = converter.convert_all(
        layers=layers,
        expert_indices=experts,
        resume=(not args.no_resume),
        progress_callback=progress_callback,
    )
    print("\n" + "=" * 70)
    print(summary.to_markdown_table())
    print(f"[SUCCESS] Model conversion complete. Manifest: {summary.manifest_path}")
    return 0


def handle_run(args: argparse.Namespace) -> int:
    """Handles 'tritforge run' subcommand. Fails non-zero until real engine exists (I-4)."""
    sys.stderr.write(
        "ERROR: Inference not implemented for qwen3_5_moe yet (I-4).\n"
        "The model architecture contains Gated DeltaNet attention and zero-centered RMSNorm.\n"
        "To run a toy in-memory demonstration with random weights, use: tritforge demo\n"
    )
    return 1


def handle_demo(args: argparse.Namespace) -> int:
    """Handles 'tritforge demo' subcommand: explicitly labeled random toy model demo (M-002, I-4)."""
    # MOCK(M-002): Demonstrates execution mechanics with a tiny randomly-initialized toy model
    print("=" * 70)
    print(" TritForge Toy Model Demonstration (DEMO ONLY: random toy model)")
    print("=" * 70)

    prompt = args.prompt or "What is Asymmetric Mixed-Precision?"
    print(f" Prompt:       {prompt}")
    print(f" Max Tokens:   {args.max_new_tokens}")
    print(f" Temperature:  {args.temperature}")
    print(f" Top-P:        {args.top_p}")
    print("-" * 70)

    # Run inference demonstration on random toy model
    model = create_dummy_asymmetric_model(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        num_experts=8,
        num_experts_per_tok=2,
        block_size=32,
    )

    # Tokenize dummy prompt into integers
    prompt_tokens = [int(abs(hash(w)) % 1000) for w in prompt.split()]
    if not prompt_tokens:
        prompt_tokens = [1, 2, 3]

    t0 = time.perf_counter()
    result = model.generate(
        prompt_token_ids=prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    t1 = time.perf_counter()

    print(f" [DEMO] Generated {len(result.generated_tokens[0])} tokens in {(t1 - t0)*1000.0:.2f} ms "
          f"({result.tokens_per_second:.1f} tok/s) using random toy model")
    print(f" Output Sequence Token IDs: {result.sequences[0].tolist()}")
    return 0


def handle_audit(args: argparse.Namespace) -> int:
    """Handles 'tritforge audit' subcommand."""
    spark_ram = float(args.spark_ram)
    report = ModelFootprintAuditor.audit_qwen_2_4t(spark_node_ram_gb=spark_ram)

    if args.format == "json":
        data = asdict(report)
        print(json.dumps(data, indent=2))
    else:
        print("=" * 70)
        print(" TritForge Hardware Footprint & Hardware Reduction Audit")
        print("=" * 70)
        print(report.to_markdown_table())
    return 0


def handle_benchmark(args: argparse.Namespace) -> int:
    """Handles 'tritforge benchmark' subcommand."""
    from tritforge.benchmark import main as bench_main
    return bench_main()


def handle_verify_shards(args: argparse.Namespace) -> int:
    """Handles 'tritforge verify-shards' subcommand."""
    from tritforge.verifier import verify_shard_directory

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"[ERROR] Model directory does not exist: {model_dir}", file=sys.stderr)
        return 1

    print("=" * 70)
    print(" TritForge Master Shard Archive Health & Integrity Auditor")
    print("=" * 70)
    print(f" Target Directory: {model_dir}")
    print(" Scanning shards for truncation, header validity, and byte bounds...")
    print("-" * 70)

    report = verify_shard_directory(model_dir, max_workers=args.workers)
    print(report.to_markdown_table())
    print("=" * 70)

    if report.all_healthy:
        print("[SUCCESS] All discovered shards are 100% healthy, intact, and ready for conversion.")
        return 0
    elif report.incomplete_aria2_shards > 0 and report.corrupted_shards == 0:
        print(f"[INFO] Download in progress: {report.healthy_shards}/{report.total_shards} complete, {report.incomplete_aria2_shards} active marker(s).")
        return 0
    else:
        print(f"[ALERT] {report.corrupted_shards} corrupted shard(s) detected. Please re-download affected files.")
        return 1


def build_parser() -> argparse.ArgumentParser:
    """Builds unified CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="tritforge",
        description="TritForge: 1.58-Bit Post-Training Quantization & Inference Engine for Sparse MoE",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # 1. convert
    p_conv = subparsers.add_parser("convert", help="Convert raw Safetensors shards into .trit container format")
    p_conv.add_argument("--input-dir", "-i", type=str, default=str(DEFAULT_MODEL_DIR),
                        help=f"Input directory with .safetensors shards (default: {DEFAULT_MODEL_DIR})")
    p_conv.add_argument("--output-dir", "-o", type=str, required=True,
                        help="Output directory to save .trit containers")
    p_conv.add_argument("--layers", "-l", type=str, default="",
                        help="Comma-separated layers or ranges to convert (e.g., '0' or '0-3')")
    p_conv.add_argument("--experts", "-e", type=str, default="",
                        help="Comma-separated expert indices or ranges to convert (e.g., '0-7')")
    p_conv.add_argument("--packing", "-p", choices=["base3", "2bit"], default="base3",
                        help="Trit packing scheme: 'base3' (1.6 b/w, default) or '2bit' (2.0 b/w)")
    p_conv.add_argument("--block-size", "-b", type=int, default=128,
                        help="Block size for quantization and scales (default: 128)")
    p_conv.add_argument("--no-resume", action="store_true",
                        help="Disable resuming from existing conversion_manifest.json")

    # 2. run
    p_run = subparsers.add_parser("run", help="Run streaming inference and autoregressive generation")
    p_run.add_argument("--model-dir", "-m", type=str, default="",
                       help="Path to directory containing .trit model files")
    p_run.add_argument("--prompt", type=str, default="What is Asymmetric Mixed-Precision?",
                       help="Text prompt for generation")
    p_run.add_argument("--max-new-tokens", type=int, default=15,
                       help="Maximum new tokens to generate (default: 15)")
    p_run.add_argument("--temperature", type=float, default=0.0,
                       help="Sampling temperature (0.0 for deterministic greedy decoding)")
    p_run.add_argument("--top-p", type=float, default=1.0,
                       help="Nucleus top-p probability threshold (default: 1.0)")
    p_run.add_argument("--top-k", type=int, default=50,
                       help="Top-k filtering threshold (default: 50)")

    # 2b. demo
    p_demo = subparsers.add_parser("demo", help="Run toy model demonstration (random toy model demo only)")
    p_demo.add_argument("--prompt", type=str, default="What is Asymmetric Mixed-Precision?",
                        help="Text prompt for generation")
    p_demo.add_argument("--max-new-tokens", type=int, default=15,
                        help="Maximum new tokens to generate (default: 15)")
    p_demo.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0.0 for deterministic greedy decoding)")
    p_demo.add_argument("--top-p", type=float, default=1.0,
                        help="Nucleus top-p probability threshold (default: 1.0)")
    p_demo.add_argument("--top-k", type=int, default=50,
                        help="Top-k filtering threshold (default: 50)")

    # 3. audit
    p_audit = subparsers.add_parser("audit", help="Audit model footprint, memory savings, and Spark node requirements")
    p_audit.add_argument("--model", type=str, default="qwen-2.4t",
                         help="Target model architecture (default: 'qwen-2.4t')")
    p_audit.add_argument("--spark-ram", type=float, default=512.0,
                         help="Spark accelerator node RAM in GB (default: 512.0)")
    p_audit.add_argument("--format", choices=["table", "json"], default="table",
                         help="Output format: 'table' (markdown) or 'json'")

    # 4. benchmark
    subparsers.add_parser("benchmark", help="Run empirical Frobenius and activation drift benchmarks")

    # 5. verify-shards
    p_ver = subparsers.add_parser("verify-shards", help="Verify integrity of master Safetensors shards on disk")
    p_ver.add_argument("--model-dir", "-m", type=str, default=str(DEFAULT_MODEL_DIR),
                       help=f"Path to directory containing .safetensors shards (default: {DEFAULT_MODEL_DIR})")
    p_ver.add_argument("--workers", "-w", type=int, default=4,
                       help="Worker threads for parallel scanning (default: 4)")

    # 6. serve
    p_srv = subparsers.add_parser("serve", help="Launch OpenAI-compatible local API server and Web Chat UI")
    p_srv.add_argument("--host", type=str, default="127.0.0.1",
                       help="Host to bind server (default: 127.0.0.1)")
    p_srv.add_argument("--port", "-p", type=int, default=8000,
                       help="Port to bind server (default: 8000)")
    p_srv.add_argument("--model-dir", "-m", type=str, default="",
                       help="Directory containing .trit container files")

    return parser


def handle_serve(args: argparse.Namespace) -> int:
    """Handles 'tritforge serve' subcommand."""
    from tritforge.server import run_server
    model_dir = Path(args.model_dir) if args.model_dir else None
    run_server(host=args.host, port=args.port, model_dir=model_dir)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    if args.command == "convert":
        return handle_convert(args)
    elif args.command == "run":
        return handle_run(args)
    elif args.command == "demo":
        return handle_demo(args)
    elif args.command == "audit":
        return handle_audit(args)
    elif args.command == "benchmark":
        return handle_benchmark(args)
    elif args.command == "verify-shards":
        return handle_verify_shards(args)
    elif args.command == "serve":
        return handle_serve(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
