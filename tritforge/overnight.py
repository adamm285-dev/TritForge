"""
TritForge Overnight Conversion Daemon (tritforge.overnight)
===========================================================
Autonomous batch conversion daemon that monitors downloaded shards
and converts completed layers into high-density .trit binary containers.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Optional, Set, Union

from tritforge.container import read_trit_header
from tritforge.converter import BatchConverter, ShardIndex, check_git_dirty, file_sha256


def preflight() -> None:
    """Pre-flight dirty-tree check; exits non-zero if tritforge/ has uncommitted changes."""
    if check_git_dirty():
        print(
            "[ERROR] Refusing to run overnight daemon with uncommitted changes in tritforge/ (W2-15).",
            file=sys.stderr,
        )
        sys.exit(1)


def verified_done_layers(output_dir: Union[str, Path]) -> Set[int]:
    """
    Evaluates existing .trit containers in output_dir against conversion_manifest.json.
    Side-effect free: does NOT delete or unlink any files.

    A layer index counts as done ONLY if all of the following hold:
      1. It is listed in manifest['completed_layers'];
      2. It is NOT listed in manifest['failed_layers'];
      3. Its physical file size equals manifest metrics 'output_bytes';
      4. Its SHA-256 equals manifest metrics 'sha256';
      5. Its header parses successfully.
    """
    out_path = Path(output_dir)
    manifest_path = out_path / "conversion_manifest.json"
    manifest_completed: Set[int] = set()
    manifest_failed: Set[int] = set()
    manifest_metrics: dict = {}

    if manifest_path.exists():
        try:
            m_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_completed = set(m_data.get("completed_layers", []))
            manifest_failed = set(int(k) for k in m_data.get("failed_layers", {}).keys())
            manifest_metrics = m_data.get("metrics", {})
        except Exception:
            manifest_completed = set()
            manifest_failed = set()
            manifest_metrics = {}

    done_layers: Set[int] = set()

    for p in sorted(list(out_path.glob("layer_*.trit"))):
        num_str = p.stem.replace("layer_", "")
        if not num_str.isdigit():
            continue
        l_num = int(num_str)

        # 1. Listed in completed_layers and 2. Not listed in failed_layers
        if (l_num not in manifest_completed) or (l_num in manifest_failed):
            continue

        m_entry = manifest_metrics.get(str(l_num))
        if not m_entry:
            continue

        # 3. File size equals manifest output_bytes
        expected_size = m_entry.get("output_bytes")
        if expected_size is None or p.stat().st_size != expected_size:
            continue

        # 4. SHA-256 equals manifest sha256 (if recorded)
        expected_sha = m_entry.get("sha256")
        if expected_sha and file_sha256(p) != expected_sha:
            continue

        # 5. Header parses successfully
        try:
            read_trit_header(p)
            done_layers.add(l_num)
        except Exception:
            continue

    return done_layers


def run(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    total_layers: int = 92,
    max_loops: Optional[int] = None,
) -> None:
    """
    Main conversion loop for the overnight daemon.
    Calls preflight() before anything else.
    """
    preflight()

    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    converter = BatchConverter(
        input_dir=in_path,
        output_dir=out_path,
        gatekeeper_dtype="float32",
        scale_method="optimal",
        allow_dirty=True,
    )

    loop_count = 0
    while True:
        loop_count += 1
        existing = verified_done_layers(out_path)
        print(f"\n[DAEMON] Converted layers locked on disk: {len(existing)}/{total_layers}")
        if len(existing) >= total_layers:
            print(f"[SUCCESS] All {total_layers} layers are converted and locked on disk!")
            break

        if max_loops is not None and loop_count >= max_loops:
            break

        # Re-index incoming download shards
        try:
            idx = ShardIndex(in_path)
        except Exception as e:
            print(f"[WARN] Error indexing shards: {e}. Retrying in 30s...")
            time.sleep(30)
            continue

        converted_any = False
        for l_idx in range(total_layers):
            if l_idx in existing:
                continue

            exp_ids = idx.get_layer_expert_indices(l_idx)
            if len(exp_ids) == 512:
                print(f"\n>> [START] Layer {l_idx:02d} (512/512 experts ready). Converting now...", flush=True)
                t0 = time.time()
                try:
                    summary = converter.convert_all(layers=[l_idx], resume=True)
                    elapsed = time.time() - t0
                    print(f">> [DONE] Layer {l_idx:02d} converted in {elapsed/60:.1f} mins!", flush=True)
                    converted_any = True
                except Exception as ex:
                    print(f">> [ERROR] Failed to convert layer {l_idx}: {ex}", flush=True)

        if not converted_any:
            print("[DAEMON] Waiting for next batch of shards to download... (sleeping 60s)")
            time.sleep(60)
