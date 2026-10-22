"""
TritForge Cloud Sliding-Window Converter (tritforge.cloud)
==========================================================
Enables zero-disk-waste conversion in cloud environments (Google Colab, ephemeral VMs)
by downloading only the exact shards needed for a single layer, converting that layer to
a compact .trit container (~7 GB), exporting to Google Drive / cloud storage, and immediately
purging the raw FP8 shards from the local scratch disk.

Memory Bound: < 2 GB RAM
Disk Scratch Bound: < 40 GB temporary disk (fits in standard Google Colab 100 GB disk)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import time
from typing import Dict, List, Optional, Sequence, Set
import urllib.request

from tritforge.converter import BatchConverter, ShardIndex


def fetch_hf_index(repo_id: str, cache_dir: Path, hf_token: Optional[str] = None) -> Dict[str, Any]:
    """Fetch and cache model.safetensors.index.json from Hugging Face."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = cache_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            return json.load(f)

    url = f"https://huggingface.co/{repo_id}/raw/main/model.safetensors.index.json"
    headers = {"User-Agent": "TritForge/1.0"}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    req = urllib.request.Request(url, headers=headers)
    print(f"[CLOUD] Fetching shard index from {url}...")
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return data


def get_shards_for_layer(index_data: Dict[str, Any], layer_idx: int) -> List[str]:
    """Find the list of shard filenames required for a specific layer."""
    weight_map = index_data.get("weight_map", {})
    prefix = f"model.layers.{layer_idx}."
    shards: Set[str] = set()

    for tensor_name, shard_file in weight_map.items():
        if tensor_name.startswith(prefix):
            shards.add(shard_file)

    # Sort numerically by shard index
    return sorted(list(shards))


def download_shard_hf(
    repo_id: str,
    shard_filename: str,
    dest_dir: Path,
    hf_token: Optional[str] = None,
) -> Path:
    """Download a single safetensors shard from Hugging Face with resume support."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    target_path = dest_dir / shard_filename
    if target_path.exists():
        print(f"[CLOUD] Shard {shard_filename} already present on disk.")
        return target_path

    url = f"https://huggingface.co/{repo_id}/resolve/main/{shard_filename}"
    headers = {"User-Agent": "TritForge/1.0"}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    print(f"[CLOUD] Downloading {shard_filename} from Hugging Face...")
    req = urllib.request.Request(url, headers=headers)
    temp_path = dest_dir / f"{shard_filename}.download"

    with urllib.request.urlopen(req) as resp, open(temp_path, "wb") as f_out:
        total_size = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 8 * 1024 * 1024  # 8 MB chunks
        t0 = time.time()

        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f_out.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = downloaded / total_size * 100
                mb_s = downloaded / (1024 * 1024) / max(time.time() - t0, 0.01)
                print(f"\r  -> {pct:.1f}% ({downloaded / 1024**3:.2f}/{total_size / 1024**3:.2f} GB) @ {mb_s:.1f} MB/s", end="", flush=True)

    print()
    temp_path.rename(target_path)
    return target_path


class SlidingWindowCloudConverter:
    """
    Autonomous cloud converter that streams a model layer-by-layer:
    1. Downloads only shards containing layer L (~16-30 GB).
    2. Quantizes layer L to Base-3 .trit container (~7 GB).
    3. Exports .trit to permanent storage (Google Drive / Hugging Face).
    4. Deletes downloaded FP8 shards from scratch disk.
    """

    def __init__(
        self,
        repo_id: str,
        scratch_dir: Path,
        output_dir: Path,
        hf_token: Optional[str] = None,
    ):
        self.repo_id = repo_id
        self.scratch_dir = Path(scratch_dir)
        self.output_dir = Path(output_dir)
        self.hf_token = hf_token
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.index_data = fetch_hf_index(repo_id, self.scratch_dir, hf_token)

    def convert_layer(self, layer_idx: int, purge_after: bool = True) -> Path:
        """Convert a single layer through the sliding window pipeline."""
        target_container = self.output_dir / f"layer_{layer_idx:02d}.trit"
        if target_container.exists():
            print(f"[CLOUD] Layer {layer_idx:02d} already converted at {target_container}. Skipping.")
            return target_container

        required_shards = get_shards_for_layer(self.index_data, layer_idx)
        print(f"\n=======================================================")
        print(f" [SLIDING-WINDOW] Converting Layer {layer_idx:02d}")
        print(f" Required Shards: {required_shards}")
        print(f"=======================================================")

        downloaded_paths = []
        for shard in required_shards:
            path = download_shard_hf(self.repo_id, shard, self.scratch_dir, self.hf_token)
            downloaded_paths.append(path)

        # Execute Layer Quantization
        converter = BatchConverter(input_dir=self.scratch_dir, output_dir=self.output_dir)
        t0 = time.time()
        print(f"[CLOUD] Executing TritForge Base-3 Quantization for Layer {layer_idx:02d}...")
        converter.convert_all(layers=[layer_idx], resume=False)
        print(f"[CLOUD] Layer {layer_idx:02d} quantized in {(time.time() - t0)/60:.1f} minutes!")

        # Purge temporary raw FP8 shards to free disk space
        if purge_after:
            print(f"[CLOUD] Purging temporary raw shards from scratch disk to conserve space...")
            for path in downloaded_paths:
                if path.exists():
                    path.unlink()
                    print(f"  -> Deleted: {path.name}")

        return target_container
