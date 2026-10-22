"""
Tests for TritForge Cloud Sliding-Window Converter (tritforge.cloud)
"""

import json
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch

from tritforge.cloud import fetch_hf_index, get_shards_for_layer, SlidingWindowCloudConverter


def test_get_shards_for_layer():
    mock_index = {
        "weight_map": {
            "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00213.safetensors",
            "model.layers.0.mlp.experts.0.gate_proj.weight": "model-00002-of-00213.safetensors",
            "model.layers.0.mlp.experts.511.down_proj.weight": "model-00003-of-00213.safetensors",
            "model.layers.1.self_attn.q_proj.weight": "model-00004-of-00213.safetensors",
            "model.layers.1.mlp.experts.0.gate_proj.weight": "model-00004-of-00213.safetensors",
        }
    }

    shards_l0 = get_shards_for_layer(mock_index, 0)
    assert shards_l0 == [
        "model-00001-of-00213.safetensors",
        "model-00002-of-00213.safetensors",
        "model-00003-of-00213.safetensors",
    ]

    shards_l1 = get_shards_for_layer(mock_index, 1)
    assert shards_l1 == ["model-00004-of-00213.safetensors"]

    shards_l2 = get_shards_for_layer(mock_index, 2)
    assert shards_l2 == []


def test_fetch_hf_index_cached(tmp_path):
    cached_index = tmp_path / "model.safetensors.index.json"
    dummy_data = {"weight_map": {"tensorA": "shard1.safetensors"}}
    with open(cached_index, "w", encoding="utf-8") as f:
        json.dump(dummy_data, f)

    # Should read from cache without network request
    data = fetch_hf_index("dummy/repo", tmp_path)
    assert data == dummy_data
