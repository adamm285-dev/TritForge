"""Unit tests for TritForge Modular Plugins and dynaMoE routing."""

import pytest
import numpy as np
from tritforge.plugins import (
    PluginRegistry,
    DefaultRouterPlugin,
    DynaMoERouterPlugin,
    BaseRouterPlugin,
)


def test_plugin_registry():
    routers = PluginRegistry.list_routers()
    assert "default" in routers
    assert "dynamoe" in routers

    default_router = PluginRegistry.get_router("default")
    assert isinstance(default_router, DefaultRouterPlugin)

    dynamoe_router = PluginRegistry.get_router("dynamoe")
    assert isinstance(dynamoe_router, DynaMoERouterPlugin)

    with pytest.raises(ValueError):
        PluginRegistry.get_router("nonexistent_router")


def test_default_router_exact_k():
    router = PluginRegistry.get_router("default")
    logits = np.random.randn(512).astype(np.float32)

    indices, weights, telem = router.select_experts(logits, top_k=10)
    assert len(indices) == 10
    assert len(weights) == 10
    assert np.isclose(np.sum(weights), 1.0)
    assert telem["k_selected"] == 10
    assert not telem["is_dynamic"]


def test_dynamoe_router_high_confidence():
    router = DynaMoERouterPlugin(min_k=2, max_k=12, prob_mass_threshold=0.90)

    # Construct logits where top-2 capture 95% of probability mass
    logits = np.zeros(512, dtype=np.float32)
    logits[42] = 20.0
    logits[99] = 19.0

    indices, weights, telem = router.select_experts(logits, top_k=10)
    # Should dynamically collapse to 2 experts instead of 10
    assert len(indices) == 2
    assert telem["k_selected"] == 2
    assert 42 in indices
    assert 99 in indices
    assert np.isclose(np.sum(weights), 1.0)
    assert telem["is_dynamic"]


def test_dynamoe_router_flat_distribution():
    router = DynaMoERouterPlugin(min_k=2, max_k=12, prob_mass_threshold=0.90)

    # Completely flat logits across all 512 experts (uniform distribution)
    logits = np.ones(512, dtype=np.float32)

    indices, weights, telem = router.select_experts(logits, top_k=10)
    # High entropy should hit max_k clamp (12)
    assert len(indices) == 12
    assert telem["k_selected"] == 12
    assert np.isclose(np.sum(weights), 1.0)
