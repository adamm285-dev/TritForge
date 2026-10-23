"""TritForge Modular Plugin Subsystem - Base Definitions & Registry.

Conforms to SPEC-001 & SPEC-002 cybernetic control loops.
Provides strict abstract interfaces for swappable MoE routing policies.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Type
import numpy as np


class BaseRouterPlugin(ABC):
    """Abstract base class for all TritForge MoE routing policies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for the router plugin."""
        pass

    @abstractmethod
    def select_experts(
        self,
        logits: np.ndarray,
        top_k: int = 10,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Select active experts given router logits.

        Args:
            logits: 1D array of router logits across all experts (e.g. shape (512,)).
            top_k: Nominal/baseline expert count.
            **kwargs: Optional runtime telemetry, context tokens, or thresholds.

        Returns:
            Tuple of:
              - selected_indices: 1D integer ndarray of chosen expert IDs.
              - routing_weights: 1D float ndarray of normalized softmax weights.
              - telemetry: Dict containing routing diagnostics (entropy, k, confidence).
        """
        pass


class PluginRegistry:
    """Thread-safe registry for managing swappable router and kernel plugins."""

    _routers: Dict[str, Type[BaseRouterPlugin]] = {}

    @classmethod
    def register_router(cls, name: str, router_cls: Type[BaseRouterPlugin]) -> None:
        """Register a router plugin by unique name."""
        cls._routers[name.lower()] = router_cls

    @classmethod
    def get_router(cls, name: str, **kwargs: Any) -> BaseRouterPlugin:
        """Instantiate a registered router plugin."""
        key = name.lower()
        if key not in cls._routers:
            available = list(cls._routers.keys())
            raise ValueError(
                f"Unknown router plugin '{name}'. Registered plugins: {available}"
            )
        return cls._routers[key](**kwargs)

    @classmethod
    def list_routers(cls) -> list[str]:
        """Return list of all registered router plugin names."""
        return sorted(cls._routers.keys())
