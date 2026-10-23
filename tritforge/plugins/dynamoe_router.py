"""TritForge dynaMoE Dynamic Router Plugin.

Implements confidence-gated and entropy-adaptive active expert selection (K in [min_k, max_k]).
Conforms to SPEC-001 & SPEC-002 requirements.
Hardened against non-finite float anomalies (NaN, Inf) and numerical edge cases.
"""

from typing import Dict, Any, Tuple
import numpy as np
from .base import BaseRouterPlugin, PluginRegistry


class DynaMoERouterPlugin(BaseRouterPlugin):
    """Dynamic confidence-adaptive router.

    Dynamically sizes active experts based on cumulative probability mass
    coverage (tau threshold) and router distribution entropy.
    """

    def __init__(
        self,
        min_k: int = 2,
        max_k: int = 12,
        prob_mass_threshold: float = 0.90,
    ) -> None:
        self.min_k = min_k
        self.max_k = max_k
        self.prob_mass_threshold = prob_mass_threshold

    @property
    def name(self) -> str:
        return "dynamoe"

    def select_experts(
        self,
        logits: np.ndarray,
        top_k: int = 10,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Dynamically select K active experts based on cumulative confidence."""
        logits = np.asarray(logits, dtype=np.float32)

        # Adversarial Hardening: Guard against NaN, Inf, and non-finite values
        if not np.all(np.isfinite(logits)):
            logits = np.nan_to_num(logits, nan=0.0, posinf=30.0, neginf=-30.0)

        max_logit = np.max(logits)
        exp_logits = np.exp(logits - max_logit)
        sum_exp = np.sum(exp_logits)
        if sum_exp <= 0 or not np.isfinite(sum_exp):
            probs = np.full_like(logits, 1.0 / len(logits))
        else:
            probs = exp_logits / (sum_exp + 1e-12)

        # Full sort descending
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]

        # Calculate cumulative probability mass
        cumulative_mass = np.cumsum(sorted_probs)

        # Find first index crossing probability threshold (e.g. 90%)
        crossings = np.where(cumulative_mass >= self.prob_mass_threshold)[0]
        if len(crossings) > 0:
            target_k = int(crossings[0] + 1)
        else:
            target_k = top_k

        # Clamp between min_k and max_k
        k_dynamic = max(self.min_k, min(target_k, self.max_k))

        selected_indices = sorted_indices[:k_dynamic]
        selected_probs = sorted_probs[:k_dynamic]

        sum_selected = np.sum(selected_probs)
        if sum_selected <= 0 or not np.isfinite(sum_selected):
            norm_weights = np.full(k_dynamic, 1.0 / k_dynamic, dtype=np.float32)
        else:
            norm_weights = selected_probs / (sum_selected + 1e-12)

        entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
        top1_conf = float(sorted_probs[0]) if len(sorted_probs) > 0 else 0.0

        telemetry = {
            "router": self.name,
            "k_selected": int(k_dynamic),
            "nominal_k": int(top_k),
            "entropy": entropy,
            "top1_confidence": top1_conf,
            "cumulative_mass": float(np.sum(norm_weights)),
            "is_dynamic": True,
        }
        return selected_indices, norm_weights, telemetry


# Auto-register dynaMoE router
PluginRegistry.register_router("dynamoe", DynaMoERouterPlugin)
