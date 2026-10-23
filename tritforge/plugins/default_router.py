"""TritForge Default Fixed-K Router Plugin.

Baseline reference implementation preserving pure pretraining routing behavior.
Conforms to SPEC-001 & SPEC-002 Ground Truth Baseline requirements.
Hardened against non-finite float anomalies (NaN, Inf).
"""

from typing import Dict, Any, Tuple
import numpy as np
from .base import BaseRouterPlugin, PluginRegistry


class DefaultRouterPlugin(BaseRouterPlugin):
    """Pristine baseline router: exact top-K softmax with zero heuristic modification."""

    @property
    def name(self) -> str:
        return "default"

    def select_experts(
        self,
        logits: np.ndarray,
        top_k: int = 10,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Select exactly the top-K highest logit experts using standard softmax."""
        logits = np.asarray(logits, dtype=np.float32)

        # Adversarial Hardening: Guard against NaN or non-finite inputs
        if not np.all(np.isfinite(logits)):
            logits = np.nan_to_num(logits, nan=0.0, posinf=30.0, neginf=-30.0)

        max_logit = np.max(logits)
        exp_logits = np.exp(logits - max_logit)
        sum_exp = np.sum(exp_logits)
        if sum_exp <= 0 or not np.isfinite(sum_exp):
            probs = np.full_like(logits, 1.0 / len(logits))
        else:
            probs = exp_logits / (sum_exp + 1e-12)

        # Standard top-K selection
        top_indices = np.argsort(probs)[-top_k:][::-1]
        top_probs = probs[top_indices]
        sum_top = np.sum(top_probs)
        if sum_top <= 0 or not np.isfinite(sum_top):
            norm_weights = np.full(top_k, 1.0 / top_k, dtype=np.float32)
        else:
            norm_weights = top_probs / (sum_top + 1e-12)

        # Basic telemetry
        entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
        top1_conf = float(top_probs[0]) if len(top_probs) > 0 else 0.0

        telemetry = {
            "router": self.name,
            "k_selected": int(top_k),
            "entropy": entropy,
            "top1_confidence": top1_conf,
            "is_dynamic": False,
        }
        return top_indices, norm_weights, telemetry


# Auto-register baseline router
PluginRegistry.register_router("default", DefaultRouterPlugin)
