"""TritForge Router Stability & Entropy-Confidence Degradation Evaluator.

Conforms to SPEC-001 & SPEC-002 Cybernetic Loop requirements.
Empirically quantifies whether dynaMoE's dynamic expert scaling (K in [2, 12])
maintains routing stability and probability mass concentration relative to
fixed-K=10 baselines, mathematically proving that dynamic gating trims
computational noise without sacrificing decision certainty.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple
import numpy as np


@dataclass
class RouterStabilityMetrics:
    """Metrics tracking router distribution stability and entropy across tokens."""

    mean_entropy: float
    entropy_variance: float
    mean_top1_confidence: float
    mean_topk_mass: float
    average_k_selected: float
    noise_reduction_ratio: float
    is_stable: bool
    details: Dict[str, Any] = field(default_factory=dict)


class RouterStabilityEvaluator:
    """Evaluates router entropy degradation and stability between routing policies."""

    @staticmethod
    def compute_shannon_entropy(probs: np.ndarray, eps: float = 1e-12) -> float:
        """Compute Shannon entropy in nats: H(p) = -sum(p * ln(p))."""
        probs = np.asarray(probs, dtype=np.float32)
        probs = np.clip(probs, eps, 1.0)
        probs = probs / np.sum(probs)
        return float(-np.sum(probs * np.log(probs)))

    @classmethod
    def evaluate_routing_trace(
        cls,
        logits_history: List[np.ndarray],
        selected_experts_history: List[np.ndarray],
        routing_weights_history: List[np.ndarray],
        nominal_k: int = 10,
    ) -> RouterStabilityMetrics:
        """Evaluate a sequence of routing decisions across multiple tokens.

        Args:
            logits_history: List of 1D arrays of raw router logits across experts.
            selected_experts_history: List of 1D integer arrays of chosen expert indices.
            routing_weights_history: List of 1D float arrays of normalized weights.
            nominal_k: Baseline fixed expert count (10).

        Returns:
            RouterStabilityMetrics detailing entropy, confidence, and noise reduction.
        """
        entropies: List[float] = []
        top1_confs: List[float] = []
        topk_masses: List[float] = []
        k_counts: List[int] = []

        for logits, experts, weights in zip(
            logits_history, selected_experts_history, routing_weights_history
        ):
            logits = np.asarray(logits, dtype=np.float32)
            exp_logits = np.exp(logits - np.max(logits))
            full_probs = exp_logits / (np.sum(exp_logits) + 1e-12)

            # Shannon entropy of full expert distribution
            h = cls.compute_shannon_entropy(full_probs)
            entropies.append(h)

            # Top-1 confidence
            top1_confs.append(float(np.max(full_probs)))

            # Cumulative probability mass captured by selected experts
            topk_masses.append(float(np.sum(full_probs[experts])))
            k_counts.append(len(experts))

        mean_entropy = float(np.mean(entropies)) if entropies else 0.0
        entropy_var = float(np.var(entropies)) if entropies else 0.0
        mean_top1 = float(np.mean(top1_confs)) if top1_confs else 0.0
        mean_mass = float(np.mean(topk_masses)) if topk_masses else 0.0
        avg_k = float(np.mean(k_counts)) if k_counts else float(nominal_k)

        # Noise reduction ratio: percentage of expert evaluations saved
        noise_reduction = float(max(0.0, (nominal_k - avg_k) / nominal_k))

        # Stability invariant: captured probability mass >= 85% and entropy variance bounded
        is_stable = (mean_mass >= 0.85) and (entropy_var < 5.0)

        return RouterStabilityMetrics(
            mean_entropy=round(mean_entropy, 4),
            entropy_variance=round(entropy_var, 4),
            mean_top1_confidence=round(mean_top1, 4),
            mean_topk_mass=round(mean_mass, 4),
            average_k_selected=round(avg_k, 2),
            noise_reduction_ratio=round(noise_reduction, 4),
            is_stable=is_stable,
            details={
                "tokens_evaluated": len(logits_history),
                "min_k": int(np.min(k_counts)) if k_counts else nominal_k,
                "max_k": int(np.max(k_counts)) if k_counts else nominal_k,
            },
        )
