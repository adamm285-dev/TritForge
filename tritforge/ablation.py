"""TritForge Automated Multi-Round Ablation Test Runner.

Conforms to SPEC-001 & SPEC-002 Cybernetic Loop requirements.
Orchestrates the 10-round orthogonal ablation matrix:
  - Round 0: Pure Baseline (Fixed K=10, standard 5-trits, standard KV, raw prompt)
  - Round 1: dynaMoE Only (Dynamic routing K in [2, 12])
  - Round 2: TritSlack Only (In-Band tags 243-255 zero-skipping/Hessian escape)
  - Round 3: SDCS Framework Only (Disambiguated / structured prompts)
  - Round 4: CacheForge Gateway Only (Paged KV memory gateway)
  - Round 5: dynaMoE + TritSlack
  - Round 6: dynaMoE + SDCS Framework
  - Round 7: dynaMoE + CacheForge Gateway
  - Round 8: TritSlack + CacheForge Gateway
  - Round 9: Full TritForge Sovereign Stack (Compound Matrix)
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import numpy as np


@dataclass
class AblationRoundConfig:
    """Configuration specification for an isolated ablation round."""

    round_id: int
    name: str
    tier: str
    router_mode: str = "default"  # 'default' or 'dynamoe'
    tritslack_enabled: bool = False
    sdcs_framework_enabled: bool = False
    cacheforge_gateway_enabled: bool = False
    description: str = ""


@dataclass
class RoundMetricResult:
    """Empirical measurements produced by executing a single ablation round."""

    round_id: int
    name: str
    tokens_per_second: float
    time_to_first_token_ms: float
    total_tokens_generated: int
    ram_usage_gb: float
    average_k_experts: float
    disambiguation_passed: bool
    output_fidelity_score: float
    details: Dict[str, Any] = field(default_factory=dict)


class AblationMatrixOrchestrator:
    """Manages the definition, sequential execution, and logging of the 10-round test matrix."""

    ROUNDS: List[AblationRoundConfig] = [
        AblationRoundConfig(
            round_id=0,
            name="Round 0: Pure Baseline",
            tier="Tier 1: Ground Truth",
            router_mode="default",
            tritslack_enabled=False,
            sdcs_framework_enabled=False,
            cacheforge_gateway_enabled=False,
            description="Fixed K=10, standard packing, raw prompt, standard KV.",
        ),
        AblationRoundConfig(
            round_id=1,
            name="Round 1: dynaMoE Only",
            tier="Tier 2: Single-Variable",
            router_mode="dynamoe",
            tritslack_enabled=False,
            sdcs_framework_enabled=False,
            cacheforge_gateway_enabled=False,
            description="Dynamic routing K in [2, 12], standard packing, raw prompt.",
        ),
        AblationRoundConfig(
            round_id=2,
            name="Round 2: TritSlack Only",
            tier="Tier 2: Single-Variable",
            router_mode="default",
            tritslack_enabled=True,
            sdcs_framework_enabled=False,
            cacheforge_gateway_enabled=False,
            description="Fixed K=10, tags 243-255 zero-skipping enabled.",
        ),
        AblationRoundConfig(
            round_id=3,
            name="Round 3: SDCS Framework Only",
            tier="Tier 2: Single-Variable",
            router_mode="default",
            tritslack_enabled=False,
            sdcs_framework_enabled=True,
            cacheforge_gateway_enabled=False,
            description="Fixed K=10, structured input & disambiguation oracle.",
        ),
        AblationRoundConfig(
            round_id=4,
            name="Round 4: CacheForge Gateway Only",
            tier="Tier 2: Single-Variable",
            router_mode="default",
            tritslack_enabled=False,
            sdcs_framework_enabled=False,
            cacheforge_gateway_enabled=True,
            description="Fixed K=10, CacheForge KV paging gateway.",
        ),
        AblationRoundConfig(
            round_id=5,
            name="Round 5: dynaMoE + TritSlack",
            tier="Tier 3: Pairwise Synergy",
            router_mode="dynamoe",
            tritslack_enabled=True,
            sdcs_framework_enabled=False,
            cacheforge_gateway_enabled=False,
            description="Dynamic routing + in-band zero-skipping kernel.",
        ),
        AblationRoundConfig(
            round_id=6,
            name="Round 6: dynaMoE + SDCS Framework",
            tier="Tier 3: Pairwise Synergy",
            router_mode="dynamoe",
            tritslack_enabled=False,
            sdcs_framework_enabled=True,
            cacheforge_gateway_enabled=False,
            description="Dynamic routing on disambiguated prompts.",
        ),
        AblationRoundConfig(
            round_id=7,
            name="Round 7: dynaMoE + CacheForge Gateway",
            tier="Tier 3: Pairwise Synergy",
            router_mode="dynamoe",
            tritslack_enabled=False,
            sdcs_framework_enabled=False,
            cacheforge_gateway_enabled=True,
            description="Dynamic compute paired with dynamic KV cache.",
        ),
        AblationRoundConfig(
            round_id=8,
            name="Round 8: TritSlack + CacheForge Gateway",
            tier="Tier 3: Pairwise Synergy",
            router_mode="default",
            tritslack_enabled=True,
            sdcs_framework_enabled=False,
            cacheforge_gateway_enabled=True,
            description="Weight memory bus optimization + KV cache optimization.",
        ),
        AblationRoundConfig(
            round_id=9,
            name="Round 9: Full Sovereign Stack",
            tier="Tier 4: Compound Stack",
            router_mode="dynamoe",
            tritslack_enabled=True,
            sdcs_framework_enabled=True,
            cacheforge_gateway_enabled=True,
            description="All 4 systems working together for maximum performance.",
        ),
    ]

    def __init__(self, model_engine: Optional[Any] = None) -> None:
        self.model_engine = model_engine

    def execute_round(
        self,
        config: AblationRoundConfig,
        prompts: List[str],
        max_tokens: int = 32,
    ) -> RoundMetricResult:
        """Execute a single ablation round under strict configuration isolation."""
        raise NotImplementedError(
            "Ablation rounds are not implemented until live Qwen3.5-MoE runtime and subcomponents exist (I-3)"
        )

    def execute_all_rounds(
        self,
        prompts: List[str],
        max_tokens: int = 32,
    ) -> List[RoundMetricResult]:
        """Execute all 10 rounds sequentially in clean isolation."""
        raise NotImplementedError(
            "Ablation rounds are not implemented until live Qwen3.5-MoE runtime and subcomponents exist (I-3)"
        )

    def render_markdown_table(self, results: List[RoundMetricResult]) -> str:
        """Render results into standard GFM Markdown table for evals.md."""
        lines = [
            "| Round | Tier | Configuration | Avg K | Tok/s | TTFT (ms) | RAM (GB) | Fidelity |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in results:
            lines.append(
                f"| **{r.name.split(':')[0]}** | {r.details.get('tier', '')} | {r.name.split(': ')[1]} | {r.average_k_experts} | **{r.tokens_per_second:.1f}** | {r.time_to_first_token_ms:.1f} | {r.ram_usage_gb:.1f} | {r.output_fidelity_score:.3f} |"
            )
        return "\n".join(lines)
