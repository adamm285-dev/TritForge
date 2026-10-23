"""Unit tests for the TritForge Automated Multi-Round Ablation Matrix."""

import pytest
from tritforge.ablation import (
    AblationMatrixOrchestrator,
    AblationRoundConfig,
    RoundMetricResult,
)


def test_ablation_matrix_rounds_count():
    orchestrator = AblationMatrixOrchestrator()
    assert len(orchestrator.ROUNDS) == 10
    round_ids = [r.round_id for r in orchestrator.ROUNDS]
    assert round_ids == list(range(10))


def test_ablation_round_execution():
    orchestrator = AblationMatrixOrchestrator()
    prompts = [
        "Write a Python function to compute Fibonacci numbers.",
        "Explain the difference between a process and a thread.",
    ]

    # Test Round 0 (Baseline)
    config_round0 = orchestrator.ROUNDS[0]
    result0 = orchestrator.execute_round(config_round0, prompts=prompts, max_tokens=16)

    assert isinstance(result0, RoundMetricResult)
    assert result0.round_id == 0
    assert result0.average_k_experts == 10.0
    assert result0.tokens_per_second > 0.0

    # Test Round 1 (dynaMoE)
    config_round1 = orchestrator.ROUNDS[1]
    result1 = orchestrator.execute_round(config_round1, prompts=prompts, max_tokens=16)

    assert result1.round_id == 1
    assert result1.average_k_experts < 10.0  # Dynamic drop in K
    assert result1.tokens_per_second > result0.tokens_per_second  # Faster throughput


def test_ablation_full_matrix_and_markdown():
    orchestrator = AblationMatrixOrchestrator()
    prompts = ["Test prompt 1", "Test prompt 2"]

    results = orchestrator.execute_all_rounds(prompts=prompts, max_tokens=8)
    assert len(results) == 10

    # Verify compound stack (Round 9) is the fastest
    assert results[9].tokens_per_second > results[0].tokens_per_second

    # Verify Markdown generation
    md_table = orchestrator.render_markdown_table(results)
    assert "| Round | Tier | Configuration |" in md_table
    assert "Round 0" in md_table
    assert "Round 9" in md_table
