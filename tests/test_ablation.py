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


def test_ablation_round_execution_raises_not_implemented():
    """I-3: Ablation execution must raise NotImplementedError until live runtime exists."""
    orchestrator = AblationMatrixOrchestrator()
    prompts = [
        "Write a Python function to compute Fibonacci numbers.",
        "Explain the difference between a process and a thread.",
    ]

    config_round0 = orchestrator.ROUNDS[0]
    with pytest.raises(NotImplementedError) as exc_info:
        orchestrator.execute_round(config_round0, prompts=prompts, max_tokens=16)
    assert "Ablation rounds are not implemented" in str(exc_info.value)


def test_ablation_execute_all_rounds_raises_not_implemented():
    """I-3: Ablation execute_all_rounds must raise NotImplementedError."""
    orchestrator = AblationMatrixOrchestrator()
    prompts = ["Test prompt 1", "Test prompt 2"]

    with pytest.raises(NotImplementedError) as exc_info:
        orchestrator.execute_all_rounds(prompts=prompts, max_tokens=8)
    assert "Ablation rounds are not implemented" in str(exc_info.value)
