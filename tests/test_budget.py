from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.budget import BudgetExceeded, BudgetState


@pytest.fixture
def budget() -> BudgetState:
    return BudgetState.from_manifest(
        {
            "max_iterations": 3,
            "max_model_calls": 3,
            "max_csim_calls": 4,
            "max_cosim_calls": 1,
            "max_synthesis_calls": 3,
            "max_total_tokens": 2000,
        }
    )


def test_budget_records_shared_consumption(budget: BudgetState) -> None:
    budget.charge_iteration(stage="repair")
    budget.charge_model_call(stage="repair_generation")
    budget.record_model_tokens(
        input_tokens=556,
        output_tokens=77,
        stage="repair_generation",
    )
    budget.charge_csim(stage="initial_csim", success=False)
    budget.charge_csim(stage="repair_validation", success=True)
    budget.charge_synthesis(stage="baseline_synthesis", success=True)

    summary = budget.summary()
    assert summary["consumed"] == {
        "iterations": 1,
        "model_calls": 1,
        "csim_calls": 2,
        "cosim_calls": 0,
        "synthesis_calls": 1,
        "input_tokens": 556,
        "output_tokens": 77,
        "total_tokens": 633,
    }
    assert summary["remaining"]["csim_calls"] == 2
    assert summary["remaining"]["total_tokens"] == 1367


def test_failed_and_timed_out_calls_still_consume_budget(
    budget: BudgetState,
) -> None:
    budget.charge_synthesis(
        stage="candidate_001_synthesis",
        success=False,
        timed_out=True,
    )

    summary = budget.summary()
    assert summary["consumed"]["synthesis_calls"] == 1
    assert summary["events"][-1]["success"] is False
    assert summary["events"][-1]["timed_out"] is True


def test_budget_exhaustion_blocks_operation() -> None:
    budget = BudgetState(
        max_iterations=1,
        max_model_calls=1,
        max_csim_calls=0,
        max_cosim_calls=0,
        max_synthesis_calls=0,
    )

    with pytest.raises(BudgetExceeded, match="csim_calls"):
        budget.charge_csim(stage="initial_csim")

    assert budget.stop_reason == "csim_calls_budget_exhausted"


def test_candidate_generation_preserves_validation_headroom(
    budget: BudgetState,
) -> None:
    budget.charge_csim(stage="initial_csim")
    budget.charge_synthesis(stage="baseline_synthesis")

    assert budget.can_generate_candidate(
        reserve_csim=2,
        reserve_synthesis=2,
    )

    budget.charge_csim(stage="candidate_001_csim")
    budget.charge_synthesis(stage="candidate_001_synthesis")

    assert not budget.can_generate_candidate(
        reserve_csim=2,
        reserve_synthesis=2,
    )


def test_token_limit_is_enforced_after_provider_usage() -> None:
    budget = BudgetState(
        max_iterations=1,
        max_model_calls=1,
        max_csim_calls=0,
        max_cosim_calls=0,
        max_synthesis_calls=0,
        max_total_tokens=100,
    )
    budget.charge_model_call(stage="generation")

    with pytest.raises(BudgetExceeded, match="exceeded"):
        budget.record_model_tokens(
            input_tokens=80,
            output_tokens=30,
            stage="generation",
        )

    assert budget.total_tokens_used == 110
    assert budget.total_tokens_remaining == 0
    assert budget.stop_reason == "token_budget_exhausted"


def test_budget_summary_is_written(
    tmp_path: Path,
    budget: BudgetState,
) -> None:
    budget.set_stop_reason("repair_completed")
    path = budget.write_summary(tmp_path / "budget_summary.json")

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["initial"]["max_model_calls"] == 3
    assert data["stop_reason"] == "repair_completed"
