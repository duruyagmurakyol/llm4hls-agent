from __future__ import annotations

from pathlib import Path

from agent.budget import BudgetState
from agent.config import load_task


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_PATH = REPO_ROOT / "configs" / "tasks" / "vector_add_track_a.json"


def test_vector_add_budget_supports_baseline_plus_five_candidates() -> None:
    manifest = load_task(TASK_PATH)
    budget = BudgetState.from_manifest(manifest.data["budgets"])

    assert budget.requires_cosim is False
    assert budget.max_iterations == 5
    assert budget.max_model_calls == 5
    assert budget.max_csim_calls == 6
    assert budget.max_synthesis_calls == 6
    assert budget.max_cosim_calls == 0

    budget.charge_csim(stage="baseline_csim", success=True)
    budget.charge_synthesis(stage="baseline_synthesis", success=True)

    for candidate_index in range(1, 6):
        assert budget.can_generate_candidate(
            reserve_csim=1,
            reserve_synthesis=1,
            reserve_cosim=1,
        ), f"candidate {candidate_index} should fit within the configured budget"

        budget.charge_iteration(stage=f"candidate_{candidate_index:03d}_iteration")
        budget.charge_model_call(stage=f"candidate_{candidate_index:03d}_generation")
        budget.charge_csim(
            stage=f"candidate_{candidate_index:03d}_csim",
            success=True,
        )
        budget.charge_synthesis(
            stage=f"candidate_{candidate_index:03d}_synthesis",
            success=True,
        )

    assert budget.remaining("iterations") == 0
    assert budget.remaining("model_calls") == 0
    assert budget.remaining("csim_calls") == 0
    assert budget.remaining("synthesis_calls") == 0
    assert budget.remaining("cosim_calls") == 0
