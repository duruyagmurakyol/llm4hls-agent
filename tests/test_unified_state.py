from agent.state import AgentResult, BudgetState, TrajectoryEvent


def test_budget_state_from_manifest() -> None:
    budget = BudgetState.from_manifest(
        {
            "max_iterations": 3,
            "max_csim_calls": 4,
            "max_cosim_calls": 0,
            "max_synthesis_calls": 2,
            "max_model_calls": 3,
        }
    )
    assert budget.max_iterations == 3
    assert budget.synthesis_calls_used == 0


def test_agent_result_serialises_trajectory() -> None:
    result = AgentResult(
        task_id="task",
        success=True,
        status="completed",
        termination_reason="done",
        output_dir="results/task",
        trajectory=[TrajectoryEvent(step=1, stage="repair", status="passed")],
    )
    data = result.to_dict()
    assert data["trajectory"][0]["stage"] == "repair"
    assert data["success"] is True
