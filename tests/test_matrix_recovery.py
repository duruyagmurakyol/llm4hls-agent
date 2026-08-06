from __future__ import annotations

from pathlib import Path

from agent.budget import BudgetState
from agent.config import load_task, validate_task
from agent.onboarding_safe import resolve_benchmark
from agent.stage_aware import supports_stage_aware_task

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_discovered_benchmark_manifest_supports_optimisation_mode() -> None:
    task = resolve_benchmark(REPO_ROOT / "benchmarks" / "hls_eval" / "atax")

    validate_task(task.data)
    assert task.data["adapter"]["kind"] == "auto"
    assert task.data["optimisation"]["selection"]["mode"] == "research_pareto"
    assert task.data["optimisation"]["validation"] == {}


def test_discovered_non_cosim_benchmark_can_start_optimisation_candidate() -> None:
    task = resolve_benchmark(REPO_ROOT / "benchmarks" / "hls_eval" / "atax")
    budget = BudgetState.from_manifest(task.data["budgets"])

    assert task.data["validation_policy"]["requires_cosim"] is False
    assert budget.requires_cosim is False

    budget.charge_csim(stage="initial_csim", success=True)
    budget.charge_synthesis(stage="initial_synthesis", success=True)

    assert budget.can_generate_candidate(
        reserve_csim=1,
        reserve_synthesis=1,
        reserve_cosim=1,
    ) is True


def test_materialised_stage_aware_manifest_remains_stage_aware(tmp_path: Path) -> None:
    source_task = REPO_ROOT / "benchmarks" / "capability_suite" / "vector_add_generate"
    from agent.track_a import onboard_track_a_task

    task = onboard_track_a_task(source_task)
    data = dict(task.data)
    data["task_id"] = "vector_add_generate__matrix_model"
    data["output_dir"] = str(tmp_path / "output")
    manifest = tmp_path / "materialised.json"
    import json

    manifest.write_text(json.dumps(data), encoding="utf-8")
    loaded = load_task(manifest)

    assert loaded.data["task_kind"] == "generate"
    assert supports_stage_aware_task(loaded) is True


def test_public_cli_loads_json_manifests_before_dispatch() -> None:
    source = (REPO_ROOT / "scripts" / "run_agent.py").read_text(encoding="utf-8")

    assert "from agent.config import TaskManifest, load_task" in source
    assert "task_input = load_task(target)" in source
