from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from agent.config import TaskManifest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_experiment_matrix.py"
SUITE = REPO_ROOT / "configs" / "suites" / "overnight_60.json"


def _module():
    spec = importlib.util.spec_from_file_location("run_experiment_matrix", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_overnight_matrix_expands_to_exactly_sixty_task_first_runs() -> None:
    module = _module()
    suite = module._load_suite(SUITE)
    planned = module.expand_matrix(suite)

    assert len(planned) == 60
    assert len({item.run_key for item in planned}) == 60

    first = planned[:3]
    assert {item.task.task_id for item in first} == {"vector_add_generate"}
    assert [item.model.slug for item in first] == [
        "deepseek_v4_pro",
        "qwen35_122b_awq4",
        "qwen36_27b_fp8",
    ]

    assert all(item.task.tier == "core" for item in planned[:36])
    assert all(item.task.tier == "extended" for item in planned[36:])
    assert planned[35].task.task_id == "multi_fault_feedback"
    assert planned[36].task.task_id == "interface_wrong_top_name"


def test_core_only_matrix_contains_twelve_tasks_and_thirty_six_runs() -> None:
    module = _module()
    suite = module._load_suite(SUITE)
    planned = module.expand_matrix(suite, core_only=True)

    assert len(planned) == 36
    assert len({item.task.task_id for item in planned}) == 12
    assert all(item.task.tier == "core" for item in planned)


def test_only_unique_golden_designs_use_optimisation_mode() -> None:
    module = _module()
    suite = module._load_suite(SUITE)
    tasks = module._tasks(suite)

    optimisation = [task for task in tasks if task.mode == "optimise"]
    assert [task.task_id for task in optimisation] == [
        "dotProduct_optimize",
        "atax",
        "bicg",
        "gemm",
        "vector_add",
        "stream_pipeline",
    ]
    assert len({task.canonical_design for task in optimisation}) == 6
    assert all(task.mode == "repair" for task in tasks if task not in optimisation)


def test_materialisation_isolates_model_outputs_and_preserves_source_task(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    suite = {
        "schema_version": 2,
        "suite_id": "test_matrix",
        "expected_runs": 2,
        "order": "task_then_model",
        "defaults": {"timeout_seconds": 60},
        "models": [
            {"id": "model/A", "slug": "model_a", "provider": "siliconflow"},
            {"id": "model/B", "slug": "model_b", "provider": "siliconflow"},
        ],
        "tasks": [
            {
                "priority": 1,
                "tier": "core",
                "id": "repair_case",
                "path": "benchmarks/repair_case",
                "mode": "repair",
                "role": "repair",
                "subtype": "functional",
                "canonical_design": "kernel",
            }
        ],
    }
    base_data = {
        "task_id": "source_repair_case",
        "task_kind": "repair",
        "output_dir": "experiments/original",
        "model": {"name": "old-model", "provider": "siliconflow"},
    }

    monkeypatch.setattr(
        module,
        "_onboard_task",
        lambda task: (
            "source_repair_case",
            dict(base_data),
            REPO_ROOT / task.path,
        ),
    )

    run_dir = tmp_path / "results" / "test_run"
    run_dir.mkdir(parents=True)
    runs = module.materialise_runs(
        suite,
        run_id="test_run",
        run_dir=run_dir,
        core_only=False,
        maximum=None,
    )

    assert len(runs) == 2
    assert runs[0].output_dir != runs[1].output_dir
    assert runs[0].source_task_id == runs[1].source_task_id == "source_repair_case"

    manifests = [json.loads(run.resolved_manifest.read_text()) for run in runs]
    assert [manifest["model"]["name"] for manifest in manifests] == [
        "model/A",
        "model/B",
    ]
    assert manifests[0]["task_id"] == "source_repair_case__model_a"
    assert manifests[1]["task_id"] == "source_repair_case__model_b"
    assert manifests[0]["matrix"]["execution_mode"] == "repair"
    assert manifests[0]["output_dir"] != manifests[1]["output_dir"]
