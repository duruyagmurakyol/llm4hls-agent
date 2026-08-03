#!/usr/bin/env python3

"""Unified entry layer for autonomous repair and PPA optimisation tasks."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from agent.config import TaskManifest, load_task
from agent.optimise.runner import run_optimisation
from agent.repair.runner import run_repair
from agent.state import AgentResult, TrajectoryEvent
from agent.tools.command_runner import CommandResult, run_command

REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def _output_dir(task: TaskManifest) -> Path:
    path = _resolve(task.output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_resolved_task(task: TaskManifest) -> Path:
    snapshot = {
        "task_id": task.task_id,
        "task_root": task.data.get("task_root"),
        "artifacts": task.data["artifacts"],
        "interface": task.data["interface"],
        "target": task.data["target"],
        "model": task.data["model"],
        "budgets": task.data["budgets"],
        "output_dir": str(task.output_dir),
    }
    path = _output_dir(task) / "resolved_task.json"
    path.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    return path


def _write_result(result: AgentResult) -> Path:
    output_dir = _resolve(result.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "unified_agent_result.json"
    path.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def _run_autonomous_ppa(
    task: TaskManifest,
    *,
    status_only: bool,
    max_steps: int | None,
    config_path: Path | None = None,
) -> AgentResult:
    if config_path is None:
        config_path = _resolve(task.data["adapter"]["config"])
    optimisation = run_optimisation(
        config_path,
        status_only=status_only,
        max_steps=max_steps,
    )
    trajectory = [
        TrajectoryEvent(
            step=index,
            stage=str(item.get("stage", "optimisation")),
            status="passed" if item.get("passed", True) else "failed",
            details=item,
        )
        for index, item in enumerate(optimisation.trajectory, 1)
    ]
    return AgentResult(
        task_id=task.task_id,
        success=optimisation.success,
        status=optimisation.status,
        termination_reason=optimisation.termination_reason,
        output_dir=str(task.output_dir),
        trajectory=trajectory,
    )


def _repair_config(task: TaskManifest) -> dict[str, object]:
    repair = task.data["repair"]
    model = task.data["model"]
    return {
        "repair_mode": "direct_api",
        "experiment_id": task.task_id,
        "benchmark_source": repair["benchmark_source"],
        "editable_files": repair["editable_files"],
        "protected_files": repair["protected_files"],
        "context_files": repair.get("context_files", repair["protected_files"]),
        "host_validation": repair["host_validation"],
        "independent_validation": repair["independent_validation"],
        "model": model["name"],
        "temperature": model.get("temperature", 0.0),
        "max_output_tokens": model.get("max_tokens", 2048),
        "api_timeout_seconds": model.get("timeout_seconds", 120),
        "thinking_budget": model.get("thinking_budget"),
    }


def _run_direct_api_repair(task: TaskManifest) -> AgentResult:
    print("\n=== Unified repair workflow ===", flush=True)
    passed, run_dir, repair_result = run_repair(
        _repair_config(task),
        keep_workspace=True,
    )
    print(f"Experiment: {repair_result['experiment_id']}")
    print(f"Model: {repair_result['model']}")
    print(f"Failure class: {repair_result['failure_class']}")
    print(
        "Tokens: "
        f"{repair_result['tokens_used']} "
        f"(input={repair_result['input_tokens']}, output={repair_result['output_tokens']})"
    )
    print(f"Modified files: {', '.join(repair_result['modified_files']) if repair_result['modified_files'] else 'none'}")
    print(f"Post-repair host test passed: {repair_result['post_host_validation_passed']}")
    print(f"Independent validation passed: {repair_result['independent_validation_passed']}")
    print(f"Results: {run_dir.relative_to(REPO_ROOT)}")

    return AgentResult(
        task_id=task.task_id,
        success=passed,
        status="correctness_established" if passed else "repair_failed",
        termination_reason="repair_completed" if passed else "repair_failed",
        output_dir=str(task.output_dir),
        trajectory=[
            TrajectoryEvent(
                step=1,
                stage="repair",
                status="passed" if passed else "failed",
                details={
                    "run_dir": str(run_dir.relative_to(REPO_ROOT)),
                    "failure_class": repair_result["failure_class"],
                    "tokens_used": repair_result["tokens_used"],
                    "modified_files": repair_result["modified_files"],
                    "post_host_validation_passed": repair_result["post_host_validation_passed"],
                    "independent_validation_passed": repair_result["independent_validation_passed"],
                },
            )
        ],
    )


def _initial_csim(task: TaskManifest) -> CommandResult:
    task_root = _resolve(task.data["task_root"])
    build_file = _resolve(task.data["artifacts"]["build_files"][0])

    with TemporaryDirectory(prefix=f"{task.task_id}_initial_csim_") as temp_dir:
        if build_file.suffix.lower() == ".cfg":
            command = [
                "vitis-run",
                "--mode",
                "hls",
                "--csim",
                "--config",
                str(build_file.relative_to(task_root)),
                "--work_dir",
                str(Path(temp_dir) / "vitis_work"),
            ]
        else:
            command = ["vitis-run", "--mode", "hls", "--tcl", str(build_file)]
        result = run_command(command, cwd=task_root)

    (_output_dir(task) / "initial_csim.log").write_text(result.output, encoding="utf-8")
    return result


def _ppa_config(task: TaskManifest) -> dict[str, object]:
    budgets = task.data["budgets"]
    source = task.data["artifacts"]["source"]
    build_file = task.data["artifacts"]["build_files"][0]
    return {
        "experiment_name": f"{task.task_id}_ppa",
        "benchmark": Path(task.data["task_root"]).name,
        "top_function": task.data["interface"]["top_function"],
        "baseline": {
            "source": source,
            "tcl": build_file,
            "project_dir": f"/tmp/llm4hls-agent/{task.task_id}_baseline",
        },
        "validation": {
            "constant_loop_tail_bounds": True,
            "preserve_diagnosed_loop_label": True,
        },
        "prompt_constraints": [
            "Preserve the top-level function signature and all testbench-observed semantics.",
            "Do not modify the supplied testbench or baseline source in place.",
        ],
        "output_dir": str(task.output_dir),
        "model": task.data["model"],
        "budget": {
            "max_candidates": budgets["max_iterations"],
            "max_synthesis_calls": budgets["max_synthesis_calls"],
        },
    }


def _prepend_initial_csim(result: AgentResult, csim: CommandResult) -> AgentResult:
    for index, event in enumerate(result.trajectory, 2):
        event.step = index
    result.trajectory.insert(
        0,
        TrajectoryEvent(
            step=1,
            stage="initial_csim",
            status="passed" if csim.passed else "failed",
            details={
                "command": list(csim.command),
                "return_code": csim.return_code,
                "log_path": str((_output_dir_from_result(result) / "initial_csim.log").relative_to(REPO_ROOT)),
            },
        ),
    )
    return result


def _output_dir_from_result(result: AgentResult) -> Path:
    path = _resolve(result.output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _run_auto(
    task: TaskManifest,
    *,
    status_only: bool,
    max_steps: int | None,
) -> AgentResult:
    if status_only:
        raise ValueError("status-only is not supported for automatically discovered tasks")

    print("\n=== Initial CSim ===", flush=True)
    csim = _initial_csim(task)
    if not csim.passed:
        print("Initial CSim failed; entering repair.", flush=True)
        return _prepend_initial_csim(_run_direct_api_repair(task), csim)

    print("Initial CSim passed; entering PPA optimisation.", flush=True)
    with TemporaryDirectory(prefix=f"{task.task_id}_ppa_") as temp_dir:
        config_path = Path(temp_dir) / "optimisation.json"
        config_path.write_text(json.dumps(_ppa_config(task), indent=2) + "\n", encoding="utf-8")
        result = _run_autonomous_ppa(
            task,
            status_only=False,
            max_steps=max_steps,
            config_path=config_path,
        )
    return _prepend_initial_csim(result, csim)


def run_agent(
    task_input: Path | TaskManifest,
    *,
    status_only: bool = False,
    max_steps: int | None = None,
) -> AgentResult:
    task = task_input if isinstance(task_input, TaskManifest) else load_task(task_input)
    resolved_path = _write_resolved_task(task)
    print(f"Resolved task: {resolved_path.relative_to(REPO_ROOT)}")

    if task.adapter_kind in {"autonomous_ppa", "legacy_ppa"}:
        result = _run_autonomous_ppa(task, status_only=status_only, max_steps=max_steps)
    elif task.adapter_kind == "direct_api_repair":
        if status_only:
            raise ValueError("status-only is not supported by direct_api_repair tasks")
        result = _run_direct_api_repair(task)
    elif task.adapter_kind == "auto":
        result = _run_auto(task, status_only=status_only, max_steps=max_steps)
    else:
        raise ValueError(f"Unsupported adapter kind: {task.adapter_kind}")
    result_path = _write_result(result)
    print(f"\nUnified result: {result_path.relative_to(REPO_ROOT)}")
    return result
