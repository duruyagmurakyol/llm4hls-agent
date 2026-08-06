#!/usr/bin/env python3

"""Run an explicit, model-isolated LLM4HLS experiment matrix sequentially.

Unlike the discovery-oriented task-suite runner, this command executes only the
named task/model pairs in a schema-v2 suite.  Every pair receives an immutable
materialised task manifest, an isolated output directory and one CSV row.  State
is written atomically after every run so a long overnight experiment can be
resumed without repeating completed work.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent.onboarding_safe import onboard_benchmark  # noqa: E402
from agent.track_a import is_track_a_task, onboard_track_a_task  # noqa: E402

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]


SUMMARY_FIELDS = (
    "suite_run_id",
    "run_key",
    "priority",
    "tier",
    "task_id",
    "source_task_id",
    "task_path",
    "task_role",
    "task_subtype",
    "canonical_design",
    "execution_mode",
    "model_id",
    "model_slug",
    "provider",
    "started_at",
    "finished_at",
    "elapsed_seconds",
    "exit_code",
    "timed_out",
    "success",
    "status",
    "termination_reason",
    "selection_mode",
    "final_design_verified",
    "meets_submission_frequency",
    "initial_failure_stage",
    "initial_failure_class",
    "attempt_count",
    "selected_candidate_index",
    "selected_candidate_file",
    "estimated_frequency_mhz",
    "clock_period_ns",
    "latency_cycles",
    "latency_ns",
    "throughput_period_ns",
    "lut",
    "ff",
    "dsp",
    "bram",
    "model_calls",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "csim_calls",
    "synthesis_calls",
    "cosim_calls",
    "reference_harness_credits_spent",
    "reference_harness_credits_remaining",
    "reference_harness_score_estimate",
    "manifest_path",
    "result_path",
    "log_path",
    "error",
)


class MatrixError(RuntimeError):
    """Raised for invalid or unsafe matrix execution."""


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    slug: str
    provider: str


@dataclass(frozen=True)
class TaskSpec:
    priority: int
    tier: str
    task_id: str
    path: str
    mode: str
    role: str
    subtype: str
    canonical_design: str
    timeout_seconds: int


@dataclass(frozen=True)
class RunSpec:
    run_key: str
    priority: int
    tier: str
    task_id: str
    source_task_id: str
    task_path: str
    mode: str
    role: str
    subtype: str
    canonical_design: str
    model_id: str
    model_slug: str
    provider: str
    timeout_seconds: int
    manifest_path: str
    output_dir: str

    @property
    def resolved_manifest(self) -> Path:
        return _resolve(self.manifest_path)

    @property
    def resolved_output(self) -> Path:
        return _resolve(self.output_dir)


@dataclass(frozen=True)
class PlannedRun:
    task: TaskSpec
    model: ModelSpec

    @property
    def run_key(self) -> str:
        return f"{self.task.priority:02d}_{_safe_id(self.task.task_id)}__{self.model.slug}"


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _safe_id(value: str) -> str:
    rendered = "".join(character if character.isalnum() else "_" for character in value)
    rendered = "_".join(part for part in rendered.split("_") if part)
    return rendered.lower() or "run"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in SUMMARY_FIELDS})
    temporary.replace(path)


def _git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def _command_output(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=60,
    )
    return completed.stdout.strip()


def _load_suite(path: Path) -> dict[str, Any]:
    suite = _load_json(path)
    if suite.get("schema_version") != 2:
        raise MatrixError(f"Expected schema_version 2 in {path}")
    if not isinstance(suite.get("suite_id"), str) or not suite["suite_id"]:
        raise MatrixError("Matrix suite requires a non-empty suite_id")
    if not isinstance(suite.get("models"), list) or not suite["models"]:
        raise MatrixError("Matrix suite requires a non-empty models list")
    if not isinstance(suite.get("tasks"), list) or not suite["tasks"]:
        raise MatrixError("Matrix suite requires a non-empty tasks list")
    if suite.get("order") != "task_then_model":
        raise MatrixError("Only task_then_model ordering is supported")
    return suite


def _models(suite: dict[str, Any]) -> list[ModelSpec]:
    result: list[ModelSpec] = []
    slugs: set[str] = set()
    for raw in suite["models"]:
        if not isinstance(raw, dict):
            raise MatrixError("Every model entry must be an object")
        model_id = str(raw.get("id", "")).strip()
        slug = str(raw.get("slug", "")).strip()
        provider = str(raw.get("provider", "")).strip().casefold()
        if not model_id or not slug or not provider:
            raise MatrixError("Every model requires id, slug and provider")
        if slug in slugs:
            raise MatrixError(f"Duplicate model slug: {slug}")
        if provider not in {"siliconflow", "openrouter"}:
            raise MatrixError(f"Unsupported provider for {model_id}: {provider}")
        slugs.add(slug)
        result.append(ModelSpec(model_id=model_id, slug=slug, provider=provider))
    return result


def _tasks(suite: dict[str, Any]) -> list[TaskSpec]:
    defaults = dict(suite.get("defaults") or {})
    timeout = int(defaults.get("timeout_seconds", 5400))
    result: list[TaskSpec] = []
    identities: set[str] = set()
    priorities: set[int] = set()
    for raw in suite["tasks"]:
        if not isinstance(raw, dict):
            raise MatrixError("Every task entry must be an object")
        priority = int(raw["priority"])
        task_id = str(raw.get("id", "")).strip()
        mode = str(raw.get("mode", "")).strip().casefold()
        if not task_id or not str(raw.get("path", "")).strip():
            raise MatrixError("Every task requires id and path")
        if mode not in {"repair", "optimise"}:
            raise MatrixError(f"Task {task_id} has unsupported mode {mode!r}")
        if task_id in identities:
            raise MatrixError(f"Duplicate task id: {task_id}")
        if priority in priorities:
            raise MatrixError(f"Duplicate task priority: {priority}")
        identities.add(task_id)
        priorities.add(priority)
        result.append(
            TaskSpec(
                priority=priority,
                tier=str(raw.get("tier", "extended")),
                task_id=task_id,
                path=str(raw["path"]),
                mode=mode,
                role=str(raw.get("role", mode)),
                subtype=str(raw.get("subtype", "unspecified")),
                canonical_design=str(raw.get("canonical_design", task_id)),
                timeout_seconds=int(raw.get("timeout_seconds", timeout)),
            )
        )
    return sorted(result, key=lambda item: item.priority)


def expand_matrix(suite: dict[str, Any], *, core_only: bool = False) -> list[PlannedRun]:
    models = _models(suite)
    tasks = [task for task in _tasks(suite) if not core_only or task.tier == "core"]
    planned = [PlannedRun(task=task, model=model) for task in tasks for model in models]
    keys = [item.run_key for item in planned]
    if len(keys) != len(set(keys)):
        raise MatrixError("Expanded matrix contains duplicate run keys")
    expected = suite.get("expected_runs")
    if not core_only and isinstance(expected, int) and len(planned) != expected:
        raise MatrixError(
            f"Matrix expands to {len(planned)} runs but expected_runs is {expected}"
        )
    return planned


def _track_a_id(path: Path) -> str | None:
    if tomllib is None:
        return None
    task_file = path / "task.toml"
    if not task_file.is_file():
        return None
    try:
        with task_file.open("rb") as handle:
            value = tomllib.load(handle).get("task_id")
    except (OSError, ValueError):
        return None
    return str(value) if isinstance(value, str) and value else None


def _resolve_task_path(task: TaskSpec) -> Path:
    configured = _resolve(task.path)
    if configured.is_dir():
        return configured

    parent = configured.parent
    if parent.is_dir():
        for child in sorted(parent.iterdir()):
            if child.is_dir() and _track_a_id(child) == task.task_id:
                return child

    raise MatrixError(
        f"Task path does not exist and no matching task.toml was found: {task.path}"
    )


def _onboard_task(task: TaskSpec) -> tuple[str, dict[str, Any], Path]:
    path = _resolve_task_path(task)
    manifest = onboard_track_a_task(path) if is_track_a_task(path) else onboard_benchmark(path)
    return manifest.task_id, copy.deepcopy(manifest.data), path


def materialise_runs(
    suite: dict[str, Any],
    *,
    run_id: str,
    run_dir: Path,
    core_only: bool,
    maximum: int | None,
) -> list[RunSpec]:
    planned = expand_matrix(suite, core_only=core_only)
    if maximum is not None:
        planned = planned[:maximum]

    base_tasks: dict[str, tuple[str, dict[str, Any], Path]] = {}
    for item in planned:
        if item.task.task_id not in base_tasks:
            base_tasks[item.task.task_id] = _onboard_task(item.task)

    manifests_dir = run_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    runs: list[RunSpec] = []
    for item in planned:
        source_task_id, data, resolved_task_path = base_tasks[item.task.task_id]
        data = copy.deepcopy(data)
        unique_task_id = f"{source_task_id}__{item.model.slug}"
        output_dir = (
            REPO_ROOT
            / "experiments"
            / "model_comparison"
            / run_id
            / item.model.slug
            / _safe_id(item.task.task_id)
        )
        data["task_id"] = unique_task_id
        data["output_dir"] = _display(output_dir)
        model = dict(data.get("model") or {})
        model["name"] = item.model.model_id
        model["provider"] = item.model.provider
        data["model"] = model
        data["matrix"] = {
            "suite_run_id": run_id,
            "run_key": item.run_key,
            "source_task_id": source_task_id,
            "matrix_task_id": item.task.task_id,
            "priority": item.task.priority,
            "tier": item.task.tier,
            "role": item.task.role,
            "subtype": item.task.subtype,
            "canonical_design": item.task.canonical_design,
            "execution_mode": item.task.mode,
            "model_id": item.model.model_id,
            "model_slug": item.model.slug,
            "provider": item.model.provider,
        }
        manifest_path = manifests_dir / f"{item.run_key}.json"
        manifest_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        runs.append(
            RunSpec(
                run_key=item.run_key,
                priority=item.task.priority,
                tier=item.task.tier,
                task_id=item.task.task_id,
                source_task_id=source_task_id,
                task_path=_display(resolved_task_path),
                mode=item.task.mode,
                role=item.task.role,
                subtype=item.task.subtype,
                canonical_design=item.task.canonical_design,
                model_id=item.model.model_id,
                model_slug=item.model.slug,
                provider=item.model.provider,
                timeout_seconds=item.task.timeout_seconds,
                manifest_path=_display(manifest_path),
                output_dir=_display(output_dir),
            )
        )
    _write_json_atomic(
        run_dir / "matrix_manifest.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "runs": [asdict(item) for item in runs],
        },
    )
    return runs


def _load_materialised_runs(run_dir: Path) -> list[RunSpec]:
    value = _load_json(run_dir / "matrix_manifest.json")
    runs = value.get("runs")
    if not isinstance(runs, list):
        raise MatrixError(f"No matrix_manifest.json found under {run_dir}")
    return [RunSpec(**item) for item in runs if isinstance(item, dict)]


def _print_planned(planned: list[PlannedRun]) -> None:
    print(f"Planned {len(planned)} run(s):")
    for index, item in enumerate(planned, 1):
        task = item.task
        print(
            f"{index:02d}. {item.run_key}\n"
            f"    tier={task.tier} mode={task.mode} role={task.role}/{task.subtype}\n"
            f"    design={task.canonical_design} model={item.model.model_id}\n"
            f"    path={task.path}"
        )


def _required_key(provider: str) -> str:
    return "OPENROUTER_API_KEY" if provider == "openrouter" else "SILICONFLOW_API_KEY"


def _active_hls_processes() -> list[str]:
    completed = subprocess.run(
        ["pgrep", "-af", "scripts/run_agent.py|vitis-run|vitis_hls|xsim"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if completed.returncode not in {0, 1}:
        return []
    current_pid = str(os.getpid())
    return [
        line
        for line in completed.stdout.splitlines()
        if line.strip()
        and not line.lstrip().startswith(current_pid + " ")
        and "pgrep -af" not in line
    ]


def _check_prerequisites(runs: list[RunSpec]) -> None:
    if shutil.which("vitis-run") is None:
        raise MatrixError("vitis-run is unavailable; source Vitis 2025.2 settings first")
    missing = sorted(
        {
            _required_key(run.provider)
            for run in runs
            if not os.environ.get(_required_key(run.provider))
        }
    )
    if missing:
        raise MatrixError("Missing API key environment variable(s): " + ", ".join(missing))
    active = _active_hls_processes()
    if active:
        raise MatrixError(
            "Another agent/Vitis process is active; do not overlap matrix runs:\n"
            + "\n".join(f"  {line}" for line in active)
        )


def _kill_process_group(process: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _run_command(
    command: list[str],
    *,
    log_path: Path,
    timeout_seconds: int,
    environment: dict[str, str],
) -> tuple[int, bool, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    timed_out = False
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("Command: " + " ".join(command) + "\n\n")
        handle.flush()
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
            start_new_session=True,
        )
        while process.poll() is None:
            if time.monotonic() - started > timeout_seconds:
                timed_out = True
                _kill_process_group(process)
                break
            time.sleep(5)
        return_code = process.poll()
        if return_code is None:
            return_code = 124
    return int(return_code), timed_out, time.monotonic() - started


def _clean_vitis_cache() -> None:
    cache = Path("/tmp/llm4hls-agent")
    if cache.exists():
        shutil.rmtree(cache)


def _first_failure(result: dict[str, Any]) -> tuple[Any, Any]:
    trajectory = result.get("trajectory")
    if not isinstance(trajectory, list):
        return None, None
    for event in trajectory:
        if not isinstance(event, dict) or event.get("status") != "failed":
            continue
        stage = event.get("stage")
        if isinstance(stage, str) and stage.startswith("initial_"):
            details = event.get("details")
            details = details if isinstance(details, dict) else {}
            return stage.removeprefix("initial_"), details.get("failure_class")
    return None, None


def _selected_metrics(output: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    state = _load_json(output / "candidate_state.json")
    selected = state.get("selected_design")
    selected = selected if isinstance(selected, dict) else {}
    metrics = selected.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        baseline = _load_json(output / "verified_baseline.json")
        metrics = baseline.get("metrics")
    return selected, metrics if isinstance(metrics, dict) else {}


def _latency_cycles(metrics: dict[str, Any]) -> Any:
    for key in (
        "latency_worst_cycles",
        "latency_average_cycles",
        "latency_best_cycles",
        "latency_cycles",
    ):
        if metrics.get(key) is not None:
            return metrics[key]
    return None


def _derived_latency_ns(metrics: dict[str, Any]) -> Any:
    if metrics.get("latency_ns") is not None:
        return metrics["latency_ns"]
    cycles = _latency_cycles(metrics)
    period = metrics.get("clock_period_ns")
    if isinstance(cycles, (int, float)) and isinstance(period, (int, float)):
        return float(cycles) * float(period)
    return None


def _result_row(
    run: RunSpec,
    *,
    suite_run_id: str,
    started_at: str,
    finished_at: str,
    elapsed_seconds: float,
    exit_code: int,
    timed_out: bool,
    log_path: Path,
    error: str | None,
) -> dict[str, Any]:
    output = run.resolved_output
    result_path = output / "unified_agent_result.json"
    result = _load_json(result_path)
    budget = _load_json(output / "budget_summary.json")
    consumed = budget.get("consumed")
    consumed = consumed if isinstance(consumed, dict) else {}
    track_a = budget.get("track_a")
    track_a = track_a if isinstance(track_a, dict) else {}
    selected, metrics = _selected_metrics(output)
    failure_stage, failure_class = _first_failure(result)
    state = _load_json(output / "candidate_state.json")
    policy = state.get("selection_policy")
    policy = policy if isinstance(policy, dict) else {}
    attempts = consumed.get("model_calls")
    if attempts is None:
        stage_dir = output / "stage_aware"
        attempts = len(list(stage_dir.glob("attempt_*"))) if stage_dir.is_dir() else None

    return {
        "suite_run_id": suite_run_id,
        "run_key": run.run_key,
        "priority": run.priority,
        "tier": run.tier,
        "task_id": run.task_id,
        "source_task_id": run.source_task_id,
        "task_path": run.task_path,
        "task_role": run.role,
        "task_subtype": run.subtype,
        "canonical_design": run.canonical_design,
        "execution_mode": run.mode,
        "model_id": run.model_id,
        "model_slug": run.model_slug,
        "provider": run.provider,
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "success": result.get("success"),
        "status": result.get("status"),
        "termination_reason": result.get("termination_reason"),
        "selection_mode": result.get("selection_mode") or policy.get("mode"),
        "final_design_verified": result.get("final_design_verified")
        if result.get("final_design_verified") is not None
        else state.get("selected_design_fully_verified"),
        "meets_submission_frequency": result.get("meets_submission_frequency"),
        "initial_failure_stage": failure_stage,
        "initial_failure_class": failure_class,
        "attempt_count": attempts,
        "selected_candidate_index": selected.get("candidate_index"),
        "selected_candidate_file": selected.get("candidate_file")
        or selected.get("archived_file"),
        "estimated_frequency_mhz": metrics.get("frequency_mhz")
        or metrics.get("estimated_frequency_mhz"),
        "clock_period_ns": metrics.get("clock_period_ns"),
        "latency_cycles": _latency_cycles(metrics),
        "latency_ns": _derived_latency_ns(metrics),
        "throughput_period_ns": metrics.get("throughput_period_ns"),
        "lut": metrics.get("resources_lut_used"),
        "ff": metrics.get("resources_ff_used"),
        "dsp": metrics.get("resources_dsp_used"),
        "bram": metrics.get("resources_bram_used"),
        "model_calls": consumed.get("model_calls"),
        "input_tokens": consumed.get("input_tokens"),
        "output_tokens": consumed.get("output_tokens"),
        "total_tokens": consumed.get("total_tokens"),
        "csim_calls": consumed.get("csim_calls"),
        "synthesis_calls": consumed.get("synthesis_calls"),
        "cosim_calls": consumed.get("cosim_calls"),
        "reference_harness_credits_spent": result.get(
            "reference_harness_credits_spent"
        )
        or track_a.get("credits_spent"),
        "reference_harness_credits_remaining": result.get(
            "reference_harness_credits_remaining"
        )
        or track_a.get("credits_remaining"),
        "reference_harness_score_estimate": result.get(
            "reference_harness_score_estimate"
        ),
        "manifest_path": run.manifest_path,
        "result_path": _display(result_path) if result_path.is_file() else None,
        "log_path": _display(log_path),
        "error": error,
    }


def _environment_snapshot(runs: list[RunSpec]) -> dict[str, Any]:
    return {
        "captured_at": _utc_now(),
        "repository_commit": _git_output("rev-parse", "HEAD"),
        "repository_status": _git_output("status", "--short"),
        "python": sys.version,
        "vitis_run": shutil.which("vitis-run"),
        "vitis_version": _command_output(["vitis-run", "--version"]),
        "models": sorted(
            {
                (run.model_id, run.model_slug, run.provider)
                for run in runs
            }
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an explicit task-by-model LLM4HLS experiment matrix."
    )
    parser.add_argument(
        "--suite",
        type=Path,
        default=REPO_ROOT / "configs" / "suites" / "overnight_60.json",
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--core-only", action="store_true")
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--run-id")
    parser.add_argument("--resume-suite", type=Path)
    parser.add_argument("--rerun-failed", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--no-cleanup-vitis-cache", action="store_true")
    args = parser.parse_args()

    if args.resume_suite and (args.core_only or args.max_runs or args.run_id):
        raise MatrixError(
            "--resume-suite cannot be combined with --core-only, --max-runs or --run-id"
        )

    if args.resume_suite:
        run_dir = args.resume_suite.expanduser().resolve()
        state = _load_json(run_dir / "suite_state.json")
        if not state:
            raise MatrixError(f"No suite_state.json found under {run_dir}")
        run_id = str(state["run_id"])
        suite = _load_suite(run_dir / "suite_definition.json")
        runs = _load_materialised_runs(run_dir)
        rows = [row for row in state.get("results", []) if isinstance(row, dict)]
    else:
        suite_path = args.suite.expanduser().resolve()
        suite = _load_suite(suite_path)
        planned = expand_matrix(suite, core_only=args.core_only)
        if args.max_runs is not None:
            planned = planned[: args.max_runs]
        if args.list:
            _print_planned(planned)
            return

        commit = _git_output("rev-parse", "HEAD")
        run_id = args.run_id or (
            f"{suite['suite_id']}_{datetime.now().strftime('%Y%m%dT%H%M%S')}_{commit[:7]}"
        )
        run_dir = REPO_ROOT / "results" / "suites" / run_id
        if run_dir.exists():
            raise MatrixError(f"Suite run directory already exists: {run_dir}")
        run_dir.mkdir(parents=True)
        shutil.copy2(suite_path, run_dir / "suite_definition.json")
        runs = materialise_runs(
            suite,
            run_id=run_id,
            run_dir=run_dir,
            core_only=args.core_only,
            maximum=args.max_runs,
        )
        rows = []
        state = {
            "schema_version": 1,
            "run_id": run_id,
            "suite_id": suite["suite_id"],
            "repository_commit": commit,
            "started_at": _utc_now(),
            "finished_at": None,
            "status": "running",
            "total_runs": len(runs),
            "results": rows,
        }

    if args.list:
        print(f"Materialised {len(runs)} run(s) under {_display(run_dir)}")
        for index, run in enumerate(runs, 1):
            print(
                f"{index:02d}. {run.run_key} mode={run.mode} model={run.model_id}\n"
                f"    output={run.output_dir}"
            )
        return

    _check_prerequisites(runs)
    cleanup_cache = not args.no_cleanup_vitis_cache and bool(
        suite.get("defaults", {}).get("cleanup_vitis_cache_after_run", True)
    )
    if cleanup_cache:
        _clean_vitis_cache()

    _write_json_atomic(run_dir / "environment.json", _environment_snapshot(runs))
    _write_json_atomic(run_dir / "suite_state.json", state)
    _write_csv(run_dir / "suite_summary.csv", rows)

    completed = {
        str(row.get("run_key")): row
        for row in rows
        if row.get("run_key")
    }
    print(f"Suite run: {run_id}")
    print(f"Runs: {len(runs)}")
    print(f"Results: {_display(run_dir)}")

    interrupted = False
    try:
        for position, run in enumerate(runs, 1):
            previous = completed.get(run.run_key)
            if previous and not (
                args.rerun_failed and previous.get("success") is not True
            ):
                print(f"[{position}/{len(runs)}] skip completed {run.run_key}")
                continue

            print(f"\n[{position}/{len(runs)}] {run.run_key}")
            print(f"  task: {run.task_id} ({run.role}/{run.subtype})")
            print(f"  model: {run.model_id}")
            print(f"  mode: {run.mode}")
            print(f"  output: {run.output_dir}")

            command = [
                sys.executable,
                "-u",
                str(REPO_ROOT / "scripts" / "run_agent.py"),
                str(run.resolved_manifest),
                "--mode",
                run.mode,
            ]
            environment = dict(os.environ)
            environment["PYTHONUNBUFFERED"] = "1"
            environment["LLM4HLS_PROVIDER"] = run.provider
            environment["LLM4HLS_SUITE_RUN_ID"] = run_id
            environment["LLM4HLS_SUITE_RUN_KEY"] = run.run_key
            log_path = run_dir / "logs" / f"{run.run_key}.log"
            started_at = _utc_now()
            exit_code, timed_out, elapsed = _run_command(
                command,
                log_path=log_path,
                timeout_seconds=run.timeout_seconds,
                environment=environment,
            )
            finished_at = _utc_now()
            error = "run timeout" if timed_out else None
            row = _result_row(
                run,
                suite_run_id=run_id,
                started_at=started_at,
                finished_at=finished_at,
                elapsed_seconds=elapsed,
                exit_code=exit_code,
                timed_out=timed_out,
                log_path=log_path,
                error=error,
            )

            if previous is not None:
                rows = [item for item in rows if item.get("run_key") != run.run_key]
            rows.append(row)
            completed[run.run_key] = row
            state["results"] = rows
            state["completed_runs"] = len(rows)
            _write_json_atomic(run_dir / "suite_state.json", state)
            _write_csv(run_dir / "suite_summary.csv", rows)
            print(
                "  result: "
                f"exit={exit_code} success={row.get('success')} "
                f"status={row.get('status')} verified={row.get('final_design_verified')}"
            )

            if cleanup_cache:
                _clean_vitis_cache()

            failed = exit_code != 0 or row.get("success") is not True
            if failed and args.stop_on_error:
                break
    except KeyboardInterrupt:
        interrupted = True
        print("\nMatrix interrupted; completed state has been preserved.", file=sys.stderr)
    finally:
        state["finished_at"] = _utc_now()
        state["status"] = "interrupted" if interrupted else "completed"
        state["results"] = rows
        state["completed_runs"] = len(rows)
        _write_json_atomic(run_dir / "suite_state.json", state)
        _write_csv(run_dir / "suite_summary.csv", rows)

    successful = sum(row.get("success") is True for row in rows)
    failed = sum(row.get("success") is not True for row in rows)
    print("\nMatrix summary")
    print("==============")
    print(f"Completed rows: {len(rows)} / {len(runs)}")
    print(f"Successful: {successful}")
    print(f"Failed/incomplete: {failed}")
    print(f"CSV: {_display(run_dir / 'suite_summary.csv')}")
    print(f"State: {_display(run_dir / 'suite_state.json')}")


if __name__ == "__main__":
    try:
        main()
    except MatrixError as error:
        print(f"Matrix configuration error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
