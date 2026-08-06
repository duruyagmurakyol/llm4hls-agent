#!/usr/bin/env python3

"""Run a provenance-aware library of LLM4HLS tasks sequentially.

The suite runner intentionally executes one task at a time because Vitis HLS,
co-simulation and model budgets are easier to audit when runs do not share tool
processes or mutable output directories. State and CSV summaries are updated
after every task so an interrupted overnight run can be resumed safely.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
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
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent.onboarding_safe import resolve_benchmark  # noqa: E402

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.11+ is required in CI
    tomllib = None  # type: ignore[assignment]


SUMMARY_FIELDS = (
    "suite_run_id",
    "task_id",
    "task_path",
    "collection",
    "source",
    "tags",
    "started_at",
    "finished_at",
    "elapsed_seconds",
    "exit_code",
    "timed_out",
    "preflight",
    "success",
    "status",
    "termination_reason",
    "selection_mode",
    "final_design_verified",
    "meets_submission_frequency",
    "selected_candidate_index",
    "selected_candidate_file",
    "estimated_frequency_mhz",
    "latency_ns",
    "throughput_period_ns",
    "lut",
    "ff",
    "dsp",
    "bram",
    "total_tokens",
    "csim_calls",
    "synthesis_calls",
    "cosim_calls",
    "reference_harness_credits_spent",
    "reference_harness_credits_remaining",
    "reference_harness_score_estimate",
    "result_path",
    "log_path",
    "error",
)


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    path: str
    output_dir: str
    collection: str
    source: str
    tags: tuple[str, ...]
    timeout_seconds: int
    max_agent_steps: int | None
    resume: bool
    preflight: bool

    @property
    def resolved_path(self) -> Path:
        value = Path(self.path).expanduser()
        return value if value.is_absolute() else REPO_ROOT / value

    @property
    def resolved_output_dir(self) -> Path:
        value = Path(self.output_dir).expanduser()
        return value if value.is_absolute() else REPO_ROOT / value


class SuiteError(RuntimeError):
    """Raised for invalid suite definitions or unsafe execution conditions."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_id(value: str) -> str:
    rendered = "".join(character if character.isalnum() else "_" for character in value)
    rendered = "_".join(part for part in rendered.split("_") if part)
    return rendered.lower() or "task"


def _display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


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


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def _load_suite(path: Path) -> dict[str, Any]:
    suite = _load_json(path)
    if suite.get("schema_version") != 1:
        raise SuiteError(f"Unsupported suite schema in {path}")
    if not isinstance(suite.get("suite_id"), str) or not suite["suite_id"]:
        raise SuiteError("Suite requires a non-empty suite_id")
    if not isinstance(suite.get("collections"), list):
        raise SuiteError("Suite requires a collections list")
    return suite


def _track_a_task_id(path: Path) -> str:
    if tomllib is None:
        raise SuiteError("Python tomllib is required to read Track-A task packages")
    task_file = path / "task.toml"
    with task_file.open("rb") as handle:
        data = tomllib.load(handle)
    task_id = data.get("task_id")
    return str(task_id) if isinstance(task_id, str) and task_id else path.name


def _excluded(path: Path, patterns: Iterable[str]) -> bool:
    relative = _display(path)
    return any(
        fnmatch.fnmatch(relative, pattern)
        or fnmatch.fnmatch(f"/{relative}/", pattern)
        for pattern in patterns
    )


def _spec_from_track_a(
    path: Path,
    *,
    collection: dict[str, Any],
    defaults: dict[str, Any],
) -> TaskSpec:
    task_id = _track_a_task_id(path)
    return TaskSpec(
        task_id=f"track_a_{task_id}",
        path=_display(path),
        output_dir=f"experiments/track_a/{task_id}",
        collection=str(collection["id"]),
        source=str(collection.get("source", "unknown")),
        tags=tuple(str(item) for item in collection.get("tags", [])),
        timeout_seconds=int(collection.get("timeout_seconds", defaults["timeout_seconds"])),
        max_agent_steps=collection.get("max_agent_steps", defaults.get("max_agent_steps")),
        resume=bool(collection.get("resume", defaults.get("resume", False))),
        preflight=bool(collection.get("preflight", defaults.get("preflight", True))),
    )


def _ancestor_candidates(path: Path, root: Path, *, maximum_depth: int = 3) -> list[Path]:
    candidates: list[Path] = []
    current = path.resolve()
    root = root.resolve()
    for _ in range(maximum_depth + 1):
        try:
            current.relative_to(root)
        except ValueError:
            break
        candidates.append(current)
        if current == root:
            break
        current = current.parent
    return candidates


def _discover_auto_benchmarks(
    root: Path,
    *,
    collection: dict[str, Any],
    defaults: dict[str, Any],
    exclude_patterns: Iterable[str],
) -> list[TaskSpec]:
    if not root.is_dir():
        return []

    descriptors = sorted(
        {
            *root.rglob("task.cfg"),
            *root.rglob("*.tcl"),
        }
    )
    resolved: dict[tuple[str, str], TaskSpec] = {}
    attempted_roots: set[Path] = set()

    for descriptor in descriptors:
        if _excluded(descriptor, exclude_patterns):
            continue
        for candidate_root in _ancestor_candidates(descriptor.parent, root):
            if candidate_root in attempted_roots:
                continue
            attempted_roots.add(candidate_root)
            try:
                task = resolve_benchmark(candidate_root)
            except (FileNotFoundError, KeyError, RuntimeError, ValueError):
                continue

            source = str(task.data.get("artifacts", {}).get("source", ""))
            key = (task.task_id, source)
            resolved[key] = TaskSpec(
                task_id=task.task_id,
                path=_display(candidate_root),
                output_dir=str(task.output_dir),
                collection=str(collection["id"]),
                source=str(collection.get("source", "unknown")),
                tags=tuple(str(item) for item in collection.get("tags", [])),
                timeout_seconds=int(
                    collection.get("timeout_seconds", defaults["timeout_seconds"])
                ),
                max_agent_steps=collection.get(
                    "max_agent_steps", defaults.get("max_agent_steps")
                ),
                resume=bool(collection.get("resume", defaults.get("resume", False))),
                preflight=bool(
                    collection.get("preflight", defaults.get("preflight", True))
                ),
            )
            break

    return sorted(resolved.values(), key=lambda item: (item.collection, item.task_id))


def _expand_suite(suite: dict[str, Any]) -> list[TaskSpec]:
    defaults = dict(suite.get("defaults") or {})
    defaults.setdefault("timeout_seconds", 14400)
    defaults.setdefault("continue_on_error", True)
    defaults.setdefault("preflight", True)
    defaults.setdefault("resume", False)
    defaults.setdefault("max_agent_steps", None)
    excludes = [str(item) for item in suite.get("exclude_paths", [])]

    tasks: list[TaskSpec] = []
    for collection in suite["collections"]:
        if not isinstance(collection, dict) or collection.get("enabled", True) is False:
            continue
        collection_type = collection.get("type")
        root_value = collection.get("root")
        if not isinstance(root_value, str) or not root_value:
            raise SuiteError(f"Collection {collection.get('id')} requires root")
        root = Path(root_value)
        root = root if root.is_absolute() else REPO_ROOT / root

        if collection_type == "track_a_packages":
            if not root.is_dir():
                continue
            for task_file in sorted(root.glob("*/task.toml")):
                task_root = task_file.parent
                if not _excluded(task_root, excludes):
                    tasks.append(
                        _spec_from_track_a(
                            task_root,
                            collection=collection,
                            defaults=defaults,
                        )
                    )
            continue

        if collection_type == "auto_benchmarks":
            tasks.extend(
                _discover_auto_benchmarks(
                    root,
                    collection=collection,
                    defaults=defaults,
                    exclude_patterns=excludes,
                )
            )
            continue

        raise SuiteError(
            f"Unsupported collection type {collection_type!r} in {collection.get('id')}"
        )

    deduplicated: dict[tuple[str, str], TaskSpec] = {}
    for task in tasks:
        deduplicated[(task.task_id, str(task.resolved_output_dir))] = task
    return sorted(
        deduplicated.values(),
        key=lambda item: (item.collection, item.task_id, item.path),
    )


def _matches(task: TaskSpec, patterns: list[str]) -> bool:
    if not patterns:
        return True
    values = [task.task_id, task.path, task.collection, task.source, *task.tags]
    return any(
        fnmatch.fnmatch(value, pattern) or pattern.lower() in value.lower()
        for pattern in patterns
        for value in values
    )


def _filter_tasks(
    tasks: list[TaskSpec],
    *,
    only: list[str],
    skip: list[str],
    maximum: int | None,
) -> list[TaskSpec]:
    selected = [
        task
        for task in tasks
        if _matches(task, only) and not (skip and _matches(task, skip))
    ]
    return selected[:maximum] if maximum is not None else selected


def _preflight(task: TaskSpec, run_dir: Path) -> tuple[str, str | None]:
    if not task.preflight:
        return "disabled", None
    if not task.resolved_path.is_dir():
        return "not_applicable", None

    log_path = run_dir / "preflight" / f"{_safe_id(task.task_id)}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_agent.py"),
        str(task.resolved_path),
        "--onboard-only",
    ]
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=180,
            check=False,
        )
    return (
        ("passed", _display(log_path))
        if completed.returncode == 0
        else ("failed", _display(log_path))
    )


def _archive_existing_output(task: TaskSpec, run_dir: Path) -> str | None:
    output = task.resolved_output_dir
    if not output.exists():
        return None
    try:
        output.resolve().relative_to(REPO_ROOT)
    except ValueError as error:
        raise SuiteError(
            f"Refusing to archive output outside the repository: {output}"
        ) from error

    destination = run_dir / "archived_outputs" / _safe_id(task.task_id)
    suffix = 1
    while destination.exists():
        suffix += 1
        destination = run_dir / "archived_outputs" / f"{_safe_id(task.task_id)}_{suffix}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(output), str(destination))
    return _display(destination)


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


def _selected_values(result: dict[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    selected = result.get("selected_design")
    if isinstance(selected, dict):
        metrics = selected.get("metrics")
        return (
            selected.get("candidate_index"),
            selected.get("candidate_file") or selected.get("archived_file"),
            metrics if isinstance(metrics, dict) else {},
        )
    if isinstance(selected, str):
        return None, selected, {}
    return None, None, {}


def _result_row(
    task: TaskSpec,
    *,
    suite_run_id: str,
    started_at: str,
    finished_at: str,
    elapsed_seconds: float,
    exit_code: int | None,
    timed_out: bool,
    preflight: str,
    log_path: Path,
    error: str | None = None,
) -> dict[str, Any]:
    output_dir = task.resolved_output_dir
    result_path = output_dir / "unified_agent_result.json"
    budget_path = output_dir / "budget_summary.json"
    result = _load_json(result_path)
    budget = _load_json(budget_path)
    consumed = budget.get("consumed") if isinstance(budget.get("consumed"), dict) else {}
    track_a = budget.get("track_a") if isinstance(budget.get("track_a"), dict) else {}
    candidate_index, candidate_file, metrics = _selected_values(result)

    return {
        "suite_run_id": suite_run_id,
        "task_id": task.task_id,
        "task_path": task.path,
        "collection": task.collection,
        "source": task.source,
        "tags": ";".join(task.tags),
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "preflight": preflight,
        "success": result.get("success"),
        "status": result.get("status"),
        "termination_reason": result.get("termination_reason"),
        "selection_mode": result.get("selection_mode"),
        "final_design_verified": result.get("final_design_verified"),
        "meets_submission_frequency": result.get("meets_submission_frequency"),
        "selected_candidate_index": candidate_index,
        "selected_candidate_file": candidate_file,
        "estimated_frequency_mhz": metrics.get("frequency_mhz")
        or metrics.get("estimated_frequency_mhz"),
        "latency_ns": metrics.get("latency_ns"),
        "throughput_period_ns": metrics.get("throughput_period_ns"),
        "lut": metrics.get("resources_lut_used"),
        "ff": metrics.get("resources_ff_used"),
        "dsp": metrics.get("resources_dsp_used"),
        "bram": metrics.get("resources_bram_used"),
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
        "result_path": _display(result_path) if result_path.is_file() else None,
        "log_path": _display(log_path),
        "error": error,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in SUMMARY_FIELDS})
    temporary.replace(path)


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
        if line.strip() and not line.lstrip().startswith(current_pid + " ")
    ]


def _check_prerequisites(*, allow_concurrent: bool) -> None:
    if shutil.which("vitis-run") is None:
        raise SuiteError("vitis-run is not available; source Vitis 2025.2 settings first")

    provider = os.environ.get("LLM4HLS_PROVIDER", "siliconflow").lower()
    required_key = "OPENROUTER_API_KEY" if provider == "openrouter" else "SILICONFLOW_API_KEY"
    if not os.environ.get(required_key):
        raise SuiteError(f"{required_key} is not set")

    active = _active_hls_processes()
    if active and not allow_concurrent:
        rendered = "\n".join(f"  {line}" for line in active)
        raise SuiteError(
            "Another agent/Vitis process is active. Start the suite after it finishes "
            "or pass --allow-concurrent deliberately:\n" + rendered
        )


def _load_resumed_state(path: Path) -> dict[str, Any]:
    state = _load_json(path / "suite_state.json")
    if not state:
        raise SuiteError(f"No suite_state.json found under {path}")
    return state


def _print_tasks(tasks: list[TaskSpec]) -> None:
    print(f"Discovered {len(tasks)} runnable task(s):")
    for index, task in enumerate(tasks, 1):
        print(
            f"{index:02d}. {task.task_id}\n"
            f"    path={task.path}\n"
            f"    output={task.output_dir}\n"
            f"    source={task.source} tags={','.join(task.tags)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a sequential, resumable library of LLM4HLS tasks."
    )
    parser.add_argument(
        "--suite",
        type=Path,
        default=REPO_ROOT / "configs" / "suites" / "overnight_full.json",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--resume-suite", type=Path)
    parser.add_argument("--rerun-failed", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--skip", action="append", default=[])
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--clear-vitis-cache", action="store_true")
    parser.add_argument("--resume-tasks", action="store_true")
    parser.add_argument("--no-preflight", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--allow-concurrent", action="store_true")
    args = parser.parse_args()

    suite_path = args.suite.resolve()
    suite = _load_suite(suite_path)
    tasks = _filter_tasks(
        _expand_suite(suite),
        only=args.only,
        skip=args.skip,
        maximum=args.max_tasks,
    )
    if args.no_preflight:
        tasks = [TaskSpec(**{**asdict(task), "preflight": False}) for task in tasks]
    if args.resume_tasks:
        tasks = [TaskSpec(**{**asdict(task), "resume": True}) for task in tasks]

    if args.list or args.dry_run:
        _print_tasks(tasks)
        return
    if not tasks:
        raise SystemExit("No runnable tasks matched the suite and filters")

    _check_prerequisites(allow_concurrent=args.allow_concurrent)

    commit = _git_commit()
    if args.resume_suite:
        run_dir = args.resume_suite.resolve()
        state = _load_resumed_state(run_dir)
        run_id = str(state["run_id"])
        rows = [row for row in state.get("results", []) if isinstance(row, dict)]
    else:
        run_id = args.run_id or (
            f"{suite['suite_id']}_{datetime.now().strftime('%Y%m%dT%H%M%S')}_{commit[:7]}"
        )
        run_dir = REPO_ROOT / "results" / "suites" / run_id
        if run_dir.exists():
            raise SuiteError(f"Suite run directory already exists: {run_dir}")
        run_dir.mkdir(parents=True)
        rows = []
        state = {
            "schema_version": 1,
            "run_id": run_id,
            "suite_id": suite["suite_id"],
            "suite_file": _display(suite_path),
            "repository_commit": commit,
            "started_at": _utc_now(),
            "finished_at": None,
            "status": "running",
            "source_registry": suite.get("source_registry", []),
            "tasks": [asdict(task) for task in tasks],
            "results": rows,
        }
        shutil.copy2(suite_path, run_dir / "suite_definition.json")

    if args.clear_vitis_cache:
        cache = Path("/tmp/llm4hls-agent")
        if cache.exists():
            shutil.rmtree(cache)

    completed_by_id = {
        str(row.get("task_id")): row
        for row in rows
        if row.get("task_id")
    }
    continue_on_error = not args.stop_on_error and bool(
        suite.get("defaults", {}).get("continue_on_error", True)
    )

    _write_json_atomic(run_dir / "suite_state.json", state)
    _write_csv(run_dir / "suite_summary.csv", rows)
    print(f"Suite run: {run_id}")
    print(f"Tasks: {len(tasks)}")
    print(f"Results: {_display(run_dir)}")

    interrupted = False
    try:
        for position, task in enumerate(tasks, 1):
            previous = completed_by_id.get(task.task_id)
            if previous and not (
                args.rerun_failed
                and previous.get("success") is not True
            ):
                print(f"[{position}/{len(tasks)}] skip completed {task.task_id}")
                continue

            print(f"\n[{position}/{len(tasks)}] {task.task_id}")
            print(f"  path: {task.path}")
            print(f"  timeout: {task.timeout_seconds}s")

            preflight_status = "disabled"
            try:
                preflight_status, preflight_log = _preflight(task, run_dir)
            except (OSError, subprocess.SubprocessError) as error:
                preflight_status = "failed"
                preflight_log = None
                preflight_error = f"{type(error).__name__}: {error}"
            else:
                preflight_error = None

            if preflight_status == "failed":
                print(f"  preflight failed: {preflight_log or preflight_error}")
                now = _utc_now()
                row = _result_row(
                    task,
                    suite_run_id=run_id,
                    started_at=now,
                    finished_at=now,
                    elapsed_seconds=0.0,
                    exit_code=None,
                    timed_out=False,
                    preflight=preflight_status,
                    log_path=Path(preflight_log) if preflight_log else run_dir / "preflight",
                    error=preflight_error or "onboarding preflight failed",
                )
                rows.append(row)
                completed_by_id[task.task_id] = row
                state["results"] = rows
                _write_json_atomic(run_dir / "suite_state.json", state)
                _write_csv(run_dir / "suite_summary.csv", rows)
                if not continue_on_error:
                    break
                continue

            if args.fresh:
                archived = _archive_existing_output(task, run_dir)
                if archived:
                    print(f"  archived previous output: {archived}")

            command = [
                sys.executable,
                "-u",
                str(REPO_ROOT / "scripts" / "run_agent.py"),
                str(task.resolved_path),
            ]
            if task.resume:
                command.append("--resume")
            if task.max_agent_steps is not None:
                command.extend(["--max-agent-steps", str(task.max_agent_steps)])

            log_path = run_dir / "logs" / f"{_safe_id(task.task_id)}.log"
            environment = dict(os.environ)
            environment["PYTHONUNBUFFERED"] = "1"
            environment["LLM4HLS_SUITE_RUN_ID"] = run_id
            environment["LLM4HLS_SUITE_TASK_ID"] = task.task_id
            started_at = _utc_now()
            print(f"  log: {_display(log_path)}")
            exit_code, timed_out, elapsed = _run_command(
                command,
                log_path=log_path,
                timeout_seconds=task.timeout_seconds,
                environment=environment,
            )
            finished_at = _utc_now()
            error = "task timeout" if timed_out else None
            row = _result_row(
                task,
                suite_run_id=run_id,
                started_at=started_at,
                finished_at=finished_at,
                elapsed_seconds=elapsed,
                exit_code=exit_code,
                timed_out=timed_out,
                preflight=preflight_status,
                log_path=log_path,
                error=error,
            )
            rows.append(row)
            completed_by_id[task.task_id] = row
            state["results"] = rows
            _write_json_atomic(run_dir / "suite_state.json", state)
            _write_csv(run_dir / "suite_summary.csv", rows)

            print(
                "  result: "
                f"exit={exit_code} success={row.get('success')} "
                f"status={row.get('status')} verified={row.get('final_design_verified')}"
            )
            if (exit_code != 0 or row.get("success") is not True) and not continue_on_error:
                break
    except KeyboardInterrupt:
        interrupted = True
        print("\nSuite interrupted; state has been preserved.", file=sys.stderr)
    finally:
        state["finished_at"] = _utc_now()
        state["status"] = "interrupted" if interrupted else "completed"
        state["results"] = rows
        _write_json_atomic(run_dir / "suite_state.json", state)
        _write_csv(run_dir / "suite_summary.csv", rows)

    print(f"\nSuite status: {state['status']}")
    print(f"Summary CSV: {_display(run_dir / 'suite_summary.csv')}")
    print(f"State JSON: {_display(run_dir / 'suite_state.json')}")


if __name__ == "__main__":
    try:
        main()
    except SuiteError as error:
        print(f"Suite configuration error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
