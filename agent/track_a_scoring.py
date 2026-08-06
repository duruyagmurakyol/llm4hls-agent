"""Public Track-A baseline capture and score estimation.

The organiser's hidden tests remain unavailable to the agent. This module
therefore reports a public score estimate using only public correctness and
synthesis evidence already produced by the normal workflow. It never launches
Vitis and never consumes Track-A credits.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

from agent.config import TaskManifest
from agent.state import AgentResult

REPO_ROOT = Path(__file__).resolve().parent.parent
ACCELERATION_CAP = 8.0
CORRECTNESS_WEIGHT = 0.5
SYNTHESIS_WEIGHT = 0.2
PPA_WEIGHT = 0.3


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def official_latency_cycles(metrics: dict[str, Any]) -> int | float | None:
    """Return the organiser-facing latency metric: worst, then average cycles."""

    for key in ("latency_worst_cycles", "latency_average_cycles"):
        value = metrics.get(key)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value >= 0
        ):
            return value
    return None


def estimate_public_track_a_score(
    *,
    difficulty: int,
    public_correct: bool,
    synthesis_passed: bool,
    original_latency_cycles: int | float | None,
    candidate_latency_cycles: int | float | None,
) -> dict[str, Any]:
    """Mirror the public portion of the Track-A grading equation."""

    if not public_correct:
        return {
            "acceleration": None,
            "acceleration_cap": ACCELERATION_CAP,
            "ppa_norm": 0.0,
            "components": {
                "correctness": 0.0,
                "synthesis": 0.0,
                "ppa": 0.0,
            },
            "public_score_estimate": 0.0,
            "maximum_score": float(difficulty),
        }

    baseline = _number(original_latency_cycles)
    candidate = _number(candidate_latency_cycles)
    acceleration: float | None = None
    if synthesis_passed and baseline is not None and candidate is not None:
        if baseline > 0 and candidate > 0:
            acceleration = baseline / candidate

    ppa_norm = (
        min(acceleration, ACCELERATION_CAP) / ACCELERATION_CAP
        if acceleration
        else 0.0
    )
    correctness_component = CORRECTNESS_WEIGHT
    synthesis_component = SYNTHESIS_WEIGHT if synthesis_passed else 0.0
    ppa_component = PPA_WEIGHT * ppa_norm
    score = difficulty * (
        correctness_component + synthesis_component + ppa_component
    )

    return {
        "acceleration": acceleration,
        "acceleration_cap": ACCELERATION_CAP,
        "ppa_norm": ppa_norm,
        "components": {
            "correctness": correctness_component,
            "synthesis": synthesis_component,
            "ppa": ppa_component,
        },
        "public_score_estimate": score,
        "maximum_score": float(difficulty),
    }


def _track_a_metadata(task: TaskManifest) -> dict[str, Any] | None:
    value = task.data.get("track_a")
    return value if isinstance(value, dict) else None


def _target_record(task: TaskManifest) -> dict[str, Any]:
    target = task.data.get("target")
    target = target if isinstance(target, dict) else {}
    clock = _number(target.get("clock_period_ns"))
    return {
        "part": target.get("part"),
        "platform": target.get("platform"),
        "clock_period_ns": clock,
        "frequency_mhz": 1000.0 / clock if clock and clock > 0 else None,
    }


def capture_original_scoring_baseline(task: TaskManifest) -> Path | None:
    """Copy the original public kernel before repair or optimisation starts."""

    if _track_a_metadata(task) is None:
        return None

    source = _resolve(task.data["artifacts"]["source"])
    if not source.is_file():
        raise FileNotFoundError(f"Track-A original source not found: {source}")

    output_dir = _resolve(task.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stable_source = output_dir / f"original_scoring_baseline{source.suffix or '.cpp'}"
    record_path = output_dir / "original_scoring_baseline.json"
    digest = _sha256(source)
    target = _target_record(task)
    previous = _load_json(record_path)
    reusable = (
        previous.get("candidate_hash") == digest
        and previous.get("target") == target
        and stable_source.is_file()
    )

    shutil.copy2(source, stable_source)
    if _sha256(stable_source) != digest:
        raise RuntimeError("Original scoring-baseline copy does not match its source")

    if reusable:
        record = dict(previous)
        record.update(
            {
                "schema_version": 1,
                "source": _display(stable_source),
                "original_source": _display(source),
                "candidate_hash": digest,
                "target": target,
            }
        )
    else:
        shutil.rmtree(
            output_dir / "original_scoring_baseline_project",
            ignore_errors=True,
        )
        record = {
            "schema_version": 1,
            "source": _display(stable_source),
            "original_source": _display(source),
            "candidate_hash": digest,
            "target": target,
            "synthesis_passed": None,
            "project_dir": None,
            "top_csynth_xml": None,
            "reports": [],
            "metrics": {},
            "official_latency_cycles": None,
        }

    return _write_json(record_path, record)


def _original_synthesis_result(
    task: TaskManifest,
    candidate_hash: str,
) -> dict[str, Any]:
    output_dir = _resolve(task.output_dir)
    direct = output_dir / "synthesis" / candidate_hash[:12] / "result.json"
    candidates = (
        [direct]
        if direct.is_file()
        else sorted((output_dir / "synthesis").glob("*/result.json"))
    )
    for path in candidates:
        report = _load_json(path)
        if report.get("candidate_hash") == candidate_hash:
            return report
    return {}


def refresh_original_scoring_baseline(
    task: TaskManifest,
) -> dict[str, Any] | None:
    """Attach existing synthesis evidence without launching another tool call."""

    record_path = capture_original_scoring_baseline(task)
    if record_path is None:
        return None

    record = _load_json(record_path)
    synthesis = _original_synthesis_result(task, str(record["candidate_hash"]))
    if not synthesis:
        return record

    passed = synthesis.get("passed") is True
    record["synthesis_passed"] = passed
    record["metrics"] = dict(synthesis.get("metrics") or {}) if passed else {}
    record["official_latency_cycles"] = official_latency_cycles(record["metrics"])
    record["top_csynth_xml"] = None
    record["reports"] = []
    record["project_dir"] = None

    if passed:
        project_dir = Path(str(synthesis.get("project_dir", ""))).expanduser()
        report_root = project_dir / "solution1/syn/report"
        reports = sorted(report_root.glob("*_csynth.xml"))
        if reports:
            output_dir = _resolve(task.output_dir)
            stable_project = output_dir / "original_scoring_baseline_project"
            shutil.rmtree(stable_project, ignore_errors=True)
            stable_report_root = stable_project / "solution1/syn/report"
            stable_report_root.mkdir(parents=True)
            copied: list[str] = []
            for report in reports:
                destination = stable_report_root / report.name
                shutil.copy2(report, destination)
                copied.append(_display(destination))
            top = str(task.data["interface"]["top_function"])
            top_report = stable_report_root / f"{top}_csynth.xml"
            record["project_dir"] = _display(stable_project)
            record["reports"] = copied
            record["top_csynth_xml"] = (
                _display(top_report) if top_report.is_file() else None
            )

    _write_json(record_path, record)
    return record


def _normalise_validation(value: dict[str, Any]) -> dict[str, bool | None]:
    return {
        "csim": value.get("csim", value.get("csim_passed")),
        "synthesis": value.get("synthesis", value.get("synthesis_passed")),
        "cosim": value.get("cosim", value.get("cosim_passed")),
    }


def _selected_evidence(
    task: TaskManifest,
    result: AgentResult,
) -> dict[str, Any]:
    output_dir = _resolve(task.output_dir)
    selected = result.to_dict().get("selected_design")

    if isinstance(selected, dict):
        validation = selected.get("validation")
        return {
            "source": selected.get("archived_file")
            or selected.get("candidate_file"),
            "candidate_hash": selected.get("candidate_hash"),
            "metrics": dict(selected.get("metrics") or {}),
            "validation": _normalise_validation(
                validation if isinstance(validation, dict) else {}
            ),
        }

    state = _load_json(output_dir / "candidate_state.json")
    state_selected = state.get("selected_design")
    if isinstance(state_selected, dict) and isinstance(selected, str):
        archived = state_selected.get("archived_file")
        if isinstance(archived, str) and _resolve(archived) == _resolve(selected):
            validation = state_selected.get("validation")
            return {
                "source": archived,
                "candidate_hash": state_selected.get("candidate_hash"),
                "metrics": dict(state_selected.get("metrics") or {}),
                "validation": _normalise_validation(
                    validation if isinstance(validation, dict) else {}
                ),
            }

    baseline = _load_json(output_dir / "verified_baseline.json")
    if baseline:
        validation = baseline.get("validation")
        return {
            "source": baseline.get("source") or selected,
            "candidate_hash": baseline.get("candidate_hash"),
            "metrics": dict(baseline.get("metrics") or {}),
            "validation": _normalise_validation(
                validation if isinstance(validation, dict) else {}
            ),
        }

    if isinstance(selected, str):
        match = re.search(r"candidate_(\d{3})\.cpp$", selected)
        if match:
            prefix = f"candidate_{match.group(1)}"
            synthesis = _load_json(output_dir / f"{prefix}_synthesis.json")
            csim = _load_json(output_dir / f"{prefix}_csim_validation.json")
            cosim = _load_json(output_dir / f"{prefix}_cosim.json")
            return {
                "source": selected,
                "candidate_hash": synthesis.get("candidate_hash"),
                "metrics": dict(synthesis.get("metrics") or {}),
                "validation": {
                    "csim": csim.get("passed"),
                    "synthesis": synthesis.get("passed"),
                    "cosim": cosim.get("passed") if cosim else None,
                },
            }

    return {
        "source": selected,
        "candidate_hash": None,
        "metrics": {},
        "validation": {
            "csim": None,
            "synthesis": None,
            "cosim": None,
        },
    }


def write_track_a_score_estimate(
    task: TaskManifest,
    result: AgentResult,
) -> tuple[Path, dict[str, Any]] | None:
    """Write a transparent public score estimate for one completed Track-A run."""

    metadata = _track_a_metadata(task)
    if metadata is None:
        return None

    original = refresh_original_scoring_baseline(task)
    if original is None:
        return None

    selected = _selected_evidence(task, result)
    validation = selected["validation"]
    requires_cosim = bool(metadata.get("requires_cosim", False))
    public_correct = bool(
        validation.get("csim") is True
        and (not requires_cosim or validation.get("cosim") is True)
    )
    synthesis_passed = validation.get("synthesis") is True
    selected_metrics = selected["metrics"]
    selected_latency = official_latency_cycles(selected_metrics)
    original_latency = original.get("official_latency_cycles")
    difficulty = int(metadata.get("difficulty", 1))
    score = estimate_public_track_a_score(
        difficulty=difficulty,
        public_correct=public_correct,
        synthesis_passed=synthesis_passed,
        original_latency_cycles=original_latency,
        candidate_latency_cycles=selected_latency,
    )

    target = _target_record(task)
    estimated_clock = _number(selected_metrics.get("clock_period_ns"))
    target_clock = _number(target.get("clock_period_ns"))
    estimated_frequency = (
        1000.0 / estimated_clock
        if estimated_clock is not None and estimated_clock > 0
        else None
    )
    meets_target = (
        estimated_clock <= target_clock
        if estimated_clock is not None and target_clock is not None
        else None
    )

    report = {
        "schema_version": 1,
        "task_id": task.task_id,
        "difficulty": difficulty,
        "basis": "public_tests_and_local_synthesis",
        "hidden_tests_used": False,
        "public_correct": public_correct,
        "required_cosim": requires_cosim,
        "required_cosim_passed": (
            validation.get("cosim") if requires_cosim else None
        ),
        "synthesis_passed": synthesis_passed,
        "original_scoring_baseline": {
            "source": original.get("source"),
            "candidate_hash": original.get("candidate_hash"),
            "synthesis_passed": original.get("synthesis_passed"),
            "metrics": dict(original.get("metrics") or {}),
            "official_latency_cycles": original_latency,
        },
        "selected_design": {
            "source": selected.get("source"),
            "candidate_hash": selected.get("candidate_hash"),
            "metrics": selected_metrics,
            "official_latency_cycles": selected_latency,
            "estimated_clock_period_ns": estimated_clock,
            "estimated_frequency_mhz": estimated_frequency,
            "target_clock_period_ns": target_clock,
            "target_frequency_mhz": target.get("frequency_mhz"),
            "meets_target_clock": meets_target,
        },
        **score,
        "note": (
            "This is a public estimate only. The organiser's hidden functional "
            "tests can reduce the final score to zero."
        ),
    }
    path = _write_json(
        _resolve(task.output_dir) / "track_a_score_estimate.json",
        report,
    )
    return path, report
