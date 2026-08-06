"""Promote a source that passed every verification stage required by its task."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from agent.config import TaskManifest

REPO_ROOT = Path(__file__).resolve().parent.parent
DERIVED_BASELINE_FILES = (
    "baseline_hierarchical_diagnosis.json",
    "baseline_source_target.json",
    "baseline_source_cause.json",
    "baseline_metrics.json",
    "candidate_001_prompt.txt",
    "experiment_summary.json",
)


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _invalidate_changed_baseline(
    output_dir: Path,
    *,
    candidate_hash: str,
    source: str,
) -> None:
    record_path = output_dir / "verified_baseline.json"
    if not record_path.is_file():
        changed = True
    else:
        try:
            previous = json.loads(record_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            changed = True
        else:
            changed = (
                previous.get("candidate_hash") != candidate_hash
                or previous.get("source") != source
            )
    if changed:
        for name in DERIVED_BASELINE_FILES:
            (output_dir / name).unlink(missing_ok=True)


def promote_verified_baseline(
    task: TaskManifest,
    source: Path,
    *,
    origin: str,
    csim_passed: bool,
    synthesis: dict[str, Any],
    cosim: dict[str, Any] | None,
    cosim_required: bool = True,
) -> dict[str, Any]:
    """Copy one task-valid source and its synthesis reports into stable output."""

    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Verified baseline source not found: {source}")
    if not csim_passed:
        raise ValueError("Cannot promote a baseline that did not pass CSim")
    if synthesis.get("passed") is not True:
        raise ValueError("Cannot promote a baseline that did not pass synthesis")
    if cosim_required and (cosim is None or cosim.get("passed") is not True):
        raise ValueError(
            "Cannot promote a baseline that did not pass required co-simulation"
        )
    if cosim is not None and cosim.get("passed") is not True:
        raise ValueError("Cannot promote a baseline after failed co-simulation")

    digest = _sha256(source)
    synthesis_hash = str(synthesis.get("candidate_hash", ""))
    if digest != synthesis_hash:
        raise ValueError("Verified baseline and synthesis hashes do not match")
    if cosim is not None:
        cosim_hash = str(cosim.get("candidate_hash", ""))
        if digest != cosim_hash:
            raise ValueError("Verified baseline and co-simulation hashes do not match")

    metrics = synthesis.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("Verified baseline synthesis metrics are missing")

    synthesis_project = Path(str(synthesis.get("project_dir", ""))).expanduser()
    report_root = synthesis_project / "solution1/syn/report"
    reports = sorted(report_root.glob("*_csynth.xml"))
    if not reports:
        raise FileNotFoundError(
            f"Verified baseline synthesis reports not found: {report_root}"
        )

    output_dir = _resolve(task.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_source = output_dir / f"active_baseline{source.suffix or '.cpp'}"
    displayed_source = _display(baseline_source)
    _invalidate_changed_baseline(
        output_dir,
        candidate_hash=digest,
        source=displayed_source,
    )

    shutil.copy2(source, baseline_source)
    if _sha256(baseline_source) != digest:
        raise RuntimeError("Copied baseline source hash does not match the verified source")

    stable_project = output_dir / "verified_baseline_project"
    shutil.rmtree(stable_project, ignore_errors=True)
    stable_report_root = stable_project / "solution1/syn/report"
    stable_report_root.mkdir(parents=True)
    copied_reports: list[str] = []
    for report in reports:
        destination = stable_report_root / report.name
        shutil.copy2(report, destination)
        copied_reports.append(_display(destination))

    top_name = str(synthesis.get("top_function", task.data["interface"]["top_function"]))
    top_report = stable_report_root / f"{top_name}_csynth.xml"
    if not top_report.is_file():
        raise FileNotFoundError(f"Promoted top synthesis report is missing: {top_report}")

    record = {
        "schema_version": 2,
        "origin": origin,
        "source": displayed_source,
        "original_source": _display(source),
        "candidate_hash": digest,
        "project_dir": _display(stable_project),
        "top_csynth_xml": _display(top_report),
        "reports": copied_reports,
        "metrics": dict(metrics),
        "validation": {
            "csim_passed": True,
            "synthesis_passed": True,
            "cosim_required": cosim_required,
            "cosim_passed": True if cosim is not None else None,
        },
    }
    (output_dir / "verified_baseline.json").write_text(
        json.dumps(record, indent=2) + "\n",
        encoding="utf-8",
    )
    return record
