"""Promote a fully validated source into the active PPA baseline."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from agent.config import TaskManifest

REPO_ROOT = Path(__file__).resolve().parent.parent


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


def promote_verified_baseline(
    task: TaskManifest,
    source: Path,
    *,
    origin: str,
    csim_passed: bool,
    synthesis: dict[str, Any],
    cosim: dict[str, Any],
) -> dict[str, Any]:
    """Copy one verified source and its synthesis reports into stable task output."""
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Verified baseline source not found: {source}")
    if not csim_passed:
        raise ValueError("Cannot promote a baseline that did not pass CSim")
    if synthesis.get("passed") is not True:
        raise ValueError("Cannot promote a baseline that did not pass synthesis")
    if cosim.get("passed") is not True:
        raise ValueError("Cannot promote a baseline that did not pass co-simulation")

    digest = _sha256(source)
    synthesis_hash = str(synthesis.get("candidate_hash", ""))
    cosim_hash = str(cosim.get("candidate_hash", ""))
    if digest != synthesis_hash or digest != cosim_hash:
        raise ValueError("Verified baseline hashes do not identify the same source")

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
        "schema_version": 1,
        "origin": origin,
        "source": _display(baseline_source),
        "original_source": _display(source),
        "candidate_hash": digest,
        "project_dir": _display(stable_project),
        "top_csynth_xml": _display(top_report),
        "reports": copied_reports,
        "metrics": dict(metrics),
        "validation": {
            "csim_passed": True,
            "synthesis_passed": True,
            "cosim_passed": True,
        },
    }
    (output_dir / "verified_baseline.json").write_text(
        json.dumps(record, indent=2) + "\n",
        encoding="utf-8",
    )
    return record
