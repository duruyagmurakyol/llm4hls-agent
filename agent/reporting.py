"""Offline extraction of final evidence from completed agent suite runs."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
METRIC_KEYS = (
    "latency_ns",
    "throughput_period_ns",
    "resources_lut_used",
    "resources_ff_used",
    "resources_dsp_used",
    "resources_bram_used",
    "frequency_mhz",
)


def _load_object(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"JSON file not found: {path}")
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected in {path}")
    return value


def _resolve(value: str | Path, repo_root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else repo_root / path


def _number(value: Any) -> float | int | None:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _percent_delta(value: Any, baseline: Any) -> float | None:
    current = _number(value)
    reference = _number(baseline)
    if current is None or reference in (None, 0):
        return None
    return round((float(current) - float(reference)) * 100.0 / float(reference), 6)


def _selected_outcome(selected: dict[str, Any]) -> str:
    index = selected.get("candidate_index")
    if index == 0:
        return "baseline_retained"
    if isinstance(index, int) and index > 0:
        return "optimised_candidate"
    return "unselected"


def _task_output_dir(row: dict[str, Any], suite_root: Path, repo_root: Path) -> Path:
    unified = row.get("unified_result")
    if isinstance(unified, str) and unified:
        return _resolve(unified, repo_root).parent
    task_id = row.get("task_id")
    if isinstance(task_id, str) and task_id:
        return suite_root / "tasks" / task_id
    raise ValueError("Suite row has neither unified_result nor task_id")


def extract_run_rows(suite_root: Path, *, repo_root: Path = REPO_ROOT) -> list[dict[str, Any]]:
    suite_root = suite_root.expanduser().resolve()
    repo_root = repo_root.expanduser().resolve()
    summary = _load_object(suite_root / "summary.json")
    raw_rows = summary.get("rows")
    if not isinstance(raw_rows, list):
        raise ValueError(f"summary.rows must be a list in {suite_root / 'summary.json'}")

    extracted: list[dict[str, Any]] = []
    for raw in raw_rows:
        if not isinstance(raw, dict):
            continue
        output_dir = _task_output_dir(raw, suite_root, repo_root)
        experiment = _load_object(output_dir / "experiment_summary.json", required=False)
        selected = experiment.get("selected_design")
        selected = selected if isinstance(selected, dict) else {}
        baseline_metrics = experiment.get("baseline_metrics")
        baseline_metrics = baseline_metrics if isinstance(baseline_metrics, dict) else {}
        selected_metrics = selected.get("metrics")
        selected_metrics = selected_metrics if isinstance(selected_metrics, dict) else {}
        candidates = experiment.get("candidates")
        candidates = candidates if isinstance(candidates, list) else []

        row: dict[str, Any] = {
            "benchmark": raw.get("benchmark"),
            "case": raw.get("case"),
            "repetition": raw.get("repetition"),
            "task_id": raw.get("task_id"),
            "success": raw.get("success"),
            "status": raw.get("status"),
            "termination_reason": raw.get("termination_reason"),
            "timed_out": raw.get("timed_out"),
            "elapsed_seconds": raw.get("elapsed_seconds"),
            "input_tokens": raw.get("input_tokens"),
            "output_tokens": raw.get("output_tokens"),
            "total_tokens": raw.get("total_tokens"),
            "budget_iterations_used": raw.get("budget_iterations_used"),
            "budget_model_calls_used": raw.get("budget_model_calls_used"),
            "budget_csim_calls_used": raw.get("budget_csim_calls_used"),
            "budget_synthesis_calls_used": raw.get("budget_synthesis_calls_used"),
            "budget_cosim_calls_used": raw.get("budget_cosim_calls_used"),
            "candidate_count": len(candidates),
            "fully_verified_candidate_count": sum(
                isinstance(candidate, dict) and candidate.get("fully_verified") is True
                for candidate in candidates
            ),
            "selected_candidate": selected.get("candidate_index"),
            "selected_outcome": _selected_outcome(selected),
            "selected_verdict": selected.get("verdict"),
            "selected_fully_verified": selected.get("fully_verified"),
            "selected_frequency_compliant": selected.get("meets_frequency_requirement"),
            "selected_resource_compliant": selected.get("meets_resource_limits"),
            "experiment_summary": str(output_dir / "experiment_summary.json"),
        }
        for key in METRIC_KEYS:
            baseline = baseline_metrics.get(key)
            current = selected_metrics.get(key)
            row[f"baseline_{key}"] = baseline
            row[f"selected_{key}"] = current
            row[f"{key}_delta_percent"] = _percent_delta(current, baseline)
        extracted.append(row)
    return extracted


def _numeric(values: Iterable[Any]) -> list[float]:
    return [float(value) for value in values if _number(value) is not None]


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    successes = sum(row.get("success") is True for row in rows)
    optimised = sum(row.get("selected_outcome") == "optimised_candidate" for row in rows)
    retained = sum(row.get("selected_outcome") == "baseline_retained" for row in rows)
    unselected = total - optimised - retained

    result: dict[str, Any] = {
        "runs": total,
        "successful_runs": successes,
        "failed_runs": total - successes,
        "success_rate_percent": round(successes * 100.0 / total, 3) if total else 0.0,
        "optimised_candidate_selected": optimised,
        "baseline_retained": retained,
        "unselected": unselected,
        "optimised_selection_rate_percent": round(optimised * 100.0 / total, 3) if total else 0.0,
    }
    for key in (
        "elapsed_seconds",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "budget_iterations_used",
        "budget_model_calls_used",
        "budget_csim_calls_used",
        "budget_synthesis_calls_used",
        "budget_cosim_calls_used",
    ):
        values = _numeric(row.get(key) for row in rows)
        result[f"total_{key}"] = round(sum(values), 6) if values else 0
        result[f"mean_{key}"] = round(mean(values), 6) if values else None
    result["total_elapsed_hours"] = round(float(result["total_elapsed_seconds"]) / 3600.0, 6)

    for key in METRIC_KEYS:
        values = _numeric(row.get(f"{key}_delta_percent") for row in rows)
        result[f"mean_{key}_delta_percent"] = round(mean(values), 6) if values else None
        result[f"median_{key}_delta_percent"] = round(median(values), 6) if values else None
    return result


def _group(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key) for key in keys)].append(row)
    output: list[dict[str, Any]] = []
    for identity in sorted(groups, key=lambda item: tuple(str(value) for value in item)):
        item = {key: value for key, value in zip(keys, identity)}
        item.update(_aggregate(groups[identity]))
        output.append(item)
    return output


def build_final_report(suite_root: Path, *, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    suite_root = suite_root.expanduser().resolve()
    rows = extract_run_rows(suite_root, repo_root=repo_root)
    return {
        "schema_version": 1,
        "suite_root": str(suite_root),
        "overall": _aggregate(rows),
        "by_benchmark": _group(rows, ("benchmark",)),
        "by_case": _group(rows, ("benchmark", "case")),
        "rows": rows,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _format(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return "" if value is None else str(value)


def render_markdown(report: dict[str, Any]) -> str:
    overall = report["overall"]
    lines = [
        "# Final agent results",
        "",
        "## Overall",
        "",
        "| Measure | Value |",
        "|---|---:|",
        f"| Completed runs | {_format(overall['runs'])} |",
        f"| Successful runs | {_format(overall['successful_runs'])} |",
        f"| Success rate | {_format(overall['success_rate_percent'])}% |",
        f"| Optimised candidate selected | {_format(overall['optimised_candidate_selected'])} |",
        f"| Baseline retained | {_format(overall['baseline_retained'])} |",
        f"| Input tokens | {_format(overall['total_input_tokens'])} |",
        f"| Output tokens | {_format(overall['total_output_tokens'])} |",
        f"| Total tokens | {_format(overall['total_total_tokens'])} |",
        f"| Model calls | {_format(overall['total_budget_model_calls_used'])} |",
        f"| Runtime | {_format(overall['total_elapsed_hours'])} h |",
        "",
        "## By benchmark",
        "",
        "| Benchmark | Runs | Success | Optimised | Baseline retained | Mean tokens |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in report["by_benchmark"]:
        lines.append(
            "| {benchmark} | {runs} | {success_rate}% | {optimised} | {retained} | {tokens} |".format(
                benchmark=_format(item.get("benchmark")),
                runs=_format(item.get("runs")),
                success_rate=_format(item.get("success_rate_percent")),
                optimised=_format(item.get("optimised_candidate_selected")),
                retained=_format(item.get("baseline_retained")),
                tokens=_format(item.get("mean_total_tokens")),
            )
        )
    lines.extend(
        [
            "",
            "## By case",
            "",
            "| Benchmark | Case | Runs | Success | Optimised | Baseline retained |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for item in report["by_case"]:
        lines.append(
            "| {benchmark} | {case} | {runs} | {success_rate}% | {optimised} | {retained} |".format(
                benchmark=_format(item.get("benchmark")),
                case=_format(item.get("case")),
                runs=_format(item.get("runs")),
                success_rate=_format(item.get("success_rate_percent")),
                optimised=_format(item.get("optimised_candidate_selected")),
                retained=_format(item.get("baseline_retained")),
            )
        )
    lines.extend(["", "Run-level evidence is available in `final_results.csv`.", ""])
    return "\n".join(lines)


def write_final_report(
    suite_root: Path,
    *,
    output_dir: Path | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Path]:
    suite_root = suite_root.expanduser().resolve()
    output_dir = (output_dir or suite_root).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_final_report(suite_root, repo_root=repo_root)

    json_path = output_dir / "final_results.json"
    csv_path = output_dir / "final_results.csv"
    markdown_path = output_dir / "final_results.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _write_csv(csv_path, report["rows"])
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "markdown": markdown_path}
