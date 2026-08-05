"""Human-readable terminal summaries for one agent run or an experiment suite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

METRICS = (
    ("latency_ns", "Latency", "ns"),
    ("throughput_period_ns", "Throughput period", "ns"),
    ("resources_lut_used", "LUT", ""),
    ("resources_ff_used", "FF", ""),
    ("resources_dsp_used", "DSP", ""),
    ("resources_bram_used", "BRAM", ""),
    ("frequency_mhz", "Frequency", "MHz"),
)


def _load_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _resolve(value: str | Path, repo_root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else repo_root / path


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _number(value: Any) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def _percent_delta(value: Any, baseline: Any) -> float | None:
    current = _number(value)
    reference = _number(baseline)
    if current is None or reference in (None, 0):
        return None
    return (float(current) - float(reference)) * 100.0 / float(reference)


def _task_identity(task_id: str) -> tuple[str, str]:
    base = task_id.split("__", 1)[0]
    suffix = "_repair_full_agent"
    if base.endswith(suffix):
        base = base[: -len(suffix)]

    if base.startswith("dot_product_"):
        return "dot_product", base.removeprefix("dot_product_")
    if base.startswith("gemm_"):
        return "gemm", base.removeprefix("gemm_")
    if base.startswith("hls_eval_"):
        remainder = base.removeprefix("hls_eval_")
        benchmark, separator, case = remainder.partition("_")
        if separator:
            return benchmark, case
    return base, ""


def _selection_reason(
    selected_index: Any,
    selected: dict[str, Any],
    candidates: list[Any],
) -> str | None:
    if isinstance(selected_index, int) and selected_index > 0:
        record = next(
            (
                candidate
                for candidate in candidates
                if isinstance(candidate, dict)
                and candidate.get("candidate_index") == selected_index
            ),
            None,
        )
        if isinstance(record, dict) and isinstance(record.get("reason"), str):
            return record["reason"]
        return None

    if selected_index == 0:
        verified = sum(
            isinstance(candidate, dict) and candidate.get("fully_verified") is True
            for candidate in candidates
        )
        if not candidates:
            return "No optimisation candidate was evaluated."
        if verified == 0:
            return "No optimisation candidate completed full verification."
        return (
            "No fully verified candidate ranked above the baseline under the "
            "configured selection policy."
        )
    return None


def build_run_summary(
    result: Any,
    *,
    elapsed_seconds: float,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Build one authoritative summary from the run's persisted artefacts."""
    repo_root = repo_root.expanduser().resolve()
    task_id = str(_field(result, "task_id", "unknown_task"))
    output_dir = _resolve(str(_field(result, "output_dir", "")), repo_root)
    budget = _load_object(output_dir / "budget_summary.json")
    experiment = _load_object(output_dir / "experiment_summary.json")

    consumed = budget.get("consumed")
    consumed = consumed if isinstance(consumed, dict) else {}
    candidates = experiment.get("candidates")
    candidates = candidates if isinstance(candidates, list) else []
    selected = experiment.get("selected_design")
    selected = selected if isinstance(selected, dict) else {}
    baseline_metrics = experiment.get("baseline_metrics")
    baseline_metrics = baseline_metrics if isinstance(baseline_metrics, dict) else {}
    selected_metrics = selected.get("metrics")
    selected_metrics = selected_metrics if isinstance(selected_metrics, dict) else {}

    selected_index = selected.get("candidate_index")
    status = str(_field(result, "status", "unknown"))
    success = _field(result, "success") is True
    status_only = status.startswith("status_") or status == "status_only"

    if selected_index == 0:
        final_selection = "baseline"
    elif isinstance(selected_index, int) and selected_index > 0:
        final_selection = f"candidate {selected_index}"
    elif status_only:
        final_selection = "not applicable"
    elif success:
        final_selection = "repaired design"
    else:
        final_selection = "none"

    benchmark, case = _task_identity(task_id)
    task_label = f"{benchmark}/{case}" if case else benchmark
    selected_verified = selected.get("fully_verified")
    final_verified = (
        bool(selected_verified)
        if isinstance(selected_verified, bool)
        else success
    )

    metric_rows: list[dict[str, Any]] = []
    if isinstance(selected_index, int) and selected_index > 0:
        for key, label, unit in METRICS:
            baseline = baseline_metrics.get(key)
            current = selected_metrics.get(key)
            if _number(baseline) is None or _number(current) is None:
                continue
            metric_rows.append(
                {
                    "key": key,
                    "label": label,
                    "unit": unit,
                    "baseline": baseline,
                    "selected": current,
                    "delta_percent": _percent_delta(current, baseline),
                }
            )

    return {
        "task": task_label,
        "task_id": task_id,
        "success": success,
        "status": status,
        "termination_reason": _field(result, "termination_reason"),
        "final_selection": final_selection,
        "selected_candidate": selected_index,
        "selection_verdict": selected.get("verdict") or (
            "repair_only" if not experiment and success else None
        ),
        "baseline_retained": selected_index == 0 if selected_index is not None else None,
        "selection_reason": _selection_reason(selected_index, selected, candidates),
        "candidate_count": len(candidates),
        "fully_verified_candidate_count": sum(
            isinstance(candidate, dict) and candidate.get("fully_verified") is True
            for candidate in candidates
        ),
        "model_calls": consumed.get("model_calls"),
        "input_tokens": consumed.get("input_tokens"),
        "output_tokens": consumed.get("output_tokens"),
        "total_tokens": consumed.get("total_tokens"),
        "csim_calls": consumed.get("csim_calls"),
        "synthesis_calls": consumed.get("synthesis_calls"),
        "cosim_calls": consumed.get("cosim_calls"),
        "elapsed_seconds": elapsed_seconds,
        "metric_rows": metric_rows,
        "frequency_compliant": selected.get("meets_frequency_requirement"),
        "resource_compliant": selected.get("meets_resource_limits"),
        "final_design_verified": final_verified,
    }


def _count(value: Any) -> str:
    number = _number(value)
    if number is None:
        return "—"
    return f"{int(number):,}" if float(number).is_integer() else f"{float(number):,.2f}"


def _mean(value: Any) -> str:
    number = _number(value)
    return "—" if number is None else f"{float(number):,.2f}"


def _yes_no(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "—"


def _duration(seconds: Any) -> str:
    number = _number(seconds)
    if number is None:
        return "—"
    total = max(0, int(round(float(number))))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours} h {minutes} min {secs} sec"
    if minutes:
        return f"{minutes} min {secs} sec"
    return f"{secs} sec"


def _metric_value(value: Any) -> str:
    number = _number(value)
    if number is None:
        return "—"
    if float(number).is_integer():
        return f"{int(number):,}"
    return f"{float(number):,.3f}".rstrip("0").rstrip(".")


def _line(label: str, value: Any) -> str:
    return f"{label + ':':<28}{value}"


def render_run_terminal(summary: dict[str, Any]) -> str:
    """Render the normal, single-task terminal result."""
    lines = [
        "Final agent result",
        "==================",
        _line("Task", summary.get("task")),
        _line("Outcome", "success" if summary.get("success") else "failure"),
        _line("Run status", summary.get("status")),
        _line("Termination reason", summary.get("termination_reason")),
        _line("Final selection", summary.get("final_selection")),
        _line("Selection verdict", summary.get("selection_verdict") or "—"),
        _line("Baseline retained", _yes_no(summary.get("baseline_retained"))),
        "",
        _line("Candidates evaluated", _count(summary.get("candidate_count"))),
        _line(
            "Fully verified",
            _count(summary.get("fully_verified_candidate_count")),
        ),
        "",
        _line("Model calls", _count(summary.get("model_calls"))),
        _line("Input tokens", _count(summary.get("input_tokens"))),
        _line("Output tokens", _count(summary.get("output_tokens"))),
        _line("Total tokens", _count(summary.get("total_tokens"))),
        "",
        _line("CSim calls", _count(summary.get("csim_calls"))),
        _line("Synthesis calls", _count(summary.get("synthesis_calls"))),
        _line("Co-simulation calls", _count(summary.get("cosim_calls"))),
        _line("Runtime", _duration(summary.get("elapsed_seconds"))),
    ]

    metrics = summary.get("metric_rows")
    if isinstance(metrics, list) and metrics:
        lines.extend(["", "Selected PPA trade-off", "----------------------"])
        for metric in metrics:
            if not isinstance(metric, dict):
                continue
            delta = metric.get("delta_percent")
            delta_text = (
                f" ({float(delta):+.2f}%)"
                if _number(delta) is not None
                else ""
            )
            unit = f" {metric['unit']}" if metric.get("unit") else ""
            value = (
                f"{_metric_value(metric.get('baseline'))} → "
                f"{_metric_value(metric.get('selected'))}{unit}{delta_text}"
            )
            lines.append(_line(str(metric.get("label")), value))

    reason = summary.get("selection_reason")
    if isinstance(reason, str) and reason:
        lines.extend(["", _line("Selection reason", reason)])

    lines.extend(
        [
            "",
            _line(
                "Frequency constraint",
                "passed"
                if summary.get("frequency_compliant") is True
                else "failed"
                if summary.get("frequency_compliant") is False
                else "—",
            ),
            _line(
                "Resource constraints",
                "passed"
                if summary.get("resource_compliant") is True
                else "failed"
                if summary.get("resource_compliant") is False
                else "—",
            ),
            _line(
                "Final design verified",
                _yes_no(summary.get("final_design_verified")),
            ),
        ]
    )
    return "\n".join(lines)


def render_suite_terminal(report: dict[str, Any]) -> str:
    """Render rates and per-run means first, with complete totals second."""
    overall = report.get("overall")
    overall = overall if isinstance(overall, dict) else {}
    runs = int(_number(overall.get("runs")) or 0)
    successful = int(_number(overall.get("successful_runs")) or 0)
    optimised = int(_number(overall.get("optimised_candidate_selected")) or 0)
    retained = int(_number(overall.get("baseline_retained")) or 0)

    lines = [
        "Suite results",
        "=============",
        _line("Runs", _count(runs)),
        _line("Successful final designs", f"{successful:,} / {runs:,}"),
        _line("Optimised selections", f"{optimised:,} / {runs:,}"),
        _line("Baselines retained", f"{retained:,} / {runs:,}"),
        "",
        "Mean per run",
        "============",
        _line("Candidates evaluated", _mean(overall.get("mean_candidate_count"))),
        _line(
            "Fully verified candidates",
            _mean(overall.get("mean_fully_verified_candidate_count")),
        ),
        _line("Model calls", _mean(overall.get("mean_budget_model_calls_used"))),
        _line("Input tokens", _mean(overall.get("mean_input_tokens"))),
        _line("Output tokens", _mean(overall.get("mean_output_tokens"))),
        _line("Total tokens", _mean(overall.get("mean_total_tokens"))),
        _line("CSim calls", _mean(overall.get("mean_budget_csim_calls_used"))),
        _line(
            "Synthesis calls",
            _mean(overall.get("mean_budget_synthesis_calls_used")),
        ),
        _line(
            "Co-simulation calls",
            _mean(overall.get("mean_budget_cosim_calls_used")),
        ),
        _line("Runtime", _duration(overall.get("mean_elapsed_seconds"))),
        "",
        "Complete suite totals",
        "=====================",
        _line("Candidates evaluated", _count(overall.get("total_candidate_count"))),
        _line(
            "Fully verified",
            _count(overall.get("total_fully_verified_candidate_count")),
        ),
        _line("Model calls", _count(overall.get("total_budget_model_calls_used"))),
        _line("Input tokens", _count(overall.get("total_input_tokens"))),
        _line("Output tokens", _count(overall.get("total_output_tokens"))),
        _line("Total tokens", _count(overall.get("total_total_tokens"))),
        _line("CSim calls", _count(overall.get("total_budget_csim_calls_used"))),
        _line(
            "Synthesis calls",
            _count(overall.get("total_budget_synthesis_calls_used")),
        ),
        _line(
            "Co-simulation calls",
            _count(overall.get("total_budget_cosim_calls_used")),
        ),
        _line("Runtime", _duration(overall.get("total_elapsed_seconds"))),
    ]
    return "\n".join(lines)
