"""Resource-limit and timing-balance recovery for PPA optimisation."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from agent.optimise.config_source import ConfigSource

REPO_ROOT = Path(__file__).resolve().parents[2]
RESOURCE_RECOVERY_REASONS = {
    "resource_limit_recovery_from_feasible_pareto",
    "resource_limit_recovery_from_feasible_verified",
}
RESOURCE_FREQUENCY_BALANCE_REASON = (
    "resource_frequency_balance_from_feasible_parent"
)


def _load_config(config_source: ConfigSource) -> dict[str, Any]:
    if not config_source.is_file():
        raise FileNotFoundError(f"Config not found: {config_source}")
    value = json.loads(config_source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("PPA config must contain a JSON object")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected in {path}")
    return value


def _load_optional_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def _indexed_records(
    records: Iterable[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    return {
        record["candidate_index"]: record
        for record in records
        if isinstance(record.get("candidate_index"), int)
    }


def _candidate_output_dir(record: dict[str, Any]) -> Path | None:
    value = record.get("candidate_file")
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.parent


def _candidate_strategy(record: dict[str, Any]) -> dict[str, Any]:
    index = record.get("candidate_index")
    output_dir = _candidate_output_dir(record)
    if not isinstance(index, int) or output_dir is None:
        return {}
    return _load_optional_json(
        output_dir / f"candidate_{index:03d}_strategy.json"
    )


def _candidate_duplicate_report(record: dict[str, Any]) -> dict[str, Any]:
    index = record.get("candidate_index")
    output_dir = _candidate_output_dir(record)
    if not isinstance(index, int) or output_dir is None:
        return {}
    return _load_optional_json(
        output_dir / f"candidate_{index:03d}_duplicate_check.json"
    )


def _duplicate_of(record: dict[str, Any]) -> int | None:
    direct = record.get("duplicate_of")
    if isinstance(direct, int):
        return direct
    report = _candidate_duplicate_report(record)
    value = report.get("duplicate_of")
    return value if report.get("passed") is False and isinstance(value, int) else None


def _is_duplicate(record: dict[str, Any]) -> bool:
    if record.get("verdict") == "reject_duplicate":
        return True
    report = _candidate_duplicate_report(record)
    return report.get("passed") is False and isinstance(
        report.get("duplicate_of"),
        int,
    )


def _resolve_latest_non_duplicate(
    indexed: dict[int, dict[str, Any]],
) -> dict[str, Any] | None:
    if not indexed:
        return None
    current = indexed[max(indexed)]
    visited: set[int] = set()
    while _is_duplicate(current):
        duplicate_of = _duplicate_of(current)
        if not isinstance(duplicate_of, int) or duplicate_of in visited:
            return None
        visited.add(duplicate_of)
        linked = indexed.get(duplicate_of)
        if linked is None:
            return None
        current = linked
    return current


def _is_feasible_parent(record: dict[str, Any]) -> bool:
    compliance = record.get("resource_limit_compliance")
    return bool(
        record.get("fully_verified") is True
        and record.get("meets_frequency_requirement") is True
        and isinstance(compliance, dict)
        and compliance.get("passed") is True
    )


def _validated_balance_evidence(
    indexed: dict[int, dict[str, Any]],
    *,
    parent_index: Any,
    resource_rejected_index: Any,
    frequency_rejected_index: Any,
) -> dict[str, Any] | None:
    if not all(
        isinstance(value, int)
        for value in (
            parent_index,
            resource_rejected_index,
            frequency_rejected_index,
        )
    ):
        return None

    parent = indexed.get(parent_index)
    resource_rejected = indexed.get(resource_rejected_index)
    frequency_rejected = indexed.get(frequency_rejected_index)
    if parent is None or not _is_feasible_parent(parent):
        return None
    if (
        resource_rejected is None
        or resource_rejected.get("verdict") != "reject_resource_limits"
    ):
        return None
    compliance = resource_rejected.get("resource_limit_compliance")
    if not isinstance(compliance, dict) or not compliance.get("violations"):
        return None
    if (
        frequency_rejected is None
        or frequency_rejected.get("verdict") != "reject_frequency_threshold"
    ):
        return None

    return {
        "parent": parent,
        "resource_rejected": resource_rejected,
        "frequency_rejected": frequency_rejected,
    }


def resource_limit_recovery_trigger(
    records: Iterable[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return the latest direct or duplicate-linked resource-limit rejection."""
    indexed = _indexed_records(records)
    current = _resolve_latest_non_duplicate(indexed)
    if current is None or current.get("verdict") != "reject_resource_limits":
        return None
    compliance = current.get("resource_limit_compliance")
    if not isinstance(compliance, dict) or not compliance.get("violations"):
        return None
    return current


def resource_frequency_balance_trigger(
    records: Iterable[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return both failed boundaries, preserving them across duplicate retries."""
    indexed = _indexed_records(records)
    if not indexed:
        return None

    latest = indexed[max(indexed)]
    latest_strategy = _candidate_strategy(latest)

    # A balanced-recovery model call may return the feasible parent unchanged.
    # Preserve the original resource and frequency boundaries rather than losing
    # the recovery lineage when the evaluator has not yet labelled the duplicate.
    if (
        _is_duplicate(latest)
        and latest_strategy.get("name")
        == "recover_resource_frequency_balance"
    ):
        parameters = latest_strategy.get("parameters") or {}
        evidence = _validated_balance_evidence(
            indexed,
            parent_index=latest_strategy.get("source_candidate_index"),
            resource_rejected_index=parameters.get(
                "resource_rejected_candidate_index"
            ),
            frequency_rejected_index=parameters.get(
                "frequency_rejected_candidate_index"
            ),
        )
        if evidence is None:
            return None
        return {
            **evidence,
            "source_strategy": latest_strategy,
            "duplicate_escape": {
                "candidate_index": latest.get("candidate_index"),
                "duplicate_of_candidate_index": _duplicate_of(latest),
            },
        }

    frequency_rejected = _resolve_latest_non_duplicate(indexed)
    if (
        frequency_rejected is None
        or frequency_rejected.get("verdict") != "reject_frequency_threshold"
    ):
        return None

    strategy = _candidate_strategy(frequency_rejected)
    if strategy.get("name") != "recover_resource_limits":
        return None

    parameters = strategy.get("parameters") or {}
    evidence = _validated_balance_evidence(
        indexed,
        parent_index=strategy.get("source_candidate_index"),
        resource_rejected_index=parameters.get("rejected_candidate_index"),
        frequency_rejected_index=frequency_rejected.get("candidate_index"),
    )
    if evidence is None:
        return None
    return {
        **evidence,
        "source_strategy": strategy,
        "duplicate_escape": None,
    }


def is_resource_recovery_reason(reason: str) -> bool:
    return reason in RESOURCE_RECOVERY_REASONS


def is_resource_frequency_balance_reason(reason: str) -> bool:
    return reason == RESOURCE_FREQUENCY_BALANCE_REASON


def _normalised_violations(record: dict[str, Any]) -> list[dict[str, Any]]:
    compliance = record.get("resource_limit_compliance")
    raw = compliance.get("violations") if isinstance(compliance, dict) else None
    result: list[dict[str, Any]] = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        actual = item.get("actual")
        limit = item.get("limit")
        excess = item.get("excess")
        if (
            not isinstance(excess, (int, float))
            and isinstance(actual, (int, float))
            and isinstance(limit, (int, float))
        ):
            excess = float(actual) - float(limit)
        result.append(
            {
                "metric": item.get("metric"),
                "actual": actual,
                "limit": limit,
                "excess": excess,
                "reason": item.get("reason", "limit_exceeded"),
            }
        )
    return result


def _metric_lines(metrics: dict[str, Any]) -> str:
    keys = (
        "frequency_mhz",
        "clock_period_ns",
        "latency_ns",
        "throughput_period_ns",
        "resources_lut_used",
        "resources_ff_used",
        "resources_dsp_used",
        "resources_bram_used",
    )
    return "\n".join(f"- {key}: {metrics.get(key)}" for key in keys)


def _resource_limit_lines(limits: dict[str, Any]) -> str:
    return "\n".join(
        f"- {key}: {value}" for key, value in sorted(limits.items())
    ) or "- No resource limits were configured."


def _common_inputs(
    config_source: ConfigSource,
    parent_index: int,
) -> tuple[
    dict[str, Any],
    Path,
    Path,
    Path,
    dict[str, Any],
    dict[str, Any],
    dict[int, dict[str, Any]],
]:
    config = _load_config(config_source.resolve())
    output_dir = REPO_ROOT / config["output_dir"]
    baseline_path = REPO_ROOT / config["baseline"]["source"]
    parent_path = output_dir / f"candidate_{parent_index:03d}.cpp"
    summary_path = output_dir / "experiment_summary.json"
    target_path = output_dir / "baseline_source_target.json"
    cause_path = output_dir / "baseline_source_cause.json"
    for path in (
        baseline_path,
        parent_path,
        summary_path,
        target_path,
        cause_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Required recovery input not found: {path}")

    summary = _load_json(summary_path)
    records = {
        item["candidate_index"]: item
        for item in summary.get("candidates", [])
        if isinstance(item, dict)
        and isinstance(item.get("candidate_index"), int)
    }
    return (
        config,
        output_dir,
        baseline_path,
        parent_path,
        _load_json(target_path),
        _load_json(cause_path),
        records,
    )


def prepare_resource_recovery_prompt(
    config_source: ConfigSource,
    parent_index: int,
    rejected_index: int,
    next_index: int,
) -> Path:
    """Refine a feasible parent using exact evidence from an over-budget design."""
    if next_index <= max(parent_index, rejected_index):
        raise ValueError(
            "next_index must be newer than the parent and rejected candidate"
        )

    (
        config,
        output_dir,
        baseline_path,
        parent_path,
        target,
        cause,
        records,
    ) = _common_inputs(config_source, parent_index)
    parent = records.get(parent_index)
    rejected = records.get(rejected_index)
    if parent is None or rejected is None:
        raise ValueError(
            "Resource-recovery parent or trigger is missing from the summary"
        )
    if rejected.get("verdict") != "reject_resource_limits":
        raise ValueError(
            "Resource-recovery trigger must have reject_resource_limits verdict"
        )

    violations = _normalised_violations(rejected)
    if not violations:
        raise ValueError(
            "Resource-recovery trigger has no concrete resource violations"
        )

    violation_lines = "\n".join(
        (
            f"- {item['metric']}: actual {item['actual']}, "
            f"limit {item['limit']}, excess {item['excess']}"
        )
        for item in violations
    )
    required_changes = [
        "Continue from the fully verified, resource-compliant parent rather than the rejected design.",
        "Reduce parallelism in the transformed region using a smaller bounded UNROLL factor or less aggressive replication.",
        "Remove or relax a PIPELINE placement when it implies complete unrolling or excessive operator replication.",
        "Make one focused structural change and keep every configured resource metric within its limit.",
    ]
    forbidden_changes = [
        "Do not reproduce the rejected candidate or its exact directive placement and loop transformation combination.",
        "Do not completely unroll a loop merely to pursue II=1.",
        "Do not completely partition top-level interface arrays.",
        "Do not sacrifice functional correctness, interface compatibility, or bounds safety.",
    ]
    strategy = {
        "name": "recover_resource_limits",
        "parameters": {
            "rejected_candidate_index": rejected_index,
            "violations": violations,
        },
        "reason": (
            "The rejected candidate improved performance but exceeded one or more "
            "target-device resource ceilings."
        ),
        "required_changes": required_changes,
        "forbidden_changes": forbidden_changes,
        "source_candidate_index": parent_index,
        "next_candidate_index": next_index,
        "trigger": "resource_limit_violation",
    }

    configured_constraints = config.get("prompt_constraints") or []
    constraint_lines = "\n".join(
        f"- {item}"
        for item in configured_constraints
        if isinstance(item, str)
    ) or "- Preserve all task-manifest contracts."
    top = config["top_function"]
    prompt = f"""You are performing iteration {next_index} of an AMD/Xilinx Vitis HLS PPA optimisation loop.

Benchmark: {config.get('benchmark')}
Top function: {top}

Candidate {rejected_index:03d} was rejected after synthesis because it exceeded the configured target-device resource limits. Do not use it as the source architecture. Continue from candidate {parent_index:03d}, which is fully verified and resource compliant.

Exact rejected-candidate violations:
{violation_lines}

Feasible parent metrics:
{_metric_lines(parent.get('metrics') or {})}

Rejected candidate metrics:
{_metric_lines(rejected.get('metrics') or {})}

Selected target:
- Function/report: {target.get('target_name')}
- Loop label: {target.get('loop_label')}
- Primary cause: {(cause.get('primary_hypothesis') or {}).get('category')}
- Interpretation: {(cause.get('primary_hypothesis') or {}).get('interpretation')}

Required recovery changes:
{chr(10).join(f'- {item}' for item in required_changes)}

Forbidden changes:
{chr(10).join(f'- {item}' for item in forbidden_changes)}

Task-specific contracts:
{constraint_lines}

Constraints:
1. Preserve the exact {top} signature and algorithmic behaviour.
2. Preserve required loop labels and the existing HLS interface contract.
3. Keep all array accesses in bounds and process every required element exactly once.
4. Return one complete compilable C++ source file only.
5. Do not include Markdown fences or explanations.

Feasible parent source to modify:
{parent_path.read_text(encoding='utf-8')}

Original baseline source:
{baseline_path.read_text(encoding='utf-8')}
"""

    prompt_path = output_dir / f"candidate_{next_index:03d}_prompt.txt"
    feedback_path = (
        output_dir
        / f"candidate_{next_index:03d}_resource_recovery_feedback.json"
    )
    strategy_path = output_dir / f"candidate_{next_index:03d}_strategy.json"
    prompt_path.write_text(prompt, encoding="utf-8")
    strategy_path.write_text(
        json.dumps(strategy, indent=2) + "\n",
        encoding="utf-8",
    )
    feedback_path.write_text(
        json.dumps(
            {
                "parent_candidate_index": parent_index,
                "rejected_candidate_index": rejected_index,
                "next_candidate_index": next_index,
                "verdict": "recover_resource_limits",
                "violations": violations,
                "required_changes": required_changes,
                "forbidden_changes": forbidden_changes,
                "prompt_file": str(prompt_path.relative_to(REPO_ROOT)),
                "strategy_file": str(strategy_path.relative_to(REPO_ROOT)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return prompt_path


def _latest_balanced_duplicate(
    records: dict[int, dict[str, Any]],
    next_index: int,
) -> dict[str, Any] | None:
    previous = [index for index in records if index < next_index]
    if not previous:
        return None
    latest = records[max(previous)]
    if not _is_duplicate(latest):
        return None
    strategy = _candidate_strategy(latest)
    if strategy.get("name") != "recover_resource_frequency_balance":
        return None
    return {
        "candidate_index": latest.get("candidate_index"),
        "duplicate_of_candidate_index": _duplicate_of(latest),
    }


def prepare_resource_frequency_balance_prompt(
    config_source: ConfigSource,
    parent_index: int,
    resource_rejected_index: int,
    frequency_rejected_index: int,
    next_index: int,
) -> Path:
    """Search between an over-parallel and an under-timed failed design."""
    if next_index <= max(
        parent_index,
        resource_rejected_index,
        frequency_rejected_index,
    ):
        raise ValueError("next_index must be newer than all recovery evidence")

    (
        config,
        output_dir,
        baseline_path,
        parent_path,
        target,
        cause,
        records,
    ) = _common_inputs(config_source, parent_index)
    parent = records.get(parent_index)
    resource_rejected = records.get(resource_rejected_index)
    frequency_rejected = records.get(frequency_rejected_index)
    if parent is None or resource_rejected is None or frequency_rejected is None:
        raise ValueError("Balanced-recovery evidence is missing from the summary")
    if resource_rejected.get("verdict") != "reject_resource_limits":
        raise ValueError(
            "Resource boundary must have reject_resource_limits verdict"
        )
    if frequency_rejected.get("verdict") != "reject_frequency_threshold":
        raise ValueError(
            "Timing boundary must have reject_frequency_threshold verdict"
        )

    violations = _normalised_violations(resource_rejected)
    if not violations:
        raise ValueError("Resource boundary has no concrete resource violations")

    minimum_frequency_mhz = float(config.get("minimum_frequency_mhz", 0.0))
    maximum_clock_period_ns = float(
        config.get(
            "target_clock_period_ns",
            1000.0 / minimum_frequency_mhz
            if minimum_frequency_mhz > 0
            else 0.0,
        )
    )
    timing_metrics = frequency_rejected.get("metrics") or {}
    actual_frequency_mhz = timing_metrics.get("frequency_mhz")
    actual_clock_period_ns = timing_metrics.get("clock_period_ns")
    if not isinstance(actual_clock_period_ns, (int, float)) and isinstance(
        actual_frequency_mhz,
        (int, float),
    ) and actual_frequency_mhz > 0:
        actual_clock_period_ns = 1000.0 / float(actual_frequency_mhz)
    frequency_shortfall_mhz = (
        minimum_frequency_mhz - float(actual_frequency_mhz)
        if isinstance(actual_frequency_mhz, (int, float))
        else None
    )
    clock_period_excess_ns = (
        float(actual_clock_period_ns) - maximum_clock_period_ns
        if isinstance(actual_clock_period_ns, (int, float))
        else None
    )

    resource_violation_lines = "\n".join(
        (
            f"- {item['metric']}: actual {item['actual']}, "
            f"limit {item['limit']}, excess {item['excess']}"
        )
        for item in violations
    )
    resource_limits = dict(config.get("resource_limits") or {})
    parent_latency_ns = (parent.get("metrics") or {}).get("latency_ns")
    duplicate_escape = _latest_balanced_duplicate(records, next_index)

    required_changes = [
        "Continue from the fully verified parent; do not refine either failed boundary directly.",
        (
            f"Treat frequency >= {minimum_frequency_mhz} MHz and clock period <= "
            f"{maximum_clock_period_ns} ns as hard constraints, not secondary goals."
        ),
        "Keep every configured resource metric within its target-device limit.",
        (
            f"Seek latency below the feasible parent's {parent_latency_ns} ns using a "
            "middle-ground architecture."
        ),
        "Prefer moderate partial unrolling, local loop pipelining, or a bounded local-memory transformation in one critical region.",
        "Preserve register cuts and a short critical path; do not trade timing away merely to achieve II=1.",
    ]
    forbidden_changes = [
        "Do not reproduce the over-parallel resource-rejected design or its complete-unroll structure.",
        "Do not reproduce the timing-rejected recovery design or its exact directive placement and loop rewrite.",
        "Do not flatten or pipeline the entire loop nest into one long combinational datapath.",
        "Do not force II=1 when the resulting clock period exceeds the hard timing limit.",
        "Do not completely partition top-level interface arrays.",
        "Do not sacrifice functional correctness, interface compatibility, or bounds safety.",
    ]
    duplicate_section = ""
    if duplicate_escape:
        duplicate_index = duplicate_escape["candidate_index"]
        duplicate_of = duplicate_escape["duplicate_of_candidate_index"]
        required_changes.append(
            (
                f"Candidate {duplicate_index:03d} duplicated candidate "
                f"{duplicate_of:03d}; this retry must introduce at least one "
                "hardware-relevant structural change in the selected target region."
            )
        )
        forbidden_changes.append(
            "Do not return the feasible parent verbatim or with only comments, whitespace, identifier renaming, or algebraically neutral edits."
        )
        duplicate_section = f"""

Previous balanced-recovery attempt:
- Candidate {duplicate_index:03d} duplicated candidate {duplicate_of:03d} and was rejected before CSim and synthesis.
- Change at least one hardware-relevant mechanism: a bounded UNROLL factor, local PIPELINE placement, or bounded local-memory/banking structure in the selected target region.
- A comment-only, formatting-only, renaming-only, or no-op arithmetic edit is not acceptable.
"""

    strategy_parameters: dict[str, Any] = {
        "resource_rejected_candidate_index": resource_rejected_index,
        "frequency_rejected_candidate_index": frequency_rejected_index,
        "minimum_frequency_mhz": minimum_frequency_mhz,
        "maximum_clock_period_ns": maximum_clock_period_ns,
        "frequency_shortfall_mhz": frequency_shortfall_mhz,
        "clock_period_excess_ns": clock_period_excess_ns,
        "resource_limits": resource_limits,
        "resource_violations": violations,
    }
    if duplicate_escape:
        strategy_parameters.update(
            duplicate_retry_of_candidate_index=duplicate_escape[
                "candidate_index"
            ],
            duplicate_of_candidate_index=duplicate_escape[
                "duplicate_of_candidate_index"
            ],
        )
    strategy = {
        "name": "recover_resource_frequency_balance",
        "parameters": strategy_parameters,
        "reason": (
            "The first failed boundary used excessive parallelism and exceeded device "
            "resources; the recovery then reduced resources but collapsed timing."
        ),
        "required_changes": required_changes,
        "forbidden_changes": forbidden_changes,
        "source_candidate_index": parent_index,
        "next_candidate_index": next_index,
        "trigger": (
            "resource_frequency_balance_duplicate_escape"
            if duplicate_escape
            else "resource_recovery_frequency_failure"
        ),
    }

    configured_constraints = config.get("prompt_constraints") or []
    constraint_lines = "\n".join(
        f"- {item}"
        for item in configured_constraints
        if isinstance(item, str)
    ) or "- Preserve all task-manifest contracts."
    top = config["top_function"]
    prompt = f"""You are performing iteration {next_index} of an AMD/Xilinx Vitis HLS PPA optimisation loop.

Benchmark: {config.get('benchmark')}
Top function: {top}

Use candidate {parent_index:03d} as the only source architecture. It is fully verified and satisfies both timing and resource constraints.

Two failed boundaries define the search direction:

1. Candidate {resource_rejected_index:03d} was over-parallel and exceeded resource limits.
Exact resource violations:
{resource_violation_lines}

2. Candidate {frequency_rejected_index:03d} reduced resources but failed timing.
- Estimated frequency: {actual_frequency_mhz} MHz
- Required minimum frequency: {minimum_frequency_mhz} MHz
- Frequency shortfall: {frequency_shortfall_mhz} MHz
- Estimated clock period: {actual_clock_period_ns} ns
- Maximum clock period: {maximum_clock_period_ns} ns
- Clock-period excess: {clock_period_excess_ns} ns
- Achieved latency: {timing_metrics.get('latency_ns')} ns{duplicate_section}

Hard target-device resource limits:
{_resource_limit_lines(resource_limits)}

Feasible parent metrics:
{_metric_lines(parent.get('metrics') or {})}

Over-parallel boundary metrics:
{_metric_lines(resource_rejected.get('metrics') or {})}

Timing-rejected boundary metrics:
{_metric_lines(timing_metrics)}

Selected target:
- Function/report: {target.get('target_name')}
- Loop label: {target.get('loop_label')}
- Primary cause: {(cause.get('primary_hypothesis') or {}).get('category')}
- Interpretation: {(cause.get('primary_hypothesis') or {}).get('interpretation')}

Required balanced-recovery changes:
{chr(10).join(f'- {item}' for item in required_changes)}

Forbidden changes:
{chr(10).join(f'- {item}' for item in forbidden_changes)}

Task-specific contracts:
{constraint_lines}

Constraints:
1. Preserve the exact {top} signature and algorithmic behaviour.
2. Preserve required loop labels and the existing HLS interface contract.
3. Keep all array accesses in bounds and process every required element exactly once.
4. Return one complete compilable C++ source file only.
5. Do not include Markdown fences or explanations.

Feasible parent source to modify:
{parent_path.read_text(encoding='utf-8')}

Original baseline source:
{baseline_path.read_text(encoding='utf-8')}
"""

    prompt_path = output_dir / f"candidate_{next_index:03d}_prompt.txt"
    feedback_path = (
        output_dir
        / f"candidate_{next_index:03d}_resource_frequency_balance_feedback.json"
    )
    strategy_path = output_dir / f"candidate_{next_index:03d}_strategy.json"
    prompt_path.write_text(prompt, encoding="utf-8")
    strategy_path.write_text(
        json.dumps(strategy, indent=2) + "\n",
        encoding="utf-8",
    )
    feedback_path.write_text(
        json.dumps(
            {
                "parent_candidate_index": parent_index,
                "resource_rejected_candidate_index": resource_rejected_index,
                "frequency_rejected_candidate_index": frequency_rejected_index,
                "next_candidate_index": next_index,
                "verdict": "recover_resource_frequency_balance",
                "minimum_frequency_mhz": minimum_frequency_mhz,
                "maximum_clock_period_ns": maximum_clock_period_ns,
                "frequency_shortfall_mhz": frequency_shortfall_mhz,
                "clock_period_excess_ns": clock_period_excess_ns,
                "resource_limits": resource_limits,
                "resource_violations": violations,
                "duplicate_escape": duplicate_escape,
                "required_changes": required_changes,
                "forbidden_changes": forbidden_changes,
                "prompt_file": str(prompt_path.relative_to(REPO_ROOT)),
                "strategy_file": str(strategy_path.relative_to(REPO_ROOT)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return prompt_path
