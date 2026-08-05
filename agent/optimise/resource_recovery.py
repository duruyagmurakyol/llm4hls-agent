"""Resource-limit recovery selection and prompt construction."""

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


def resource_limit_recovery_trigger(
    records: Iterable[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return the latest direct or duplicate-linked resource-limit rejection."""
    indexed = {
        record["candidate_index"]: record
        for record in records
        if isinstance(record.get("candidate_index"), int)
    }
    if not indexed:
        return None

    current = indexed[max(indexed)]
    visited: set[int] = set()
    while current.get("verdict") == "reject_duplicate":
        duplicate_of = current.get("duplicate_of")
        if not isinstance(duplicate_of, int) or duplicate_of in visited:
            return None
        visited.add(duplicate_of)
        linked = indexed.get(duplicate_of)
        if linked is None:
            return None
        current = linked

    if current.get("verdict") != "reject_resource_limits":
        return None
    compliance = current.get("resource_limit_compliance")
    if not isinstance(compliance, dict) or not compliance.get("violations"):
        return None
    return current


def is_resource_recovery_reason(reason: str) -> bool:
    return reason in RESOURCE_RECOVERY_REASONS


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
        "latency_ns",
        "throughput_period_ns",
        "resources_lut_used",
        "resources_ff_used",
        "resources_dsp_used",
        "resources_bram_used",
    )
    return "\n".join(f"- {key}: {metrics.get(key)}" for key in keys)


def prepare_resource_recovery_prompt(
    config_source: ConfigSource,
    parent_index: int,
    rejected_index: int,
    next_index: int,
) -> Path:
    """Refine a feasible parent using exact evidence from an over-budget design."""
    if next_index <= max(parent_index, rejected_index):
        raise ValueError("next_index must be newer than the parent and rejected candidate")

    config = _load_config(config_source.resolve())
    output_dir = REPO_ROOT / config["output_dir"]
    baseline_path = REPO_ROOT / config["baseline"]["source"]
    parent_path = output_dir / f"candidate_{parent_index:03d}.cpp"
    summary_path = output_dir / "experiment_summary.json"
    target_path = output_dir / "baseline_source_target.json"
    cause_path = output_dir / "baseline_source_cause.json"
    for path in (baseline_path, parent_path, summary_path, target_path, cause_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required resource-recovery input not found: {path}")

    summary = _load_json(summary_path)
    records = {
        item["candidate_index"]: item
        for item in summary.get("candidates", [])
        if isinstance(item, dict) and isinstance(item.get("candidate_index"), int)
    }
    parent = records.get(parent_index)
    rejected = records.get(rejected_index)
    if parent is None or rejected is None:
        raise ValueError("Resource-recovery parent or trigger is missing from the summary")
    if rejected.get("verdict") != "reject_resource_limits":
        raise ValueError("Resource-recovery trigger must have reject_resource_limits verdict")

    violations = _normalised_violations(rejected)
    if not violations:
        raise ValueError("Resource-recovery trigger has no concrete resource violations")

    target = _load_json(target_path)
    cause = _load_json(cause_path)
    violation_lines = "\n".join(
        (
            f"- {item['metric']}: actual {item['actual']}, limit {item['limit']}, "
            f"excess {item['excess']}"
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
        f"- {item}" for item in configured_constraints if isinstance(item, str)
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
    feedback_path = output_dir / f"candidate_{next_index:03d}_resource_recovery_feedback.json"
    strategy_path = output_dir / f"candidate_{next_index:03d}_strategy.json"
    prompt_path.write_text(prompt, encoding="utf-8")
    strategy_path.write_text(json.dumps(strategy, indent=2) + "\n", encoding="utf-8")
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
