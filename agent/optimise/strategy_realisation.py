"""Post-synthesis checks that requested HLS strategies reached the hardware flow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CONTROLLED_STRATEGIES = {"partial_unroll", "recover_latency_tradeoff"}
COMPLETE_UNROLL_CONVERSION = "converted into complete unroll"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def _resolve_log(output_dir: Path, synthesis: dict[str, Any]) -> Path | None:
    value = synthesis.get("log_file") or synthesis.get("log_path")
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        repository_root = output_dir
        while repository_root.name != "experiments" and repository_root.parent != repository_root:
            repository_root = repository_root.parent
        if repository_root.name == "experiments":
            repository_root = repository_root.parent
        path = repository_root / path
    return path


def inspect_strategy_realisation(
    output_dir: Path,
    candidate_index: int,
) -> dict[str, Any]:
    """Return whether Vitis preserved a source-enforced bounded strategy."""
    prefix = f"candidate_{candidate_index:03d}"
    strategy = _load_json(output_dir / f"{prefix}_strategy.json")
    name = strategy.get("name")
    parameters = strategy.get("parameters") or {}
    factor = parameters.get("factor")

    if (
        name not in CONTROLLED_STRATEGIES
        or isinstance(factor, bool)
        or not isinstance(factor, int)
        or factor <= 0
    ):
        return {
            "required": False,
            "passed": True,
            "strategy": name,
            "factor": factor,
            "reason": "no_source_enforced_strategy_to_check",
        }

    synthesis = _load_json(output_dir / f"{prefix}_synthesis.json")
    log_path = _resolve_log(output_dir, synthesis)
    if not synthesis or synthesis.get("synthesis_run") is not True or log_path is None:
        return {
            "required": True,
            "passed": None,
            "strategy": name,
            "factor": factor,
            "reason": "awaiting_synthesis_log",
        }
    if not log_path.is_file():
        return {
            "required": True,
            "passed": None,
            "strategy": name,
            "factor": factor,
            "reason": "synthesis_log_unavailable",
            "log_file": str(log_path),
        }

    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    evidence = [
        line.strip()
        for line in lines
        if COMPLETE_UNROLL_CONVERSION in line.lower()
    ]
    return {
        "required": True,
        "passed": not evidence,
        "strategy": name,
        "factor": factor,
        "source_candidate_index": strategy.get("source_candidate_index"),
        "log_file": str(log_path),
        "reason": (
            "partial_unroll_converted_to_complete_unroll"
            if evidence
            else "no_complete_unroll_conversion_detected"
        ),
        "evidence": evidence,
    }


def apply_strategy_realisation(
    output_dir: Path,
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Reject candidates whose requested bounded strategy was not realised."""
    rejected: set[int] = set()
    for record in summary.get("candidates", []):
        if not isinstance(record, dict):
            continue
        index = record.get("candidate_index")
        if not isinstance(index, int):
            continue

        result = inspect_strategy_realisation(output_dir, index)
        if result.get("required") is True:
            record["strategy_realisation"] = result
        if result.get("passed") is not False:
            continue

        rejected.add(index)
        record.update(
            fully_verified=False,
            refinement_eligible=False,
            pareto=False,
            verdict="reject_strategy_not_realised",
            reason=(
                "Vitis converted the requested partial unroll into complete unrolling, "
                "so the bounded recovery factor was not realised in hardware."
            ),
        )

    if rejected:
        summary["pareto_archive"] = [
            item
            for item in summary.get("pareto_archive", [])
            if not (
                isinstance(item, dict)
                and item.get("candidate_index") in rejected
            )
        ]
    return summary
