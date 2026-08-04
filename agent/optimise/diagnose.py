"""Diagnosis helpers and evidence-driven PPA refinement prompt construction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agent.analysis.hierarchical_hls_analyzer import analyse_hierarchy
from agent.optimise.refinement_strategy import select_tradeoff_strategy

REPO_ROOT = Path(__file__).resolve().parents[2]
METRIC_KEYS = (
    "clock_period_ns", "latency_best_cycles", "latency_average_cycles",
    "latency_worst_cycles", "interval_min_cycles", "interval_max_cycles",
    "resources_lut_used", "resources_ff_used", "resources_dsp_used",
    "resources_bram_used",
)


def diagnose_reports(report_root: Path, *, interface_frozen: bool = False) -> dict[str, Any]:
    return analyse_hierarchy(report_root, interface_frozen=interface_frozen)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_optional(path: Path) -> dict[str, Any] | None:
    return load_json(path) if path.is_file() else None


def metric_summary(metrics: dict[str, Any]) -> str:
    return "\n".join(f"- {key}: {metrics.get(key)}" for key in METRIC_KEYS)


def _failure_evidence(report: dict[str, Any] | None) -> str:
    if not report:
        return "- No detailed report was produced."
    evidence = report.get("evidence") or report.get("bounds_issues") or []
    if isinstance(evidence, list) and evidence:
        return "\n".join(f"- {item}" for item in evidence[-12:])
    log = report.get("log_file")
    return f"- See tool log: {log}" if log else "- The stage returned a failed result."


def _tool_log(report: dict[str, Any]) -> str:
    value = report.get("log_file") or report.get("log_path")
    if not value:
        return ""
    path = Path(str(value))
    if not path.is_absolute():
        path = REPO_ROOT / path
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _completed_partial_unroll_factors(
    output_dir: Path,
    summary: dict[str, Any],
) -> list[int]:
    factors: list[int] = []
    for record in summary.get("candidates", []):
        if record.get("fully_verified") is not True:
            continue
        index = record.get("candidate_index")
        if not isinstance(index, int):
            continue
        strategy = load_optional(output_dir / f"candidate_{index:03d}_strategy.json")
        if not strategy or strategy.get("name") != "partial_unroll":
            continue
        factor = strategy.get("parameters", {}).get("factor")
        if isinstance(factor, int) and factor not in factors:
            factors.append(factor)
    return factors


def _strategy_text(strategy: dict[str, Any]) -> str:
    factor = strategy["parameters"]["factor"]
    return (
        f"Selected strategy: partial loop unrolling with factor {factor}.\n"
        f"Reason: {strategy['reason']}\n"
        "Required changes:\n"
        + "\n".join(f"- {item}" for item in strategy["required_changes"])
        + "\nForbidden changes:\n"
        + "\n".join(f"- {item}" for item in strategy["forbidden_changes"])
    )


def _write_strategy(
    output_dir: Path,
    *,
    source_index: int,
    next_index: int,
    strategy: dict[str, Any],
    trigger: str,
    retry_of: int | None = None,
) -> None:
    payload = {
        "name": strategy["name"],
        "parameters": strategy["parameters"],
        "reason": strategy["reason"],
        "required_changes": strategy["required_changes"],
        "forbidden_changes": strategy["forbidden_changes"],
        "source_candidate_index": source_index,
        "next_candidate_index": next_index,
        "trigger": trigger,
    }
    if retry_of is not None:
        payload["retry_of_candidate_index"] = retry_of
    (output_dir / f"candidate_{next_index:03d}_strategy.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def prepare_refinement_prompt(config_path: Path, previous_index: int, next_index: int) -> Path:
    if next_index <= previous_index:
        raise ValueError("next_index must be greater than previous_index")
    config = load_json(config_path.resolve())
    output_dir = REPO_ROOT / config["output_dir"]
    baseline_path = REPO_ROOT / config["baseline"]["source"]
    previous_path = output_dir / f"candidate_{previous_index:03d}.cpp"
    target_path = output_dir / "baseline_source_target.json"
    cause_path = output_dir / "baseline_source_cause.json"
    for path in (baseline_path, previous_path, target_path, cause_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required refinement input not found: {path}")

    static = load_optional(output_dir / f"candidate_{previous_index:03d}_static_validation.json")
    duplicate = load_optional(output_dir / f"candidate_{previous_index:03d}_duplicate_check.json")
    csim = load_optional(output_dir / f"candidate_{previous_index:03d}_csim_validation.json")
    synthesis = load_optional(output_dir / f"candidate_{previous_index:03d}_synthesis.json")
    summary = load_optional(output_dir / "experiment_summary.json") or {}
    record = next(
        (item for item in summary.get("candidates", []) if item.get("candidate_index") == previous_index),
        {},
    )
    verdict = str(record.get("verdict") or "incomplete")

    if static and static.get("passed") is False:
        failed = [name for name, passed in (static.get("checks") or {}).items() if not passed]
        measured = "Static validation failed before expensive tools.\n- Failed checks: " + ", ".join(failed)
        measured += "\n" + _failure_evidence(static)
        direction = "repair the rejected source structure and remove unsafe or conflicting pragmas"
    elif duplicate and duplicate.get("passed") is False:
        measured = f"The source duplicates candidate {duplicate.get('duplicate_of')}."
        direction = "produce a materially different implementation rather than a formatting or comment change"
    elif csim and csim.get("passed") is False:
        measured = "Static validation passed, but Vitis CSim failed.\n" + _failure_evidence(csim)
        direction = "repair functional, compile, interface or bounds correctness before attempting PPA optimisation"
    elif synthesis and synthesis.get("timed_out") is True:
        measured = (
            f"CSim passed, but synthesis exceeded the configured {synthesis.get('timeout_seconds')} second timeout.\n"
            + _failure_evidence(synthesis)
        )
        direction = "reduce compiler and hardware expansion; avoid complete partitioning, excessive unrolling, conflicting dataflow, or other explosive transformations"
    elif synthesis and synthesis.get("passed") is False:
        measured = "CSim passed, but Vitis synthesis failed.\n" + _failure_evidence(synthesis)
        direction = "repair synthesizability while preserving the verified functional behaviour"
    elif synthesis and synthesis.get("passed") is True:
        measured = (
            "Static validation, CSim, and synthesis passed, but the candidate did not provide the required PPA result.\n\n"
            f"Baseline metrics:\n{metric_summary(summary.get('baseline_metrics') or {})}\n\n"
            f"Candidate metrics:\n{metric_summary(synthesis.get('metrics') or {})}"
        )
        direction = "apply a genuinely different structural transformation rather than a pragma-only change"
    else:
        measured = f"Candidate evaluation is incomplete. Current verdict: {verdict}."
        direction = "produce a conservative, synthesizable structural alternative"

    previous_strategy = load_optional(
        output_dir / f"candidate_{previous_index:03d}_strategy.json"
    )
    strategy: dict[str, Any] | None = None
    strategy_trigger: str | None = None
    retry_of: int | None = None
    if previous_strategy:
        if record.get("fully_verified") is True:
            completed = _completed_partial_unroll_factors(output_dir, summary)
            strategy = select_tradeoff_strategy("", completed)
            strategy_trigger = "partial_unroll_ladder" if strategy else None
        else:
            strategy = previous_strategy
            strategy_trigger = "retry_unfinished_strategy"
            retry_of = previous_index

    strategy_section = ""
    if strategy:
        factor = strategy["parameters"]["factor"]
        direction += f"; implement and preserve partial loop unrolling with factor {factor}"
        strategy_section = "\n\n" + _strategy_text(strategy)
        _write_strategy(
            output_dir,
            source_index=previous_index,
            next_index=next_index,
            strategy=strategy,
            trigger=strategy_trigger or "strategy_refinement",
            retry_of=retry_of,
        )

    target = load_json(target_path)
    cause = load_json(cause_path)
    top = config["top_function"]
    prompt = f"""You are performing iteration {next_index} of an AMD/Xilinx Vitis HLS PPA optimisation loop.

Benchmark: {config.get('benchmark')}
Top function: {top}
Objective: improve latency or initiation interval while keeping resource growth proportionate. Correctness is mandatory.

Measured evidence from candidate {previous_index:03d}:
{measured}

Selected target:
- Function/report: {target.get('target_name')}
- Loop label: {target.get('loop_label')}
- Primary cause: {(cause.get('primary_hypothesis') or {}).get('category')}
- Interpretation: {(cause.get('primary_hypothesis') or {}).get('interpretation')}{strategy_section}

Required direction:
- {direction}.
- Keep every input element exactly once and all accesses in bounds.
- Do not repeat an implementation already evaluated.
- Prefer a focused transformation whose expected hardware effect can be explained from the measured evidence.

Constraints:
1. Preserve the exact {top} signature and algorithmic behaviour.
2. Preserve required loop labels and the existing HLS interface contract.
3. Modify only declarations and source regions directly required by the optimisation.
4. Avoid complete partitioning of top-level interface arrays.
5. Do not combine DATAFLOW and PIPELINE in a conflicting single-process region.
6. Return one complete compilable C++ source file only.
7. Do not include Markdown fences or explanations.

Previous candidate source:
{previous_path.read_text(encoding='utf-8')}

Original baseline source:
{baseline_path.read_text(encoding='utf-8')}
"""
    prompt_path = output_dir / f"candidate_{next_index:03d}_prompt.txt"
    feedback_path = output_dir / f"candidate_{previous_index:03d}_feedback.json"
    prompt_path.write_text(prompt, encoding="utf-8")
    feedback_path.write_text(json.dumps({
        "previous_candidate_index": previous_index,
        "next_candidate_index": next_index,
        "verdict": verdict,
        "required_direction": direction,
        "selected_strategy": strategy,
        "prompt_file": str(prompt_path.relative_to(REPO_ROOT)),
    }, indent=2) + "\n", encoding="utf-8")
    print("\nPPA refinement prompt")
    print(f"Previous candidate: {previous_path.relative_to(REPO_ROOT)}")
    print(f"Next prompt: {prompt_path.relative_to(REPO_ROOT)}")
    print(f"Required direction: {direction}")
    return prompt_path


def prepare_tradeoff_prompt(config_path: Path, source_index: int, next_index: int) -> Path:
    if next_index <= source_index:
        raise ValueError("next_index must be greater than source_index")
    config = load_json(config_path.resolve())
    output_dir = REPO_ROOT / config["output_dir"]
    baseline_path = REPO_ROOT / config["baseline"]["source"]
    source_path = output_dir / f"candidate_{source_index:03d}.cpp"
    synthesis = load_json(output_dir / f"candidate_{source_index:03d}_synthesis.json")
    summary = load_json(output_dir / "experiment_summary.json")
    baseline_metrics = summary.get("baseline_metrics") or {}
    source_metrics = synthesis.get("metrics") or {}
    top = config["top_function"]

    completed = _completed_partial_unroll_factors(output_dir, summary)
    strategy = select_tradeoff_strategy(_tool_log(synthesis), completed)
    if strategy:
        factor = strategy["parameters"]["factor"]
        strategy_text = _strategy_text(strategy)
        objective = (
            f"Implement partial loop unrolling with factor {factor} to retain useful "
            f"parallelism while reducing memory pressure and LUT usage relative to "
            f"candidate {source_index:03d}."
        )
        _write_strategy(
            output_dir,
            source_index=source_index,
            next_index=next_index,
            strategy=strategy,
            trigger=(
                "partial_unroll_ladder"
                if completed
                else "memory_port_limited_complete_unroll"
            ),
        )
    else:
        strategy_text = "Selected strategy: lower-cost structural alternative or lower parallelism."
        objective = (
            f"Create a genuinely different implementation that preserves a meaningful "
            f"performance improvement while reducing resource usage relative to candidate "
            f"{source_index:03d}."
        )

    prompt = f"""You are performing iteration {next_index} of an AMD/Xilinx Vitis HLS PPA optimisation loop.

Candidate {source_index:03d} is a valid Pareto point, not a failed optimisation.

Baseline metrics:
{metric_summary(baseline_metrics)}

Pareto candidate metrics:
{metric_summary(source_metrics)}

Objective:
{objective}

{strategy_text}

Required direction:
- Do not reproduce candidate {source_index:03d} verbatim.
- Preserve every input element exactly once and keep all accesses in bounds.
- Avoid complete partitioning of top-level interface arrays and conflicting DATAFLOW/PIPELINE regions.

Constraints:
1. Preserve the exact {top} signature and algorithmic behaviour.
2. Preserve required loop labels and interface directives.
3. Modify only regions directly required by the optimisation.
4. Return one complete compilable C++ source file only, without Markdown fences or explanations.

Pareto candidate source:
{source_path.read_text(encoding='utf-8')}

Original baseline source:
{baseline_path.read_text(encoding='utf-8')}
"""
    prompt_path = output_dir / f"candidate_{next_index:03d}_prompt.txt"
    feedback_path = output_dir / f"candidate_{source_index:03d}_tradeoff_feedback.json"
    prompt_path.write_text(prompt, encoding="utf-8")
    feedback_path.write_text(json.dumps({
        "source_candidate_index": source_index,
        "next_candidate_index": next_index,
        "verdict": "refine_pareto_tradeoff",
        "baseline_metrics": baseline_metrics,
        "source_metrics": source_metrics,
        "completed_partial_unroll_factors": completed,
        "required_direction": objective,
        "selected_strategy": strategy,
        "prompt_file": str(prompt_path.relative_to(REPO_ROOT)),
    }, indent=2) + "\n", encoding="utf-8")
    return prompt_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare evidence-driven PPA refinement prompts.")
    sub = parser.add_subparsers(dest="command", required=True)
    rejected = sub.add_parser("refine")
    rejected.add_argument("config", type=Path)
    rejected.add_argument("--previous-index", type=int, required=True)
    rejected.add_argument("--next-index", type=int, required=True)
    tradeoff = sub.add_parser("tradeoff")
    tradeoff.add_argument("config", type=Path)
    tradeoff.add_argument("--source-index", type=int, required=True)
    tradeoff.add_argument("--next-index", type=int, required=True)
    args = parser.parse_args()
    if args.command == "refine":
        prepare_refinement_prompt(args.config, args.previous_index, args.next_index)
    else:
        prepare_tradeoff_prompt(args.config, args.source_index, args.next_index)


if __name__ == "__main__":
    main()
