"""Diagnosis helpers and evidence-driven PPA refinement prompt construction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agent.analysis.hierarchical_hls_analyzer import analyse_hierarchy

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


def metric_summary(metrics: dict[str, Any]) -> str:
    return "\n".join(f"- {key}: {metrics.get(key)}" for key in METRIC_KEYS)


def prepare_refinement_prompt(
    config_path: Path,
    previous_index: int,
    next_index: int,
) -> Path:
    if next_index <= previous_index:
        raise ValueError("next_index must be greater than previous_index")
    config = load_json(config_path.resolve())
    output_dir = REPO_ROOT / config["output_dir"]
    baseline_path = REPO_ROOT / config["baseline"]["source"]
    previous_path = output_dir / f"candidate_{previous_index:03d}.cpp"
    static_path = output_dir / f"candidate_{previous_index:03d}_static_validation.json"
    target_path = output_dir / "baseline_source_target.json"
    cause_path = output_dir / "baseline_source_cause.json"
    for path in (baseline_path, previous_path, static_path, target_path, cause_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required refinement input not found: {path}")

    static = load_json(static_path)
    target = load_json(target_path)
    cause = load_json(cause_path)
    previous = previous_path.read_text(encoding="utf-8")
    baseline = baseline_path.read_text(encoding="utf-8")
    checks = static.get("checks") or {}
    bounds_failed = checks.get("constant_loop_tail_bounds_safe") is False
    top = config["top_function"]

    if bounds_failed:
        evidence = "\n".join(f"- {item}" for item in static.get("bounds_issues", [])) or "- Static bounds analysis found an unsafe loop tail."
        verdict = "reject_static_bounds_failure"
        direction = "repair the structural optimisation with safe remainder handling"
        measured = f"Static validation failed before CSim or synthesis.\n\nBounds evidence:\n{evidence}"
    else:
        csim_path = output_dir / f"candidate_{previous_index:03d}_csim_validation.json"
        synthesis_path = output_dir / f"candidate_{previous_index:03d}_synthesis.json"
        if not csim_path.is_file() or not synthesis_path.is_file():
            raise FileNotFoundError("Previous candidate passed static validation but CSim/synthesis evidence is missing.")
        csim, synthesis = load_json(csim_path), load_json(synthesis_path)
        if csim.get("passed") is not True or synthesis.get("passed") is not True:
            raise RuntimeError("Previous candidate did not complete CSim and synthesis successfully.")
        summary = load_json(output_dir / "experiment_summary.json")
        measured = (
            "Static validation, CSim, and synthesis passed, but the candidate did not improve top-level PPA.\n\n"
            f"Baseline metrics:\n{metric_summary(summary.get('baseline_metrics') or {})}\n\n"
            f"Candidate metrics:\n{metric_summary(synthesis.get('metrics') or {})}"
        )
        verdict = "reject_no_ppa_improvement"
        direction = "apply a genuinely different structural transformation rather than a pragma-only change"

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
- Interpretation: {(cause.get('primary_hypothesis') or {}).get('interpretation')}

Required direction:
- {direction}.
- Preserve the useful structural intent of the previous candidate where safe.
- Keep every input element exactly once and all accesses in bounds.
- Do not repeat an implementation already evaluated.

Constraints:
1. Preserve the exact {top} signature and algorithmic behaviour.
2. Preserve required loop labels and the existing HLS interface contract.
3. Modify only declarations and source regions directly required by the optimisation.
4. Return one complete compilable C++ source file only.
5. Do not include Markdown fences or explanations.

Previous candidate source:
{previous}

Original baseline source:
{baseline}
"""
    prompt_path = output_dir / f"candidate_{next_index:03d}_prompt.txt"
    feedback_path = output_dir / f"candidate_{previous_index:03d}_feedback.json"
    prompt_path.write_text(prompt, encoding="utf-8")
    feedback_path.write_text(json.dumps({
        "previous_candidate_index": previous_index,
        "next_candidate_index": next_index,
        "verdict": verdict,
        "required_direction": direction,
        "prompt_file": str(prompt_path.relative_to(REPO_ROOT)),
        "bounds_issues": static.get("bounds_issues", []) if bounds_failed else [],
    }, indent=2) + "\n", encoding="utf-8")
    print("\nPPA refinement prompt")
    print(f"Previous candidate: {previous_path.relative_to(REPO_ROOT)}")
    print(f"Next prompt: {prompt_path.relative_to(REPO_ROOT)}")
    print(f"Required direction: {direction}")
    print("No model call, CSim, or synthesis was run.")
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
    prompt = f"""You are performing iteration {next_index} of an AMD/Xilinx Vitis HLS PPA optimisation loop.

Candidate {source_index:03d} is a valid Pareto point, not a failed optimisation.

Baseline metrics:
{metric_summary(baseline_metrics)}

Pareto candidate metrics:
{metric_summary(source_metrics)}

Objective:
Create a genuinely different implementation that preserves a meaningful performance improvement while reducing resource usage relative to candidate {source_index:03d}.

Required direction:
- Do not reproduce candidate {source_index:03d} verbatim.
- Use a lower-cost structural alternative or lower parallelism.
- Preserve every input element exactly once and keep all accesses in bounds.

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
        "required_direction": "retain performance gain with lower resource cost",
        "prompt_file": str(prompt_path.relative_to(REPO_ROOT)),
    }, indent=2) + "\n", encoding="utf-8")
    print("\nPPA trade-off refinement prompt")
    print(f"Source Pareto candidate: {source_path.relative_to(REPO_ROOT)}")
    print(f"Next prompt: {prompt_path.relative_to(REPO_ROOT)}")
    print("Required direction: retain performance gain with lower resource cost")
    print("No model call, CSim, or synthesis was run.")
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
