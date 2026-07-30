#!/usr/bin/env python3

"""Prepare an evidence-driven prompt for the next HLS PPA candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def metric_summary(metrics: dict[str, Any]) -> str:
    ordered = (
        "clock_period_ns",
        "latency_best_cycles",
        "latency_average_cycles",
        "latency_worst_cycles",
        "interval_min_cycles",
        "interval_max_cycles",
        "resources_lut_used",
        "resources_ff_used",
        "resources_dsp_used",
        "resources_bram_used",
    )
    return "\n".join(f"- {key}: {metrics.get(key)}" for key in ordered)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the next PPA prompt from a rejected previous candidate."
    )
    parser.add_argument("config", type=Path, help="PPA optimisation JSON config")
    parser.add_argument("--previous-index", type=int, default=1)
    parser.add_argument("--next-index", type=int, default=2)
    args = parser.parse_args()

    if args.next_index <= args.previous_index:
        raise ValueError("--next-index must be greater than --previous-index")

    config = load_json(args.config.resolve())
    output_dir = REPO_ROOT / config["output_dir"]
    baseline_source_path = REPO_ROOT / config["baseline"]["source"]
    previous_source_path = output_dir / f"candidate_{args.previous_index:03d}.cpp"
    previous_static_path = output_dir / f"candidate_{args.previous_index:03d}_static_validation.json"
    previous_csim_path = output_dir / f"candidate_{args.previous_index:03d}_csim_validation.json"
    previous_synthesis_path = output_dir / f"candidate_{args.previous_index:03d}_synthesis.json"
    baseline_diagnosis_path = output_dir / "baseline_hierarchical_diagnosis.json"
    source_target_path = output_dir / "baseline_source_target.json"
    source_cause_path = output_dir / "baseline_source_cause.json"

    for path in (
        baseline_source_path,
        previous_source_path,
        previous_static_path,
        baseline_diagnosis_path,
        source_target_path,
        source_cause_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Required refinement input not found: {path}")

    previous_static = load_json(previous_static_path)
    diagnosis = load_json(baseline_diagnosis_path)
    source_target = load_json(source_target_path)
    source_cause = load_json(source_cause_path)
    previous_source = previous_source_path.read_text(encoding="utf-8")
    baseline_source = baseline_source_path.read_text(encoding="utf-8")
    primary = source_cause.get("primary_hypothesis", {})

    checks = previous_static.get("checks") or {}
    bounds_failed = checks.get("constant_loop_tail_bounds_safe") is False

    if bounds_failed:
        bounds_issues = previous_static.get("bounds_issues") or []
        issue_text = "\n".join(f"- {item}" for item in bounds_issues) or (
            "- The j += 4 loop can start at j = 40 and access j + 2 and j + 3, "
            "which exceed the valid 0..41 range."
        )
        prompt = f"""You are performing iteration {args.next_index} of an AMD/Xilinx Vitis HLS PPA optimisation loop.

Objective:
Repair candidate {args.previous_index:03d} while preserving its useful structural idea: four independent partial accumulators for the dot-product reduction.

Measured validation evidence:
- Candidate {args.previous_index:03d} static validation: FAIL
- CSim: not run
- Synthesis: not run
- Failure category: unsafe constant loop tail bounds

Bounds evidence:
{issue_text}

Required repair:
1. Preserve the four-accumulator structural reduction idea.
2. Process the largest safe multiple-of-four prefix of the 42-element dot product.
3. Correctly process the remaining two elements without any out-of-bounds access.
4. Preserve the dot_loop label on the main optimised loop.
5. Ensure every element j = 0 through 41 is included exactly once.
6. Do not revert to the original two-accumulator implementation.
7. Do not add HLS DEPENDENCE pragmas.

Selected optimisation target:
- Function/report: {source_target.get('target_name')}
- Loop label: {source_target.get('loop_label')}
- Primary cause: {primary.get('category')}
- Cause interpretation: {primary.get('interpretation')}

Constraints:
1. Preserve the exact kernel_atax function signature and algorithmic behaviour.
2. Preserve the existing HLS top directive.
3. Modify only declarations, dot_loop, safe tail handling, and the directly associated final reduction.
4. Do not modify init_y, update_y, or unrelated loops.
5. Keep all A and x indices within bounds for A[38][42] and x[42].
6. Preserve numerical correctness under the existing testbench.
7. Return one complete compilable C++ source file only.
8. Do not include Markdown fences or explanations.

Rejected candidate {args.previous_index:03d} source to repair:
{previous_source}

Original baseline source for reference:
{baseline_source}
"""
        verdict = "reject_static_bounds_failure"
        verdict_text = "REJECT — unsafe tail bounds"
        required_direction = "repair four-accumulator structure with safe two-element tail"
        feedback = {
            "previous_candidate_index": args.previous_index,
            "next_candidate_index": args.next_index,
            "verdict": verdict,
            "static_validation_passed": False,
            "failed_check": "constant_loop_tail_bounds_safe",
            "bounds_issues": bounds_issues,
            "csim_run": False,
            "synthesis_run": False,
            "preserve_strategy": "four independent partial accumulators",
            "required_direction": required_direction,
        }
    else:
        if not previous_csim_path.is_file() or not previous_synthesis_path.is_file():
            raise FileNotFoundError(
                "Previous candidate passed static validation but CSim/synthesis evidence is missing."
            )
        previous_csim = load_json(previous_csim_path)
        previous_synthesis = load_json(previous_synthesis_path)
        if previous_csim.get("passed") is not True:
            raise RuntimeError("Previous candidate did not pass CSim.")
        if previous_synthesis.get("passed") is not True:
            raise RuntimeError("Previous candidate synthesis did not pass.")

        previous_metrics = previous_synthesis.get("metrics") or {}
        top_baseline = diagnosis.get("top_function") or diagnosis.get("top_level") or {}
        baseline_metrics = top_baseline.get("metrics") if isinstance(top_baseline, dict) else None
        if not isinstance(baseline_metrics, dict) or not baseline_metrics:
            baseline_metrics = previous_metrics

        prompt = f"""You are performing iteration {args.next_index} of an AMD/Xilinx Vitis HLS PPA optimisation loop.

Objective:
Reduce top-level latency or initiation interval while keeping LUT, FF, DSP, and BRAM growth proportionate. Functional correctness is mandatory.

Measured evidence from candidate {args.previous_index:03d}:
- Static validation: PASS
- Vitis CSim: PASS
- Vitis synthesis: PASS
- Verdict: REJECT because it produced no top-level PPA improvement.

Baseline top-level metrics:
{metric_summary(baseline_metrics)}

Candidate {args.previous_index:03d} top-level metrics:
{metric_summary(previous_metrics)}

The metrics are identical. Do not repeat HLS DEPENDENCE pragmas or another pragma-only modification. Apply a structural transformation to shorten the floating-point reduction recurrence.

Selected optimisation target:
- Function/report: {source_target.get('target_name')}
- Loop label: {source_target.get('loop_label')}
- Primary cause: {primary.get('category')}
- Cause interpretation: {primary.get('interpretation')}

Required strategy:
Use additional independent partial accumulators followed by a safe final reduction. Preserve all 42 elements and handle any remainder explicitly.

Constraints:
1. Preserve the exact kernel_atax function signature and algorithmic behaviour.
2. Preserve the existing HLS top directive.
3. Modify only declarations, dot_loop, safe tail handling, and final reduction statements.
4. Do not modify protected loops.
5. Keep every A and x index within bounds.
6. Return one complete compilable C++ source file only.
7. Do not include Markdown fences or explanations.

Rejected candidate {args.previous_index:03d} source:
{previous_source}

Original baseline source:
{baseline_source}
"""
        verdict = "reject_no_ppa_improvement"
        verdict_text = "REJECT — no top-level PPA improvement"
        required_direction = "structural reduction transformation"
        feedback = {
            "previous_candidate_index": args.previous_index,
            "next_candidate_index": args.next_index,
            "verdict": verdict,
            "csim_passed": True,
            "synthesis_passed": True,
            "baseline_metrics": baseline_metrics,
            "candidate_metrics": previous_metrics,
            "required_direction": required_direction,
        }

    prompt_path = output_dir / f"candidate_{args.next_index:03d}_prompt.txt"
    feedback_path = output_dir / f"candidate_{args.previous_index:03d}_feedback.json"
    feedback["prompt_file"] = str(prompt_path.relative_to(REPO_ROOT))

    prompt_path.write_text(prompt, encoding="utf-8")
    feedback_path.write_text(json.dumps(feedback, indent=2) + "\n", encoding="utf-8")

    print("\nPPA refinement prompt")
    print(f"Previous candidate: {previous_source_path.relative_to(REPO_ROOT)}")
    print(f"Previous verdict: {verdict_text}")
    print(f"Feedback: {feedback_path.relative_to(REPO_ROOT)}")
    print(f"Next prompt: {prompt_path.relative_to(REPO_ROOT)}")
    print(f"Required direction: {required_direction}")
    print("No model call, CSim, or synthesis was run.")


if __name__ == "__main__":
    main()
