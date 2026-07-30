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
        description="Create the next PPA prompt from a measured previous candidate."
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
    previous_csim_path = output_dir / f"candidate_{args.previous_index:03d}_csim_validation.json"
    previous_synthesis_path = output_dir / f"candidate_{args.previous_index:03d}_synthesis.json"
    baseline_diagnosis_path = output_dir / "baseline_hierarchical_diagnosis.json"
    source_target_path = output_dir / "baseline_source_target.json"
    source_cause_path = output_dir / "baseline_source_cause.json"

    for path in (
        baseline_source_path,
        previous_source_path,
        previous_csim_path,
        previous_synthesis_path,
        baseline_diagnosis_path,
        source_target_path,
        source_cause_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Required refinement input not found: {path}")

    previous_csim = load_json(previous_csim_path)
    previous_synthesis = load_json(previous_synthesis_path)
    diagnosis = load_json(baseline_diagnosis_path)
    source_target = load_json(source_target_path)
    source_cause = load_json(source_cause_path)

    if previous_csim.get("passed") is not True:
        raise RuntimeError("Previous candidate did not pass CSim; cannot use PPA refinement flow.")
    if previous_synthesis.get("passed") is not True:
        raise RuntimeError("Previous candidate synthesis did not pass.")

    previous_metrics = previous_synthesis.get("metrics") or {}
    top_baseline = diagnosis.get("top_function") or diagnosis.get("top_level") or {}
    baseline_metrics = top_baseline.get("metrics") if isinstance(top_baseline, dict) else None
    if not isinstance(baseline_metrics, dict) or not baseline_metrics:
        # Candidate 001 was measured against the cached baseline and found identical.
        baseline_metrics = previous_metrics

    previous_source = previous_source_path.read_text(encoding="utf-8")
    baseline_source = baseline_source_path.read_text(encoding="utf-8")
    primary = source_cause.get("primary_hypothesis", {})

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

The metrics are identical. The previous edit added only dependence pragmas for acc0 and acc1. Vitis accepted them but generated the same implementation. Therefore:
- Do not repeat HLS DEPENDENCE pragmas.
- Do not return a pragma-only modification.
- Do not claim a dependency is false when the accumulator is genuinely loop-carried.
- Apply a structural transformation to shorten the floating-point reduction recurrence.

Selected optimisation target:
- Function/report: {source_target.get('target_name')}
- Loop label: {source_target.get('loop_label')}
- Source region: lines {source_target.get('region_start_line')}-{source_target.get('region_end_line')}
- Primary cause: {primary.get('category')}
- Cause interpretation: {primary.get('interpretation')}
- Cause evidence: {json.dumps(primary.get('evidence', {}), indent=2)}

Required strategy:
Use a focused structural reduction transformation, such as additional independent partial accumulators followed by a final reduction, or another balanced reduction structure. Preserve the fixed dimensions and safely handle all 42 elements. Avoid changing unrelated loops.

Constraints:
1. Preserve the exact kernel_atax function signature and algorithmic behaviour.
2. Preserve the existing HLS top directive.
3. Modify only declarations, the dot_loop, and the directly associated final reduction statements.
4. Do not modify init_y or other protected loops.
5. Keep every A and x index within bounds for A[38][42] and x[42].
6. Preserve numerical correctness under the existing testbench.
7. Avoid array partitioning, full unrolling, or aggressive parallelism unless essential.
8. Return one complete compilable C++ source file only.
9. Do not include Markdown fences or explanations.

Rejected candidate {args.previous_index:03d} source:
{previous_source}

Original baseline source:
{baseline_source}
"""

    prompt_path = output_dir / f"candidate_{args.next_index:03d}_prompt.txt"
    feedback_path = output_dir / f"candidate_{args.previous_index:03d}_feedback.json"

    feedback = {
        "previous_candidate_index": args.previous_index,
        "next_candidate_index": args.next_index,
        "verdict": "reject_no_ppa_improvement",
        "csim_passed": True,
        "synthesis_passed": True,
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": previous_metrics,
        "prohibited_repeat": [
            "HLS DEPENDENCE pragma-only changes",
            "false dependence assertions on genuine accumulators",
        ],
        "required_direction": "structural reduction transformation",
        "prompt_file": str(prompt_path.relative_to(REPO_ROOT)),
    }

    prompt_path.write_text(prompt, encoding="utf-8")
    feedback_path.write_text(json.dumps(feedback, indent=2) + "\n", encoding="utf-8")

    print("\nPPA refinement prompt")
    print(f"Previous candidate: {previous_source_path.relative_to(REPO_ROOT)}")
    print("Previous verdict: REJECT — no top-level PPA improvement")
    print(f"Feedback: {feedback_path.relative_to(REPO_ROOT)}")
    print(f"Next prompt: {prompt_path.relative_to(REPO_ROOT)}")
    print("Required direction: structural reduction transformation")
    print("No model call, CSim, or synthesis was run.")


if __name__ == "__main__":
    main()
