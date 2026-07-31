#!/usr/bin/env python3

"""Prepare a latency-area trade-off refinement prompt from a Pareto candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--source-index", type=int, required=True)
    parser.add_argument("--next-index", type=int, required=True)
    args = parser.parse_args()

    config = load_json(args.config.resolve())
    output_dir = REPO_ROOT / config["output_dir"]
    baseline_path = REPO_ROOT / config["baseline"]["source"]
    source_path = output_dir / f"candidate_{args.source_index:03d}.cpp"
    synthesis_path = output_dir / f"candidate_{args.source_index:03d}_synthesis.json"
    summary_path = output_dir / "experiment_summary.json"

    baseline = baseline_path.read_text(encoding="utf-8")
    source = source_path.read_text(encoding="utf-8")
    synthesis = load_json(synthesis_path)
    summary = load_json(summary_path)
    candidate_metrics = synthesis.get("metrics") or {}
    baseline_metrics = summary.get("baseline_metrics") or {}

    prompt = f"""You are performing iteration {args.next_index} of an AMD/Xilinx Vitis HLS PPA optimisation loop.

Candidate {args.source_index:03d} is a valid Pareto point, not a failed optimisation.
It improved top-level latency from {baseline_metrics.get('latency_best_cycles')} to {candidate_metrics.get('latency_best_cycles')} cycles, but increased LUT from {baseline_metrics.get('resources_lut_used')} to {candidate_metrics.get('resources_lut_used')}, FF from {baseline_metrics.get('resources_ff_used')} to {candidate_metrics.get('resources_ff_used')}, and DSP from {baseline_metrics.get('resources_dsp_used')} to {candidate_metrics.get('resources_dsp_used')}.

Objective:
Create a genuinely different implementation that preserves a meaningful latency improvement over the baseline while reducing resource usage relative to candidate {args.source_index:03d}.

Required direction:
- Do not reproduce candidate {args.source_index:03d} verbatim.
- Do not use the same four-accumulator implementation.
- Prefer a lower-parallelism structural reduction, such as three independent accumulators with safe remainder handling, or another balanced structure that plausibly uses fewer DSPs/LUTs/FFs.
- Target at least an 8 percent latency reduction versus the baseline if possible.
- Target DSP usage below {candidate_metrics.get('resources_dsp_used')} and substantially lower LUT/FF than candidate {args.source_index:03d}.
- Preserve every one of the 42 dot-product elements exactly once and keep all accesses in bounds.

Constraints:
1. Preserve the exact kernel_atax signature and algorithmic behaviour.
2. Preserve the dot_loop label.
3. Modify only the dot-product accumulation declarations, loop, safe tail handling, and final reduction.
4. Do not modify init_y, update_y, or unrelated loops.
5. Do not add HLS DEPENDENCE pragmas.
6. Return one complete compilable C++ source file only, without Markdown fences or explanations.

Pareto candidate source:
{source}

Original baseline source:
{baseline}
"""

    prompt_path = output_dir / f"candidate_{args.next_index:03d}_prompt.txt"
    feedback_path = output_dir / f"candidate_{args.source_index:03d}_tradeoff_feedback.json"
    prompt_path.write_text(prompt, encoding="utf-8")
    feedback_path.write_text(json.dumps({
        "source_candidate_index": args.source_index,
        "next_candidate_index": args.next_index,
        "verdict": "refine_pareto_tradeoff",
        "baseline_metrics": baseline_metrics,
        "source_metrics": candidate_metrics,
        "required_direction": "retain latency gain with lower resource cost",
        "prompt_file": str(prompt_path.relative_to(REPO_ROOT)),
    }, indent=2) + "\n", encoding="utf-8")

    print("\nPPA trade-off refinement prompt")
    print(f"Source Pareto candidate: {source_path.relative_to(REPO_ROOT)}")
    print(f"Next prompt: {prompt_path.relative_to(REPO_ROOT)}")
    print("Required direction: lower-resource alternative to the four-accumulator design")
    print("No model call, CSim, or synthesis was run.")


if __name__ == "__main__":
    main()
