from __future__ import annotations

import json
from typing import Any


def build_previous_attempts_text(
    previous_attempts: list[dict[str, Any]],
) -> str:
    if not previous_attempts:
        return "No previous generated attempts."

    lines: list[str] = []

    for attempt in previous_attempts[-5:]:
        result = attempt.get("result", {})
        metrics = result.get("metrics", {})

        lines.append(
            f"- Candidate: {attempt.get('candidate')}\n"
            f"  Outcome: {attempt.get('outcome')}\n"
            f"  Strategy: {attempt.get('strategy')}\n"
            f"  Clock: {metrics.get('estimated_clock_period_ns')} ns\n"
            f"  Target clock: {metrics.get('target_clock_period_ns')} ns\n"
            f"  Latency: {metrics.get('latency_worst_cycles')} cycles\n"
            f"  LUT: {metrics.get('resources_lut_used')}\n"
            f"  FF: {metrics.get('resources_ff_used')}\n"
            f"  DSP: {metrics.get('resources_dsp_used')}\n"
            f"  BRAM: {metrics.get('resources_bram_used')}\n"
            f"  Score: {attempt.get('score')}"
        )

    return "\n".join(lines)


def build_feedback(
    *,
    original_source: str,
    candidate_source: str,
    result: dict[str, Any],
    iteration: int,
    strategy: str,
    previous_attempts: list[dict[str, Any]],
) -> str:
    previous_attempts_text = build_previous_attempts_text(previous_attempts)

    return f"""
You are optimising an AMD/Xilinx Vitis HLS implementation of the ATAX
kernel.

This is optimisation iteration {iteration}.

Required strategy for this iteration:
{strategy}

Previous generated attempts:
{previous_attempts_text}

Do not reproduce an implementation equivalent to a previous failed or
non-improving attempt.

Optimisation priorities, in order:

1. Preserve functional correctness.
2. Pass C simulation and Vitis HLS synthesis.
3. Meet the target clock period of 10.0 ns.
4. Reduce latency.
5. Reduce LUT, FF, DSP and BRAM usage.

The estimated clock period is the primary optimisation objective until
the timing target is met.

Do not exchange a large latency increase for a small reduction in LUTs.

The current implementation appears to contain loop-carried
double-precision accumulation dependencies. Merely adding PIPELINE
pragmas may not remove these dependencies. Consider structural changes
such as independent partial accumulators, carefully bounded loop
unrolling, or a balanced final reduction.

The ATAX dimensions are:

- M = 38
- N = 42

Take care when unrolling because N=42 is not divisible by four or eight.
All array accesses must remain in bounds.

Original source:

{original_source}

Current best source:

{candidate_source}

Most recent evaluation result:

{json.dumps(result, indent=2)}

Return only the complete replacement C++ source.

Do not include explanations, Markdown code fences, or prose.
""".strip()