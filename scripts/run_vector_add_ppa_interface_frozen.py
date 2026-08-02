#!/usr/bin/env python3

"""Run a vector-add PPA experiment while freezing the HLS interface architecture."""

from __future__ import annotations

import argparse
import difflib
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.siliconflow import complete  # noqa: E402
from scripts.run_vector_add_ppa_experiment import (  # noqa: E402
    acceptable,
    delta,
    extract_cpp,
    host_validate,
    protected_hashes,
    synthesis,
)


def interface_pragmas(source: str) -> tuple[str, ...]:
    """Return a normalized, ordered representation of all HLS interface pragmas."""
    pragmas: list[str] = []
    for line in source.splitlines():
        if re.match(r"^\s*#\s*pragma\s+HLS\s+INTERFACE\b", line, flags=re.I):
            pragmas.append(" ".join(line.strip().lower().split()))
    return tuple(sorted(pragmas))


def structural_reasons(baseline_source: str, candidate_source: str) -> list[str]:
    reasons: list[str] = []
    if interface_pragmas(candidate_source) != interface_pragmas(baseline_source):
        reasons.append("interface_pragmas_changed")
    return reasons


def make_prompt(source: str, metrics: dict[str, Any]) -> str:
    return f"""Optimise this correct AMD Vitis HLS vector-add implementation for hardware PPA.

Return only the complete contents of src/vector_add.cpp.

Hard constraints:
- Preserve the exact function signature and integer behaviour.
- Do not add, remove, or modify any #pragma HLS INTERFACE directive.
- The baseline has no explicit HLS interface pragmas, so the candidate must also have none.
- Do not change headers, testbench, VECTOR_SIZE, target device, or clock.
- Do not introduce local copies of the complete input or output arrays.
- Preserve synthesizability.

Permitted transformations:
- Loop pipelining.
- Small, controlled loop-unroll factors.
- Loop restructuring that preserves exact behaviour.
- Other non-interface HLS pragmas where justified.

Objectives:
1. Reduce best-case latency cycles.
2. Do not worsen latency to reduce resources.
3. Avoid large LUT, FF, BRAM, or DSP increases.

Baseline synthesis metrics:
{json.dumps(metrics, indent=2)}

Current source:
```cpp
{source}
```
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="configs/vector_add_ppa_qwen35.json")
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))

    source_workspace = ROOT / config["workspace"]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = ROOT / "results/experiments/vector_add_ppa_interface_frozen_qwen35" / timestamp
    workspace = output_dir / "workspace"
    output_dir.mkdir(parents=True)
    shutil.copytree(source_workspace, workspace)

    editable = workspace / config["editable_file"]
    baseline_source = editable.read_text(encoding="utf-8")
    hashes = protected_hashes(workspace, config["protected_files"])

    baseline_host = host_validate(workspace, output_dir, "baseline")
    baseline_synth, baseline_metrics = synthesis(workspace, output_dir, "baseline")
    if not baseline_host or not baseline_synth:
        raise SystemExit(f"Baseline validation/synthesis failed. See {output_dir.relative_to(ROOT)}")

    response = complete(
        model=config["model"],
        system_prompt="You are an expert AMD Vitis HLS optimisation agent.",
        user_prompt=make_prompt(baseline_source, baseline_metrics),
        temperature=float(config["temperature"]),
        max_tokens=int(config["max_output_tokens"]),
        timeout_seconds=int(config["api_timeout_seconds"]),
        thinking_budget=config.get("thinking_budget"),
    )

    candidate_dir = output_dir / "candidate_1"
    candidate_dir.mkdir()
    (candidate_dir / "raw_response.txt").write_text(response.content, encoding="utf-8")

    record: dict[str, Any] = {
        "candidate": 1,
        "tokens": response.total_tokens,
        "api_latency_seconds": round(response.latency_seconds, 3),
    }
    synthesis_calls = 1
    accepted = False

    try:
        candidate_source = extract_cpp(response.content)
    except ValueError as error:
        record.update(status="parse_failed", error=str(error))
    else:
        (candidate_dir / "candidate.cpp").write_text(candidate_source, encoding="utf-8")
        diff = "".join(
            difflib.unified_diff(
                baseline_source.splitlines(True),
                candidate_source.splitlines(True),
                fromfile="baseline/src/vector_add.cpp",
                tofile="candidate/src/vector_add.cpp",
            )
        )
        (candidate_dir / "repair.diff").write_text(diff, encoding="utf-8")

        reasons = structural_reasons(baseline_source, candidate_source)
        record["structural_reasons"] = reasons
        if reasons:
            record["status"] = "structural_rejected"
            record["host_passed"] = None
            record["synthesis_passed"] = None
        else:
            editable.write_text(candidate_source, encoding="utf-8")
            host_ok = host_validate(workspace, candidate_dir, "candidate")
            record["host_passed"] = host_ok
            if not host_ok:
                record["status"] = "correctness_rejected"
            else:
                synth_ok, candidate_metrics = synthesis(workspace, candidate_dir, "candidate")
                synthesis_calls += 1
                record["synthesis_passed"] = synth_ok
                record["metrics"] = candidate_metrics
                record["delta"] = delta(baseline_metrics, candidate_metrics)
                if synth_ok:
                    accepted, acceptance_reasons = acceptable(baseline_metrics, candidate_metrics)
                else:
                    accepted, acceptance_reasons = False, ["synthesis_failed"]
                record["acceptance_reasons"] = acceptance_reasons
                record["status"] = "accepted" if accepted else "ppa_rejected"

    unchanged = protected_hashes(workspace, config["protected_files"]) == hashes
    if not accepted:
        editable.write_text(baseline_source, encoding="utf-8")

    result = {
        "schema_version": 1,
        "experiment_id": "vector_add_ppa_interface_frozen_qwen35",
        "timestamp_utc": timestamp,
        "model": config["model"],
        "interface_frozen": True,
        "baseline_interface_pragmas": list(interface_pragmas(baseline_source)),
        "baseline_host_passed": baseline_host,
        "baseline_synthesis_passed": baseline_synth,
        "baseline_metrics": baseline_metrics,
        "synthesis_calls": synthesis_calls,
        "synthesis_budget": 2,
        "accepted": accepted,
        "protected_files_unchanged": unchanged,
        "candidate": record,
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print("Interface-frozen vector-add PPA experiment")
    print(f"Baseline metrics: {baseline_metrics}")
    print(f"Candidate status: {record['status']}")
    if record.get("structural_reasons"):
        print(f"Structural rejection: {record['structural_reasons']}")
    if record.get("metrics"):
        print(f"Candidate metrics: {record['metrics']}")
        print(f"Metric deltas: {record['delta']}")
    print(f"Accepted improvement: {accepted}")
    print(f"Synthesis calls: {synthesis_calls}/2")
    print(f"Protected files unchanged: {unchanged}")
    print(f"Results: {output_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
