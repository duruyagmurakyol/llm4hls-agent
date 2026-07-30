#!/usr/bin/env python3

"""Combine hierarchical HLS diagnosis with source-aware cause hypotheses."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.analysis.source_cause_analyzer import analyse_source_cause  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map the primary hierarchical HLS target to source and infer likely causes."
    )
    parser.add_argument("project_root", type=Path, help="Benchmark/project directory containing C/C++ source")
    parser.add_argument("hierarchy_json", type=Path, help="hierarchical_diagnosis.json")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path")
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    hierarchy_path = args.hierarchy_json.expanduser().resolve()
    if not project_root.exists():
        raise SystemExit(f"Project root does not exist: {project_root}")
    if not hierarchy_path.exists():
        raise SystemExit(f"Hierarchy JSON does not exist: {hierarchy_path}")

    hierarchy = json.loads(hierarchy_path.read_text(encoding="utf-8"))
    result = analyse_source_cause(project_root, hierarchy)
    output = (args.output or hierarchy_path.with_name("source_cause_analysis.json")).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    hypothesis = result["primary_hypothesis"]
    print("Source-aware HLS cause analysis")
    print(f"Target: {result['target']}")
    match = result.get("source_match")
    if match:
        print(f"Source: {match['path']}:{match['line_start']}-{match['line_end']}")
    else:
        print("Source: not found")
    print(f"Likely cause: {hypothesis['cause']}")
    print(f"Confidence: {hypothesis['confidence']:.2f}")
    for item in hypothesis.get("evidence", []):
        print(f"  - {item}")
    print("Next checks:")
    for item in hypothesis.get("next_checks", []):
        print(f"  - {item}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
