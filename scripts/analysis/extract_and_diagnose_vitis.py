#!/usr/bin/env python3

"""Extract generic Vitis HLS evidence and run the bottleneck analyser."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent.analysis.hls_bottleneck_analyzer import analyse  # noqa: E402
from agent.analysis.vitis_evidence_extractor import extract_evidence  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract benchmark-independent evidence from Vitis HLS reports and diagnose bottlenecks."
    )
    parser.add_argument(
        "report_root",
        type=Path,
        help="Directory containing a Vitis HLS project, solution, or experiment workspace",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for evidence.json and diagnosis.json (default: report root)",
    )
    parser.add_argument(
        "--interface-frozen",
        action="store_true",
        help="Record that external interface changes are forbidden",
    )
    args = parser.parse_args()

    report_root = args.report_root.expanduser().resolve()
    if not report_root.exists():
        raise SystemExit(f"Report root does not exist: {report_root}")

    output_dir = (args.output_dir or report_root).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    evidence = extract_evidence(report_root, interface_frozen=args.interface_frozen)
    diagnosis = analyse(evidence)

    evidence_path = output_dir / "baseline_evidence.json"
    diagnosis_path = output_dir / "baseline_diagnosis.json"
    evidence_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    diagnosis_path.write_text(json.dumps(diagnosis, indent=2) + "\n", encoding="utf-8")

    primary = diagnosis["primary_diagnosis"]
    print("Vitis HLS evidence extraction")
    print(f"Report: {evidence['source_report']}")
    print(f"Loops extracted: {evidence['extraction']['loop_count']}")
    print(f"Warnings extracted: {evidence['extraction']['warning_count']}")
    if evidence["extraction"]["missing_fields"]:
        print("Missing fields: " + ", ".join(evidence["extraction"]["missing_fields"]))
    print()
    print("Primary diagnosis")
    print(f"Category: {primary['category']}")
    print(f"Target: {primary['target']}")
    print(f"Confidence: {primary['confidence']:.2f}")
    for item in primary.get("evidence", []):
        print(f"  - {item}")
    print(f"Stop recommended: {diagnosis['stop_recommended']}")
    print(f"Evidence: {evidence_path}")
    print(f"Diagnosis: {diagnosis_path}")


if __name__ == "__main__":
    main()
