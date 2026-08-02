#!/usr/bin/env python3

"""Analyse all Vitis HLS reports under a project and rank optimisation targets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.analysis.hierarchical_hls_analyzer import analyse_hierarchy  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Discover all Vitis HLS synthesis reports and rank bottlenecks across the design."
    )
    parser.add_argument("report_root", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--interface-frozen", action="store_true")
    args = parser.parse_args()

    report_root = args.report_root.expanduser().resolve()
    if not report_root.exists():
        raise SystemExit(f"Report root does not exist: {report_root}")

    result = analyse_hierarchy(report_root, interface_frozen=args.interface_frozen)
    output = (args.output or report_root / "hierarchical_diagnosis.json").expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    primary = result["primary_target"]
    diagnosis = primary["primary_diagnosis"]
    print("Hierarchical HLS diagnosis")
    print(f"Top function: {result['top_function']}")
    print(f"Reports analysed: {result['report_count']}")
    print()
    print("Ranked targets")
    for index, item in enumerate(result["ranked_targets"], start=1):
        marker = " [top]" if item["is_top"] else ""
        print(
            f"{index}. {item['function']}{marker}: "
            f"{item['primary_diagnosis']['category']} "
            f"(score={item['score']:.3f}, II={item['max_achieved_ii']:g}, "
            f"latency={item['latency_cycles']})"
        )
    print()
    print("Recommended focus")
    print(f"Target: {primary['function']}")
    print(f"Category: {diagnosis['category']}")
    for evidence in diagnosis.get("evidence", []):
        print(f"  - {evidence}")
    if result["protected_regions"]:
        print("Protected regions:")
        for item in result["protected_regions"]:
            print(f"  - {item['function']}: {item['reason']}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
