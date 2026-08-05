#!/usr/bin/env python3

"""Extract report-ready evidence from a completed agent suite run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.reporting import build_final_report, write_final_report
from agent.terminal_reporting import render_suite_terminal


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write final_results.json, final_results.csv and final_results.md."
    )
    parser.add_argument("suite_root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    paths = write_final_report(args.suite_root, output_dir=args.output_dir)
    report = build_final_report(args.suite_root)

    print(render_suite_terminal(report))
    print("\nOutput files")
    print("============")
    for label, path in paths.items():
        print(f"{label + ':':<12}{path}")


if __name__ == "__main__":
    main()
