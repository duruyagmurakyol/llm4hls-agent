#!/usr/bin/env python3

"""Apply narrowly safe ABI normalisation to a generated HLS candidate."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def has_c_linkage(text: str, name: str) -> bool:
    return bool(re.search(rf'extern\s+"C"\s+[\w:\s*&<>]+\b{re.escape(name)}\s*\(', text, re.MULTILINE))


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalise safe top-function linkage mismatches.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--candidate-index", type=int, default=1)
    args = parser.parse_args()

    config = load_json(args.config.resolve())
    top = config.get("top_function")
    if not isinstance(top, str) or not top.strip():
        raise ValueError("Config is missing a non-empty 'top_function' field")
    top = top.strip()

    output_dir = REPO_ROOT / config["output_dir"]
    baseline_path = REPO_ROOT / config["baseline"]["source"]
    candidate_path = output_dir / f"candidate_{args.candidate_index:03d}.cpp"
    backup_path = output_dir / f"candidate_{args.candidate_index:03d}_before_normalisation.cpp"
    report_path = output_dir / f"candidate_{args.candidate_index:03d}_normalisation.json"

    baseline = baseline_path.read_text(encoding="utf-8")
    candidate = candidate_path.read_text(encoding="utf-8")
    baseline_c = has_c_linkage(baseline, top)
    candidate_c = has_c_linkage(candidate, top)
    changed = False
    action = "none"

    if not baseline_c and candidate_c:
        pattern = re.compile(rf'extern\s+"C"\s+(?=[\w:\s*&<>]+\b{re.escape(top)}\s*\()', re.MULTILINE)
        updated, count = pattern.subn("", candidate, count=1)
        if count != 1:
            raise RuntimeError("Could not safely remove candidate-only extern C linkage.")
        backup_path.write_text(candidate, encoding="utf-8")
        candidate_path.write_text(updated, encoding="utf-8")
        changed = True
        action = "removed_candidate_only_extern_c"
    elif baseline_c and not candidate_c:
        raise RuntimeError("Baseline uses extern C but candidate does not; automatic insertion is disabled.")

    report = {
        "candidate_index": args.candidate_index,
        "top_function": top,
        "baseline_c_linkage": baseline_c,
        "candidate_c_linkage_before": candidate_c,
        "changed": changed,
        "action": action,
        "candidate_file": str(candidate_path.relative_to(REPO_ROOT)),
        "backup_file": str(backup_path.relative_to(REPO_ROOT)) if changed else None,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\nCandidate ABI normalisation")
    print(f"Top function: {top}")
    print(f"Action: {action}")
    print(f"Report: {report_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
