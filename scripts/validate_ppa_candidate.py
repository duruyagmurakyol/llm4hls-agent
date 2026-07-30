#!/usr/bin/env python3

"""Statically validate one generated HLS PPA candidate before Vitis runs."""

from __future__ import annotations

import argparse
import difflib
import json
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def function_signature(text: str, name: str) -> str | None:
    pattern = re.compile(
        rf'(?:extern\s+"C"\s*)?[\w:\s*&<>]+\b{re.escape(name)}\s*\([^)]*\)',
        re.MULTILINE,
    )
    match = pattern.search(text)
    return " ".join(match.group(0).split()) if match else None


def has_c_linkage(text: str, name: str) -> bool:
    pattern = re.compile(
        rf'extern\s+"C"\s+[\w:\s*&<>]+\b{re.escape(name)}\s*\(',
        re.MULTILINE,
    )
    return bool(pattern.search(text))


def normalised_signature(text: str, name: str) -> str | None:
    signature = function_signature(text, name)
    if signature is None:
        return None
    return re.sub(r'^extern\s+"C"\s+', '', signature)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare a generated PPA candidate with its configured baseline."
    )
    parser.add_argument("config", type=Path, help="PPA optimisation JSON config")
    parser.add_argument(
        "--candidate-index",
        type=int,
        default=1,
        help="Candidate number used in filenames (default: 1)",
    )
    args = parser.parse_args()

    config = load_json(args.config.resolve())
    output_dir = REPO_ROOT / config["output_dir"]
    baseline_path = REPO_ROOT / config["baseline"]["source"]
    candidate_path = output_dir / f"candidate_{args.candidate_index:03d}.cpp"
    source_target_path = output_dir / "baseline_source_target.json"

    if not baseline_path.is_file():
        raise FileNotFoundError(f"Baseline source not found: {baseline_path}")
    if not candidate_path.is_file():
        raise FileNotFoundError(f"Candidate source not found: {candidate_path}")

    baseline = baseline_path.read_text(encoding="utf-8")
    candidate = candidate_path.read_text(encoding="utf-8")
    source_target = load_json(source_target_path) if source_target_path.is_file() else {}

    required_top = "kernel_atax"
    baseline_c_linkage = has_c_linkage(baseline, required_top)
    candidate_c_linkage = has_c_linkage(candidate, required_top)

    checks = {
        "non_empty": bool(candidate.strip()),
        "contains_include": "#include" in candidate,
        "contains_required_top": bool(re.search(rf"\b{required_top}\s*\(", candidate)),
        "contains_no_markdown_fence": "```" not in candidate,
        "balanced_braces": candidate.count("{") == candidate.count("}"),
        "baseline_and_candidate_differ": baseline != candidate,
        "top_signature_preserved": (
            normalised_signature(baseline, required_top)
            == normalised_signature(candidate, required_top)
            and normalised_signature(candidate, required_top) is not None
        ),
        "top_linkage_preserved": baseline_c_linkage == candidate_c_linkage,
    }

    loop_label = source_target.get("loop_label")
    if isinstance(loop_label, str) and loop_label:
        checks["target_loop_label_preserved"] = bool(
            re.search(rf"^\s*{re.escape(loop_label)}\s*:\s*$", candidate, re.MULTILINE)
        )

    baseline_lines = baseline.splitlines(keepends=True)
    candidate_lines = candidate.splitlines(keepends=True)
    diff_text = "".join(
        difflib.unified_diff(
            baseline_lines,
            candidate_lines,
            fromfile=str(baseline_path.relative_to(REPO_ROOT)),
            tofile=str(candidate_path.relative_to(REPO_ROOT)),
        )
    )

    diff_path = output_dir / f"candidate_{args.candidate_index:03d}_diff.patch"
    report_path = output_dir / f"candidate_{args.candidate_index:03d}_static_validation.json"
    diff_path.write_text(diff_text, encoding="utf-8")

    changed_lines = sum(
        1
        for line in diff_text.splitlines()
        if (line.startswith("+") or line.startswith("-"))
        and not line.startswith("+++")
        and not line.startswith("---")
    )

    passed = all(checks.values())
    report = {
        "candidate_index": args.candidate_index,
        "baseline_file": str(baseline_path.relative_to(REPO_ROOT)),
        "candidate_file": str(candidate_path.relative_to(REPO_ROOT)),
        "diff_file": str(diff_path.relative_to(REPO_ROOT)),
        "passed": passed,
        "checks": checks,
        "changed_diff_lines": changed_lines,
        "baseline_line_count": len(baseline.splitlines()),
        "candidate_line_count": len(candidate.splitlines()),
        "target_loop_label": loop_label,
        "baseline_c_linkage": baseline_c_linkage,
        "candidate_c_linkage": candidate_c_linkage,
        "note": "Static validation is a pre-synthesis gate and does not prove functional correctness.",
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\nCandidate static validation")
    print(f"Candidate: {candidate_path.relative_to(REPO_ROOT)}")
    for name, result in checks.items():
        print(f"{'PASS' if result else 'FAIL'}: {name}")
    print(f"Changed diff lines: {changed_lines}")
    print(f"Diff: {diff_path.relative_to(REPO_ROOT)}")
    print(f"Report: {report_path.relative_to(REPO_ROOT)}")
    print(f"Overall: {'PASS' if passed else 'FAIL'}")
    print("No compilation, CSim, synthesis, or baseline modification was performed.")

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
