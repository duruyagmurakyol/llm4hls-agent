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
    match = re.search(rf'(?:extern\s+"C"\s*)?[\w:\s*&<>]+\b{re.escape(name)}\s*\([^)]*\)', text, re.MULTILINE)
    return " ".join(match.group(0).split()) if match else None


def has_c_linkage(text: str, name: str) -> bool:
    return bool(re.search(rf'extern\s+"C"\s+[\w:\s*&<>]+\b{re.escape(name)}\s*\(', text, re.MULTILINE))


def normalised_signature(text: str, name: str) -> str | None:
    signature = function_signature(text, name)
    return re.sub(r'^extern\s+"C"\s+', '', signature) if signature else None


def matching_brace(text: str, opening: int) -> int | None:
    depth = 0
    for index in range(opening, len(text)):
        depth += text[index] == "{"
        depth -= text[index] == "}"
        if depth == 0:
            return index
    return None


def loop_tail_bounds_safe(text: str) -> tuple[bool, list[dict[str, Any]]]:
    constants = {name: int(value) for name, value in re.findall(r"\b(?:const\s+)?int\s+([A-Za-z_]\w*)\s*=\s*(\d+)\s*;", text)}
    issues: list[dict[str, Any]] = []
    pattern = re.compile(r"for\s*\(\s*([A-Za-z_]\w*)\s*=\s*(\d+)\s*;\s*\1\s*<\s*([A-Za-z_]\w*|\d+)\s*;\s*\1\s*\+=\s*(\d+)\s*\)\s*\{")
    for match in pattern.finditer(text):
        variable, start_text, bound_text, step_text = match.groups()
        bound = int(bound_text) if bound_text.isdigit() else constants.get(bound_text)
        if bound is None:
            continue
        start, step = int(start_text), int(step_text)
        if step <= 0 or start >= bound:
            continue
        opening = text.find("{", match.start())
        closing = matching_brace(text, opening)
        if closing is None:
            continue
        body = text[opening + 1:closing]
        offsets = [int(item or 0) for item in re.findall(rf"\[\s*{re.escape(variable)}\s*(?:\+\s*(\d+))?\s*\]", body)] or [0]
        last = start + ((bound - 1 - start) // step) * step
        highest = last + max(offsets)
        if highest >= bound:
            issues.append({"loop_variable": variable, "start": start, "bound": bound, "step": step, "max_index_offset": max(offsets), "last_iteration_start": last, "highest_index_accessed": highest})
    return not issues, issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare a generated PPA candidate with its configured baseline.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--candidate-index", type=int, default=1)
    args = parser.parse_args()

    config = load_json(args.config.resolve())
    validation_config = config.get("validation", {})
    required_top = config.get("top_function")
    if not isinstance(required_top, str) or not required_top.strip():
        raise ValueError("Config is missing a non-empty 'top_function' field")
    required_top = required_top.strip()

    output_dir = REPO_ROOT / config["output_dir"]
    baseline_path = REPO_ROOT / config["baseline"]["source"]
    candidate_path = output_dir / f"candidate_{args.candidate_index:03d}.cpp"
    source_target_path = output_dir / "baseline_source_target.json"
    baseline = baseline_path.read_text(encoding="utf-8")
    candidate = candidate_path.read_text(encoding="utf-8")
    source_target = load_json(source_target_path) if source_target_path.is_file() else {}

    bounds_enabled = bool(validation_config.get("constant_loop_tail_bounds", True))
    bounds_safe, bounds_issues = loop_tail_bounds_safe(candidate) if bounds_enabled else (True, [])
    baseline_c, candidate_c = has_c_linkage(baseline, required_top), has_c_linkage(candidate, required_top)

    checks = {
        "non_empty": bool(candidate.strip()),
        "contains_include": "#include" in candidate,
        "contains_required_top": bool(re.search(rf"\b{re.escape(required_top)}\s*\(", candidate)),
        "contains_no_markdown_fence": "```" not in candidate,
        "balanced_braces": candidate.count("{") == candidate.count("}"),
        "baseline_and_candidate_differ": baseline != candidate,
        "top_signature_preserved": normalised_signature(baseline, required_top) == normalised_signature(candidate, required_top) and normalised_signature(candidate, required_top) is not None,
        "top_linkage_preserved": baseline_c == candidate_c,
    }
    if bounds_enabled:
        checks["constant_loop_tail_bounds_safe"] = bounds_safe

    labels = validation_config.get("required_loop_labels", [])
    discovered = source_target.get("loop_label")
    if validation_config.get("preserve_diagnosed_loop_label", True) and isinstance(discovered, str) and discovered:
        labels = [*labels, discovered]
    for label in dict.fromkeys(label for label in labels if isinstance(label, str) and label):
        checks[f"loop_label_preserved:{label}"] = bool(re.search(rf"^\s*{re.escape(label)}\s*:\s*$", candidate, re.MULTILINE))

    diff_text = "".join(difflib.unified_diff(baseline.splitlines(keepends=True), candidate.splitlines(keepends=True), fromfile=str(baseline_path.relative_to(REPO_ROOT)), tofile=str(candidate_path.relative_to(REPO_ROOT))))
    diff_path = output_dir / f"candidate_{args.candidate_index:03d}_diff.patch"
    report_path = output_dir / f"candidate_{args.candidate_index:03d}_static_validation.json"
    diff_path.write_text(diff_text, encoding="utf-8")
    changed_lines = sum(1 for line in diff_text.splitlines() if (line.startswith("+") or line.startswith("-")) and not line.startswith(("+++", "---")))
    passed = all(checks.values())
    report = {"candidate_index": args.candidate_index, "top_function": required_top, "baseline_file": str(baseline_path.relative_to(REPO_ROOT)), "candidate_file": str(candidate_path.relative_to(REPO_ROOT)), "diff_file": str(diff_path.relative_to(REPO_ROOT)), "passed": passed, "checks": checks, "bounds_check_enabled": bounds_enabled, "bounds_issues": bounds_issues, "changed_diff_lines": changed_lines, "baseline_c_linkage": baseline_c, "candidate_c_linkage": candidate_c, "note": "Static validation is a pre-synthesis gate and does not prove functional correctness."}
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\nCandidate static validation")
    print(f"Top function: {required_top}")
    for name, result in checks.items():
        print(f"{'PASS' if result else 'FAIL'}: {name}")
    print(f"Overall: {'PASS' if passed else 'FAIL'}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
