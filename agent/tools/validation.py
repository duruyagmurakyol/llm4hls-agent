"""Generic correctness and PPA candidate validation."""

from __future__ import annotations

import argparse
import difflib
import re
from pathlib import Path
from typing import Any

from agent.state import ValidationResult
from agent.tools.command_runner import CommandResult
from agent.tools.reports import load_json, write_json

REPO_ROOT = Path(__file__).resolve().parents[2]


def classify_failure(output: str) -> str:
    lower = output.lower()
    if "undefined reference" in lower or "linker" in lower:
        return "interface_or_link"
    if "error:" in lower or ("expected" in lower and "before" in lower):
        return "compile"
    if "fail index=" in lower or ("expected=" in lower and "actual=" in lower):
        return "functional"
    if "timeout" in lower or "timed out" in lower:
        return "timeout"
    return "unknown"


def extract_evidence(output: str, *, line_limit: int = 12, char_limit: int = 1200) -> list[str]:
    lines = [line for line in output.splitlines() if line.strip()]
    selected = [line for line in lines if any(token in line.lower() for token in ("error", "undefined", "fail", "expected", "actual", "timeout"))]
    joined = "\n".join((selected[-line_limit:] or lines[-line_limit:]))[-char_limit:]
    return joined.splitlines()


def from_command(result: CommandResult) -> ValidationResult:
    return ValidationResult(
        passed=result.passed,
        failure_class="none" if result.passed else classify_failure(result.output),
        return_code=result.return_code,
        evidence=[] if result.passed else extract_evidence(result.output),
    )


def _signature(text: str, name: str) -> str | None:
    match = re.search(rf'(?:extern\s+"C"\s*)?[\w:\s*&<>]+\b{re.escape(name)}\s*\([^)]*\)', text, re.MULTILINE)
    return " ".join(match.group(0).split()) if match else None


def _normalised_signature(text: str, name: str) -> str | None:
    value = _signature(text, name)
    return re.sub(r'^extern\s+"C"\s+', '', value) if value else None


def _has_c_linkage(text: str, name: str) -> bool:
    return bool(re.search(rf'extern\s+"C"\s+[\w:\s*&<>]+\b{re.escape(name)}\s*\(', text, re.MULTILINE))


def _matching_brace(text: str, opening: int) -> int | None:
    depth = 0
    for index in range(opening, len(text)):
        depth += text[index] == "{"
        depth -= text[index] == "}"
        if depth == 0:
            return index
    return None


def _loop_tail_bounds_safe(text: str) -> tuple[bool, list[dict[str, Any]]]:
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
        closing = _matching_brace(text, opening)
        if closing is None:
            continue
        body = text[opening + 1:closing]
        offsets = [int(item or 0) for item in re.findall(rf"\[\s*{re.escape(variable)}\s*(?:\+\s*(\d+))?\s*\]", body)] or [0]
        last = start + ((bound - 1 - start) // step) * step
        highest = last + max(offsets)
        if highest >= bound:
            issues.append({"loop_variable": variable, "start": start, "bound": bound, "step": step, "max_index_offset": max(offsets), "last_iteration_start": last, "highest_index_accessed": highest})
    return not issues, issues


def validate_ppa_candidate(config_path: Path, candidate_index: int = 1) -> dict[str, Any]:
    config = load_json(config_path.resolve())
    validation_config = config.get("validation", {})
    top = str(config.get("top_function", "")).strip()
    if not top:
        raise ValueError("Config is missing a non-empty 'top_function' field")

    output_dir = REPO_ROOT / config["output_dir"]
    baseline_path = REPO_ROOT / config["baseline"]["source"]
    candidate_path = output_dir / f"candidate_{candidate_index:03d}.cpp"
    baseline = baseline_path.read_text(encoding="utf-8")
    candidate = candidate_path.read_text(encoding="utf-8")
    source_target_path = output_dir / "baseline_source_target.json"
    source_target = load_json(source_target_path) if source_target_path.is_file() else {}

    bounds_enabled = bool(validation_config.get("constant_loop_tail_bounds", True))
    bounds_safe, bounds_issues = _loop_tail_bounds_safe(candidate) if bounds_enabled else (True, [])
    baseline_c, candidate_c = _has_c_linkage(baseline, top), _has_c_linkage(candidate, top)
    checks: dict[str, bool] = {
        "non_empty": bool(candidate.strip()),
        "contains_include": "#include" in candidate,
        "contains_required_top": bool(re.search(rf"\b{re.escape(top)}\s*\(", candidate)),
        "contains_no_markdown_fence": "```" not in candidate,
        "balanced_braces": candidate.count("{") == candidate.count("}"),
        "baseline_and_candidate_differ": baseline != candidate,
        "top_signature_preserved": _normalised_signature(baseline, top) == _normalised_signature(candidate, top) and _normalised_signature(candidate, top) is not None,
        "top_linkage_preserved": baseline_c == candidate_c,
    }
    if bounds_enabled:
        checks["constant_loop_tail_bounds_safe"] = bounds_safe

    labels = list(validation_config.get("required_loop_labels", []))
    discovered = source_target.get("loop_label")
    if validation_config.get("preserve_diagnosed_loop_label", True) and isinstance(discovered, str) and discovered:
        labels.append(discovered)
    for label in dict.fromkeys(label for label in labels if isinstance(label, str) and label):
        checks[f"loop_label_preserved:{label}"] = bool(re.search(rf"^\s*{re.escape(label)}\s*:\s*$", candidate, re.MULTILINE))

    diff_text = "".join(difflib.unified_diff(baseline.splitlines(keepends=True), candidate.splitlines(keepends=True), fromfile=str(baseline_path.relative_to(REPO_ROOT)), tofile=str(candidate_path.relative_to(REPO_ROOT))))
    diff_path = output_dir / f"candidate_{candidate_index:03d}_diff.patch"
    diff_path.write_text(diff_text, encoding="utf-8")
    changed_lines = sum(1 for line in diff_text.splitlines() if line.startswith(("+", "-")) and not line.startswith(("+++", "---")))
    report = {
        "candidate_index": candidate_index,
        "top_function": top,
        "baseline_file": str(baseline_path.relative_to(REPO_ROOT)),
        "candidate_file": str(candidate_path.relative_to(REPO_ROOT)),
        "diff_file": str(diff_path.relative_to(REPO_ROOT)),
        "passed": all(checks.values()),
        "checks": checks,
        "bounds_check_enabled": bounds_enabled,
        "bounds_issues": bounds_issues,
        "changed_diff_lines": changed_lines,
        "baseline_c_linkage": baseline_c,
        "candidate_c_linkage": candidate_c,
        "note": "Static validation is a pre-synthesis gate and does not prove functional correctness.",
    }
    write_json(output_dir / f"candidate_{candidate_index:03d}_static_validation.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate one generated HLS PPA candidate.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--candidate-index", type=int, default=1)
    args = parser.parse_args()
    report = validate_ppa_candidate(args.config, args.candidate_index)
    print("\nCandidate static validation")
    print(f"Top function: {report['top_function']}")
    for name, passed in report["checks"].items():
        print(f"{'PASS' if passed else 'FAIL'}: {name}")
    print(f"Overall: {'PASS' if report['passed'] else 'FAIL'}")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
