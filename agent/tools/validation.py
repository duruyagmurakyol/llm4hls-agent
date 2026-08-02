"""Generic correctness-result classification shared by repair workflows."""

from __future__ import annotations

from agent.state import ValidationResult
from agent.tools.command_runner import CommandResult


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
    selected = [
        line for line in lines
        if any(token in line.lower() for token in ("error", "undefined", "fail", "expected", "actual", "timeout"))
    ]
    text = (selected[-line_limit:] or lines[-line_limit:])
    joined = "\n".join(text)[-char_limit:]
    return joined.splitlines()


def from_command(result: CommandResult) -> ValidationResult:
    return ValidationResult(
        passed=result.passed,
        failure_class="none" if result.passed else classify_failure(result.output),
        return_code=result.return_code,
        evidence=[] if result.passed else extract_evidence(result.output),
    )
