"""Pre-write validation for model-generated HLS repair sources."""

from __future__ import annotations

import hashlib
import re
from typing import Any


class InvalidModelOutputError(ValueError):
    """Raised when a model response is unsafe to write into the workspace."""

    def __init__(self, report: dict[str, Any], *, response: Any) -> None:
        self.report = report
        self.raw_response = str(response.content)
        self.raw_api_response = response.raw_response
        self.input_tokens = response.input_tokens
        self.output_tokens = response.output_tokens
        self.total_tokens = response.total_tokens
        self.latency_seconds = response.latency_seconds
        evidence = report.get("evidence") or ["model output validation failed"]
        super().__init__("; ".join(str(item) for item in evidence))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _scrub_comments_and_literals(source: str) -> str:
    """Replace comments and literal contents while preserving positions and lines."""
    result: list[str] = []
    index = 0
    state = "code"
    quote = ""
    while index < len(source):
        char = source[index]
        nxt = source[index + 1] if index + 1 < len(source) else ""

        if state == "code":
            if char == "/" and nxt == "/":
                result.extend("  ")
                index += 2
                state = "line_comment"
                continue
            if char == "/" and nxt == "*":
                result.extend("  ")
                index += 2
                state = "block_comment"
                continue
            if char in {'"', "'"}:
                quote = char
                result.append(char)
                index += 1
                state = "literal"
                continue
            result.append(char)
            index += 1
            continue

        if state == "line_comment":
            if char == "\n":
                result.append("\n")
                state = "code"
            else:
                result.append(" ")
            index += 1
            continue

        if state == "block_comment":
            if char == "*" and nxt == "/":
                result.extend("  ")
                index += 2
                state = "code"
            else:
                result.append("\n" if char == "\n" else " ")
                index += 1
            continue

        if char == "\\" and index + 1 < len(source):
            result.extend("  ")
            index += 2
            continue
        result.append(char if char == quote else ("\n" if char == "\n" else " "))
        index += 1
        if char == quote:
            state = "code"

    return "".join(result)


def _balanced_structure(source: str) -> tuple[bool, str | None]:
    scrubbed = _scrub_comments_and_literals(source)
    opening = {"(": ")", "[": "]", "{": "}"}
    closing = {value: key for key, value in opening.items()}
    stack: list[str] = []
    for char in scrubbed:
        if char in opening:
            stack.append(char)
        elif char in closing:
            if not stack or stack[-1] != closing[char]:
                return False, f"Unexpected closing delimiter {char!r}."
            stack.pop()
    if stack:
        return False, f"Unclosed delimiter {stack[-1]!r}."
    return True, None


def _normalise_source(source: str) -> str:
    return re.sub(r"\s+", " ", _scrub_comments_and_literals(source)).strip()


def _matching_paren(source: str, opening_index: int) -> int | None:
    depth = 0
    for index in range(opening_index, len(source)):
        char = source[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index
    return None


def _top_signature(source: str, top_function: str) -> str | None:
    scrubbed = _scrub_comments_and_literals(source)
    pattern = re.compile(
        rf"(?m)^[ \t]*(?P<prefix>[A-Za-z_][\w:\s<>,*&\[\]\"]*?)\b"
        rf"{re.escape(top_function)}\s*\("
    )
    for match in pattern.finditer(scrubbed):
        opening_index = scrubbed.find("(", match.start(), match.end())
        closing_index = _matching_paren(scrubbed, opening_index)
        if closing_index is None:
            continue
        cursor = closing_index + 1
        while cursor < len(scrubbed) and scrubbed[cursor].isspace():
            cursor += 1
        if scrubbed.startswith("const", cursor):
            cursor += len("const")
            while cursor < len(scrubbed) and scrubbed[cursor].isspace():
                cursor += 1
        if cursor >= len(scrubbed) or scrubbed[cursor] != "{":
            continue
        prefix = match.group("prefix").strip()
        parameters = scrubbed[opening_index + 1 : closing_index].strip()
        signature = f"{prefix} {top_function}({parameters})"
        return re.sub(r"\s+", " ", signature).strip()
    return None


def _extract_editable_source(user_prompt: str) -> str | None:
    match = re.search(
        r"EDITABLE FILE:[^\n]*\n```[^\n]*\n(?P<source>.*?)\n```",
        user_prompt,
        re.DOTALL,
    )
    return match.group("source") if match else None


def _extract_top_function(user_prompt: str) -> str | None:
    match = re.search(
        r"Preserve the ([A-Za-z_]\w*) function name and signature\.",
        user_prompt,
    )
    return match.group(1) if match else None


def validate_model_output(
    *,
    raw_response: str,
    candidate_source: str,
    baseline_source: str,
    top_function: str,
    protected_files_unchanged: bool = True,
) -> dict[str, Any]:
    """Validate generated source before the editable file is overwritten."""
    checks: dict[str, bool] = {}
    violations: list[str] = []
    evidence: list[str] = []

    def check(name: str, passed: bool, code: str, message: str) -> None:
        checks[name] = passed
        if not passed:
            violations.append(code)
            evidence.append(message)

    check(
        "non_empty_response",
        bool(raw_response.strip()) and bool(candidate_source.strip()),
        "empty_response",
        "The model returned an empty source response.",
    )
    check(
        "no_markdown_fences",
        "```" not in raw_response,
        "markdown_fence",
        "The response contains Markdown code fences instead of plain source text.",
    )

    patch_or_multifile = bool(
        re.search(r"(?m)^(?:--- |\+\+\+ |@@ |FILE:\s|diff --git )", raw_response)
    )
    check(
        "single_source_only",
        not patch_or_multifile,
        "patch_or_multiple_files",
        "The response looks like a patch or contains multiple file sections.",
    )

    structure_ok, structure_error = _balanced_structure(candidate_source)
    check(
        "balanced_structure",
        structure_ok,
        "unbalanced_structure",
        structure_error or "The source contains unbalanced delimiters.",
    )

    expected_signature = _top_signature(baseline_source, top_function)
    candidate_signature = _top_signature(candidate_source, top_function)
    check(
        "expected_top_function",
        candidate_signature is not None,
        "missing_top_function",
        f"The generated source does not define the expected top function {top_function}.",
    )
    check(
        "unchanged_interface",
        expected_signature is not None
        and candidate_signature is not None
        and candidate_signature == expected_signature,
        "changed_top_interface",
        "The generated top-function signature differs from the previous valid source.",
    )
    check(
        "candidate_differs_from_baseline",
        _normalise_source(candidate_source) != _normalise_source(baseline_source),
        "unchanged_candidate",
        "The generated source is unchanged from the previous candidate.",
    )
    check(
        "protected_files_unchanged",
        protected_files_unchanged,
        "protected_file_modified",
        "A protected file changed before candidate acceptance.",
    )

    scrubbed_end = _scrub_comments_and_literals(candidate_source).rstrip()
    obviously_truncated = bool(scrubbed_end) and (
        scrubbed_end.endswith(("...", "\\", "=", ",", "(", "[", "{"))
        or re.search(r"(?:&&|\|\||->|::|[+\-*/%])$", scrubbed_end) is not None
    )
    check(
        "not_obviously_truncated",
        not obviously_truncated and structure_ok and candidate_signature is not None,
        "obvious_truncation",
        "The generated source appears incomplete or truncated.",
    )

    return {
        "passed": not violations,
        "failure_class": "none" if not violations else "invalid_model_output",
        "checks": checks,
        "violations": violations,
        "evidence": evidence,
        "expected_top_function": top_function,
        "expected_signature": expected_signature,
        "candidate_signature": candidate_signature,
        "baseline_hash": _sha256(baseline_source),
        "candidate_hash": _sha256(candidate_source),
    }


def validate_response_from_prompt(
    *,
    raw_response: str,
    candidate_source: str,
    user_prompt: str,
) -> dict[str, Any]:
    """Resolve the previous source and top-level contract from the repair prompt."""
    baseline_source = _extract_editable_source(user_prompt)
    top_function = _extract_top_function(user_prompt)
    if baseline_source is None or top_function is None:
        return {
            "passed": False,
            "failure_class": "invalid_model_output",
            "checks": {"prompt_contract_available": False},
            "violations": ["missing_validation_contract"],
            "evidence": [
                "The repair prompt did not expose the editable source and expected top function required for safe output validation."
            ],
            "expected_top_function": top_function,
            "expected_signature": None,
            "candidate_signature": None,
            "baseline_hash": _sha256(baseline_source or ""),
            "candidate_hash": _sha256(candidate_source),
        }
    return validate_model_output(
        raw_response=raw_response,
        candidate_source=candidate_source,
        baseline_source=baseline_source,
        top_function=top_function,
    )
