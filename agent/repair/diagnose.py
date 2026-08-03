"""Structured repair diagnoses built from classified validation evidence."""

from __future__ import annotations

import re
from pathlib import PurePath
from typing import Any, Iterable, Sequence

from agent.state import ValidationResult

_SOURCE_LOCATION = re.compile(
    r"(?P<file>(?:[A-Za-z]:)?[^\s:()]+?\.(?:c|cc|cpp|cxx|h|hh|hpp))"
    r":(?P<line>\d+)(?::\d+)?",
    re.IGNORECASE,
)

_SUMMARIES = {
    "none": "The current validation stage passed; any previous failed-stage diagnosis remains the repair target.",
    "syntax_or_compile": "The source does not compile because the compiler reported a syntax or type error.",
    "compile": "The source does not compile because the compiler reported an error.",
    "missing_header": "Compilation failed because a required header could not be found.",
    "top_function_mismatch": "The configured top function is missing or does not match the required top-level contract.",
    "linkage_or_interface": "Compilation or linking failed because declarations, symbols, or interfaces do not agree.",
    "functional_mismatch": "The design compiled but produced output that differs from the expected behaviour.",
    "functional": "The design compiled but produced output that differs from the expected behaviour.",
    "numerical_tolerance": "The design output exceeded the permitted numerical tolerance.",
    "out_of_bounds": "Validation detected an invalid or out-of-bounds memory access.",
    "csim_timeout": "C simulation did not complete within the configured timeout.",
    "timeout": "Validation did not complete within the configured timeout.",
    "synthesis_unsupported_construct": "Synthesis rejected a construct that is not supported in hardware.",
    "synthesis_timeout": "Synthesis did not complete within the configured timeout.",
    "stream_deadlock": "A stream or dataflow process made no progress because of blocked communication.",
    "cosim_mismatch": "C/RTL co-simulation completed but the RTL output did not match the reference behaviour.",
    "cosim_failed": "C/RTL co-simulation failed before a more specific cause could be established.",
    "cosim_compile": "C/RTL co-simulation could not compile or elaborate the generated RTL test environment.",
    "cosim_deadlock": "C/RTL co-simulation made no progress and appears to be deadlocked.",
    "cosim_timeout": "C/RTL co-simulation did not complete within the configured timeout.",
    "tool_report_missing": "The tool completed without producing an expected report.",
    "report_parse": "A generated tool report could not be parsed.",
    "model_generation_error": "The model call failed before a usable repaired source was produced.",
    "scope_violation": "The repair changed a file outside the editable scope.",
    "protected_file_modified": "The repair changed a protected source, header, testbench, or build file.",
    "unknown": "Validation failed, but the available evidence is insufficient for a more specific diagnosis.",
}

_SOURCE_DEFECT_CLASSES = {
    "syntax_or_compile",
    "compile",
    "missing_header",
    "top_function_mismatch",
    "linkage_or_interface",
    "functional_mismatch",
    "functional",
    "numerical_tolerance",
    "out_of_bounds",
    "synthesis_unsupported_construct",
    "stream_deadlock",
    "cosim_mismatch",
    "cosim_failed",
    "cosim_compile",
    "cosim_deadlock",
}


def _unique(values: Iterable[Any]) -> list[Any]:
    result: list[Any] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _normalise_file(
    value: str,
    configured_files: Sequence[str],
) -> str:
    normalised = value.replace("\\", "/")
    for configured in configured_files:
        candidate = configured.replace("\\", "/")
        if normalised == candidate or normalised.endswith("/" + candidate):
            return candidate

    basename = PurePath(normalised).name
    matching = [
        item.replace("\\", "/")
        for item in configured_files
        if PurePath(item).name == basename
    ]
    if len(matching) == 1:
        return matching[0]
    return normalised


def _locations(
    evidence: Sequence[str],
    configured_files: Sequence[str],
) -> list[dict[str, Any]]:
    locations: list[dict[str, Any]] = []
    for line in evidence:
        for match in _SOURCE_LOCATION.finditer(str(line)):
            location = {
                "file": _normalise_file(match.group("file"), configured_files),
                "line": int(match.group("line")),
            }
            if location not in locations:
                locations.append(location)
    return locations


def _constraints(
    *,
    editable_files: Sequence[str],
    protected_files: Sequence[str],
    top_function: str | None,
    repair_constraints: Sequence[str],
) -> list[str]:
    constraints = [str(item) for item in repair_constraints if str(item).strip()]
    if top_function:
        constraints.append(f"Preserve the {top_function} function name and signature.")
    if editable_files:
        constraints.append("Modify only: " + ", ".join(editable_files) + ".")
    if protected_files:
        constraints.append("Do not modify protected files: " + ", ".join(protected_files) + ".")
    constraints.append("Do not weaken, remove, or bypass the supplied validation testbench.")
    return _unique(constraints)


def build_diagnosis(
    *,
    stage: str,
    failure_class: str,
    evidence: Sequence[str],
    editable_files: Sequence[str] = (),
    protected_files: Sequence[str] = (),
    top_function: str | None = None,
    repair_constraints: Sequence[str] = (),
) -> dict[str, object]:
    """Create the stable FPT-502 diagnosis object from concise evidence."""
    concise_evidence = _unique(str(item).strip() for item in evidence if str(item).strip())
    configured = [*editable_files, *protected_files]
    locations = _locations(concise_evidence, configured)
    suspected_files = _unique(location["file"] for location in locations)

    if failure_class in _SOURCE_DEFECT_CLASSES and editable_files:
        if not suspected_files or all(item in protected_files for item in suspected_files):
            suspected_files = _unique([*editable_files, *suspected_files])

    return {
        "stage": stage,
        "failure_class": failure_class,
        "summary": _SUMMARIES.get(failure_class, _SUMMARIES["unknown"]),
        "suspected_files": suspected_files,
        "suspected_lines": _unique(location["line"] for location in locations),
        "suspected_locations": locations,
        "evidence": concise_evidence,
        "repair_constraints": _constraints(
            editable_files=editable_files,
            protected_files=protected_files,
            top_function=top_function,
            repair_constraints=repair_constraints,
        ),
    }


def diagnose(
    validation: ValidationResult,
    *,
    stage: str = "host_validation",
    editable_files: Sequence[str] = (),
    protected_files: Sequence[str] = (),
    top_function: str | None = None,
    repair_constraints: Sequence[str] = (),
) -> dict[str, object]:
    return build_diagnosis(
        stage=stage,
        failure_class=validation.failure_class,
        evidence=list(validation.evidence),
        editable_files=editable_files,
        protected_files=protected_files,
        top_function=top_function,
        repair_constraints=repair_constraints,
    )


def format_diagnosis(diagnosis: dict[str, object]) -> str:
    """Render a compact diagnosis for a model prompt without dumping raw logs."""
    files = diagnosis.get("suspected_files") or []
    lines = diagnosis.get("suspected_lines") or []
    evidence = diagnosis.get("evidence") or []
    constraints = diagnosis.get("repair_constraints") or []

    def bullets(values: object, empty: str) -> str:
        sequence = values if isinstance(values, list) else []
        return "\n".join(f"- {item}" for item in sequence) if sequence else f"- {empty}"

    return (
        f"Stage: {diagnosis.get('stage', 'unknown')}\n"
        f"Failure class: {diagnosis.get('failure_class', 'unknown')}\n"
        f"Summary: {diagnosis.get('summary', _SUMMARIES['unknown'])}\n"
        "Suspected files:\n"
        f"{bullets(files, 'No file location was identified.')}\n"
        f"Suspected lines: {', '.join(str(item) for item in lines) if lines else 'none identified'}\n"
        "Evidence:\n"
        f"{bullets(evidence, 'No concise evidence was extracted.')}\n"
        "Repair constraints:\n"
        f"{bullets(constraints, 'Preserve the supplied task contract.')}"
    )
