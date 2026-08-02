"""Repair diagnosis built from generic validation evidence."""

from __future__ import annotations

from agent.state import ValidationResult


def diagnose(validation: ValidationResult) -> dict[str, object]:
    return {
        "failure_class": validation.failure_class,
        "evidence": list(validation.evidence),
        "objective": "restore correctness while preserving the protected task contract",
    }
