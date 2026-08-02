#!/usr/bin/env python3

"""Shared state types for the unified repair and optimisation controller."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class BudgetState:
    max_iterations: int
    max_csim_calls: int
    max_cosim_calls: int
    max_synthesis_calls: int
    max_model_calls: int
    iterations_used: int = 0
    csim_calls_used: int = 0
    cosim_calls_used: int = 0
    synthesis_calls_used: int = 0
    model_calls_used: int = 0

    @classmethod
    def from_manifest(cls, budgets: dict[str, Any]) -> "BudgetState":
        return cls(
            max_iterations=int(budgets["max_iterations"]),
            max_csim_calls=int(budgets["max_csim_calls"]),
            max_cosim_calls=int(budgets["max_cosim_calls"]),
            max_synthesis_calls=int(budgets["max_synthesis_calls"]),
            max_model_calls=int(budgets["max_model_calls"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ValidationResult:
    """Normalised result of a host, CSim, co-sim or static validation step."""

    passed: bool
    failure_class: str
    return_code: int
    evidence: tuple[str, ...] | list[str] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evidence"] = list(self.evidence)
        return data


@dataclass(frozen=True)
class SynthesisMetrics:
    """Normalised metrics used by benchmark-independent PPA evaluation.

    ``None`` is allowed because some failed or partial synthesis reports may not
    contain every metric. Such records are not considered comparable by the
    Pareto evaluator.
    """

    latency_cycles: int | None
    interval_cycles: int | None
    clock_period_ns: float | None
    lut: int | None
    ff: int | None
    dsp: int | None
    bram: int | None

    @classmethod
    def from_mapping(cls, metrics: dict[str, Any]) -> "SynthesisMetrics":
        return cls(
            latency_cycles=metrics.get("latency_cycles", metrics.get("latency_best_cycles")),
            interval_cycles=metrics.get("interval_cycles", metrics.get("interval_min_cycles")),
            clock_period_ns=metrics.get("clock_period_ns"),
            lut=metrics.get("lut", metrics.get("resources_lut_used")),
            ff=metrics.get("ff", metrics.get("resources_ff_used")),
            dsp=metrics.get("dsp", metrics.get("resources_dsp_used")),
            bram=metrics.get("bram", metrics.get("resources_bram_used")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrajectoryEvent:
    step: int
    stage: str
    status: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentResult:
    task_id: str
    success: bool
    status: str
    termination_reason: str
    output_dir: str
    trajectory: list[TrajectoryEvent] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "success": self.success,
            "status": self.status,
            "termination_reason": self.termination_reason,
            "output_dir": self.output_dir,
            "trajectory": [event.to_dict() for event in self.trajectory],
        }
