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
