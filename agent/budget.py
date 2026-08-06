"""Authoritative task-wide budget accounting for model and Vitis calls."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


RESOURCE_LIMITS = {
    "iterations": "max_iterations",
    "model_calls": "max_model_calls",
    "csim_calls": "max_csim_calls",
    "cosim_calls": "max_cosim_calls",
    "synthesis_calls": "max_synthesis_calls",
}
TRACK_A_DEFAULT_CREDIT_COSTS = {
    "csim": 1,
    "synthesis": 4,
    "cosim": 20,
}
TOOL_RESOURCE_TO_TRACK_A_KIND = {
    "csim_calls": "csim",
    "synthesis_calls": "synthesis",
    "cosim_calls": "cosim",
}


class BudgetExceeded(RuntimeError):
    """Raised before an operation that would exceed the task budget."""


@dataclass
class BudgetState:
    """Track and enforce one shared budget across the complete agent run."""

    max_iterations: int
    max_model_calls: int
    max_csim_calls: int
    max_cosim_calls: int
    max_synthesis_calls: int
    max_total_tokens: int | None = None
    max_track_a_credits: int | None = None
    track_a_credit_costs: dict[str, int] = field(
        default_factory=lambda: dict(TRACK_A_DEFAULT_CREDIT_COSTS)
    )
    requires_cosim: bool = True

    iterations_used: int = 0
    model_calls_used: int = 0
    csim_calls_used: int = 0
    cosim_calls_used: int = 0
    synthesis_calls_used: int = 0
    input_tokens_used: int = 0
    output_tokens_used: int = 0
    track_a_credits_used: int = 0
    stop_reason: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_manifest(cls, budgets: Mapping[str, Any]) -> "BudgetState":
        raw_costs = budgets.get("track_a_credit_costs")
        costs = dict(TRACK_A_DEFAULT_CREDIT_COSTS)
        if raw_costs is not None:
            if not isinstance(raw_costs, Mapping):
                raise ValueError("budgets.track_a_credit_costs must be an object")
            for kind in TRACK_A_DEFAULT_CREDIT_COSTS:
                value = raw_costs.get(kind)
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ValueError(
                        f"budgets.track_a_credit_costs.{kind} must be a positive integer"
                    )
                costs[kind] = value

        raw_credit_budget = budgets.get("track_a_credit_budget")
        if raw_credit_budget is not None and (
            isinstance(raw_credit_budget, bool)
            or not isinstance(raw_credit_budget, int)
            or raw_credit_budget <= 0
        ):
            raise ValueError(
                "budgets.track_a_credit_budget must be null or a positive integer"
            )

        raw_requires_cosim = budgets.get("requires_cosim", True)
        if not isinstance(raw_requires_cosim, bool):
            raise ValueError("budgets.requires_cosim must be a boolean")

        return cls(
            max_iterations=int(budgets["max_iterations"]),
            max_model_calls=int(budgets["max_model_calls"]),
            max_csim_calls=int(budgets["max_csim_calls"]),
            max_cosim_calls=int(budgets["max_cosim_calls"]),
            max_synthesis_calls=int(budgets["max_synthesis_calls"]),
            max_total_tokens=(
                int(budgets["max_total_tokens"])
                if budgets.get("max_total_tokens") is not None
                else None
            ),
            max_track_a_credits=(
                int(raw_credit_budget) if raw_credit_budget is not None else None
            ),
            track_a_credit_costs=costs,
            requires_cosim=raw_requires_cosim,
        )

    def _limit(self, resource: str) -> int:
        try:
            return int(getattr(self, RESOURCE_LIMITS[resource]))
        except KeyError as error:
            raise ValueError(f"Unknown budget resource: {resource}") from error

    def _used(self, resource: str) -> int:
        if resource not in RESOURCE_LIMITS:
            raise ValueError(f"Unknown budget resource: {resource}")
        return int(getattr(self, f"{resource}_used"))

    def remaining(self, resource: str) -> int:
        return max(0, self._limit(resource) - self._used(resource))

    @property
    def total_tokens_used(self) -> int:
        return self.input_tokens_used + self.output_tokens_used

    @property
    def total_tokens_remaining(self) -> int | None:
        if self.max_total_tokens is None:
            return None
        return max(0, self.max_total_tokens - self.total_tokens_used)

    @property
    def track_a_credits_remaining(self) -> int | None:
        if self.max_track_a_credits is None:
            return None
        return max(0, self.max_track_a_credits - self.track_a_credits_used)

    def can_consume(self, resource: str, amount: int = 1, *, reserve: int = 0) -> bool:
        if amount < 0 or reserve < 0:
            raise ValueError("Budget amounts and reserves must be non-negative")
        return self.remaining(resource) >= amount + reserve

    def require(self, resource: str, amount: int = 1, *, reserve: int = 0) -> None:
        if not self.can_consume(resource, amount, reserve=reserve):
            reason = f"{resource}_budget_exhausted"
            self.set_stop_reason(reason)
            raise BudgetExceeded(
                f"Cannot consume {amount} {resource}; "
                f"remaining={self.remaining(resource)}, reserve={reserve}"
            )
        track_a_kind = TOOL_RESOURCE_TO_TRACK_A_KIND.get(resource)
        if track_a_kind is not None:
            self.require_track_a(track_a_kind, amount)

    def track_a_credit_cost(self, kind: str, amount: int = 1) -> int:
        if kind not in self.track_a_credit_costs:
            raise ValueError(f"Unknown Track-A tool kind: {kind}")
        if amount < 0:
            raise ValueError("Track-A credit amount must be non-negative")
        return int(self.track_a_credit_costs[kind]) * amount

    def can_afford_track_a(self, kind: str, amount: int = 1, *, reserve: int = 0) -> bool:
        if reserve < 0:
            raise ValueError("Track-A credit reserve must be non-negative")
        remaining = self.track_a_credits_remaining
        if remaining is None:
            return True
        return remaining >= self.track_a_credit_cost(kind, amount) + reserve

    def require_track_a(self, kind: str, amount: int = 1, *, reserve: int = 0) -> None:
        if not self.can_afford_track_a(kind, amount, reserve=reserve):
            self.set_stop_reason("track_a_credit_budget_exhausted")
            remaining = self.track_a_credits_remaining
            cost = self.track_a_credit_cost(kind, amount)
            raise BudgetExceeded(
                f"Cannot spend {cost} Track-A credits on {kind}; "
                f"remaining={remaining}, reserve={reserve}"
            )

    def charge(
        self,
        resource: str,
        amount: int = 1,
        *,
        stage: str,
        success: bool | None = None,
        timed_out: bool = False,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.require(resource, amount)
        setattr(self, f"{resource}_used", self._used(resource) + amount)
        event: dict[str, Any] = {
            "resource": resource,
            "amount": amount,
            "stage": stage,
            "success": success,
            "timed_out": timed_out,
        }
        if details:
            event["details"] = dict(details)
        self.events.append(event)

    def _charge_tool(
        self,
        resource: str,
        track_a_kind: str,
        *,
        stage: str,
        success: bool | None = None,
        timed_out: bool = False,
    ) -> None:
        """Atomically charge a call counter and the weighted Track-A ledger."""

        self.require(resource)
        setattr(self, f"{resource}_used", self._used(resource) + 1)

        event: dict[str, Any] = {
            "resource": resource,
            "amount": 1,
            "stage": stage,
            "success": success,
            "timed_out": timed_out,
        }
        if self.max_track_a_credits is not None:
            cost = self.track_a_credit_cost(track_a_kind)
            self.track_a_credits_used += cost
            event["track_a"] = {
                "kind": track_a_kind,
                "credit_cost": cost,
                "credits_spent_after": self.track_a_credits_used,
                "credits_remaining_after": self.track_a_credits_remaining,
            }
        self.events.append(event)

    def charge_iteration(self, *, stage: str) -> None:
        self.charge("iterations", stage=stage)

    def charge_model_call(self, *, stage: str) -> None:
        if self.max_total_tokens is not None and self.total_tokens_remaining == 0:
            self.set_stop_reason("token_budget_exhausted")
            raise BudgetExceeded("Cannot call model; total token budget is exhausted")
        self.charge("model_calls", stage=stage)

    def charge_csim(
        self,
        *,
        stage: str,
        success: bool | None = None,
        timed_out: bool = False,
    ) -> None:
        self._charge_tool(
            "csim_calls",
            "csim",
            stage=stage,
            success=success,
            timed_out=timed_out,
        )

    def charge_cosim(
        self,
        *,
        stage: str,
        success: bool | None = None,
        timed_out: bool = False,
    ) -> None:
        self._charge_tool(
            "cosim_calls",
            "cosim",
            stage=stage,
            success=success,
            timed_out=timed_out,
        )

    def charge_synthesis(
        self,
        *,
        stage: str,
        success: bool | None = None,
        timed_out: bool = False,
    ) -> None:
        self._charge_tool(
            "synthesis_calls",
            "synthesis",
            stage=stage,
            success=success,
            timed_out=timed_out,
        )

    def update_last_event(
        self,
        *,
        success: bool | None = None,
        timed_out: bool | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        if not self.events:
            raise RuntimeError("No budget event is available to update")
        event = self.events[-1]
        if success is not None:
            event["success"] = success
        if timed_out is not None:
            event["timed_out"] = timed_out
        if details:
            event.setdefault("details", {}).update(dict(details))

    def record_model_tokens(
        self,
        *,
        input_tokens: int | None,
        output_tokens: int | None,
        stage: str,
    ) -> None:
        input_value = 0 if input_tokens is None else input_tokens
        output_value = 0 if output_tokens is None else output_tokens
        for name, value in (("input_tokens", input_value), ("output_tokens", output_value)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer or null")

        self.input_tokens_used += input_value
        self.output_tokens_used += output_value
        self.events.append(
            {
                "resource": "tokens",
                "stage": stage,
                "input_tokens": input_value,
                "output_tokens": output_value,
                "total_tokens": input_value + output_value,
            }
        )
        if self.max_total_tokens is not None and self.total_tokens_used > self.max_total_tokens:
            self.set_stop_reason("token_budget_exhausted")
            raise BudgetExceeded(
                "Model response exceeded the total token budget: "
                f"used={self.total_tokens_used}, limit={self.max_total_tokens}"
            )

    def can_generate_candidate(
        self,
        *,
        reserve_csim: int = 1,
        reserve_synthesis: int = 1,
        reserve_cosim: int = 0,
    ) -> bool:
        effective_cosim_reserve = reserve_cosim if self.requires_cosim else 0
        token_available = self.max_total_tokens is None or self.total_tokens_remaining > 0
        call_capacity = all(
            (
                self.can_consume("iterations"),
                self.can_consume("model_calls"),
                self.can_consume("csim_calls", reserve=reserve_csim - 1)
                if reserve_csim > 0
                else True,
                self.can_consume("synthesis_calls", reserve=reserve_synthesis - 1)
                if reserve_synthesis > 0
                else True,
                self.can_consume("cosim_calls", reserve=effective_cosim_reserve - 1)
                if effective_cosim_reserve > 0
                else True,
                token_available,
            )
        )
        if not call_capacity:
            return False

        remaining = self.track_a_credits_remaining
        if remaining is None:
            return True
        required_credits = (
            self.track_a_credit_cost("csim", reserve_csim)
            + self.track_a_credit_cost("synthesis", reserve_synthesis)
            + self.track_a_credit_cost("cosim", effective_cosim_reserve)
        )
        return remaining >= required_credits

    def set_stop_reason(self, reason: str, *, overwrite: bool = False) -> None:
        if overwrite or self.stop_reason is None:
            self.stop_reason = reason

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "initial": {
                "max_iterations": self.max_iterations,
                "max_model_calls": self.max_model_calls,
                "max_csim_calls": self.max_csim_calls,
                "max_cosim_calls": self.max_cosim_calls,
                "max_synthesis_calls": self.max_synthesis_calls,
                "max_total_tokens": self.max_total_tokens,
                "requires_cosim": self.requires_cosim,
            },
            "consumed": {
                "iterations": self.iterations_used,
                "model_calls": self.model_calls_used,
                "csim_calls": self.csim_calls_used,
                "cosim_calls": self.cosim_calls_used,
                "synthesis_calls": self.synthesis_calls_used,
                "input_tokens": self.input_tokens_used,
                "output_tokens": self.output_tokens_used,
                "total_tokens": self.total_tokens_used,
            },
            "remaining": {
                "iterations": self.remaining("iterations"),
                "model_calls": self.remaining("model_calls"),
                "csim_calls": self.remaining("csim_calls"),
                "cosim_calls": self.remaining("cosim_calls"),
                "synthesis_calls": self.remaining("synthesis_calls"),
                "total_tokens": self.total_tokens_remaining,
            },
            "track_a": {
                "enabled": self.max_track_a_credits is not None,
                "credit_budget": self.max_track_a_credits,
                "credit_costs": dict(self.track_a_credit_costs),
                "credits_spent": self.track_a_credits_used,
                "credits_remaining": self.track_a_credits_remaining,
            },
            "stop_reason": self.stop_reason,
            "events": list(self.events),
        }

    def write_summary(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.summary(), indent=2) + "\n", encoding="utf-8")
        return path
