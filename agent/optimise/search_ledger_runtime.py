"""Runtime enforcement for the structured-search novelty ledger.

The pure ledger lives in :mod:`agent.optimise.search_ledger`. This module adds
three guarded runtime boundaries without changing the preserved optimisation
runner:

* inspect the proposed parent/strategy branch before candidate generation;
* register the exact model-facing prompt immediately before the provider call;
* record generated source hashes and terminal evaluation verdicts.

Legacy PPA configurations bypass every guard. Installation is idempotent and
patches only the function references held by ``runner_legacy`` plus the provider
call imported by ``generate.py``. The guards self-install on direct module import
so all import paths observe the same deterministic runtime state.
"""

from __future__ import annotations

import json
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from agent.optimise import generate as generation
from agent.optimise import runner_legacy
from agent.optimise.duplicate import source_digest
from agent.optimise.search_ledger import (
    SearchLedgerError,
    inspect_branch_novelty,
    load_search_ledger,
    record_candidate_source,
    record_search_outcome,
    register_search_attempt,
)

STRUCTURED_SEARCH_MODE = "structured_v1"
NOVELTY_REJECTION_SUFFIX = "_novelty_rejection.json"
_PENDING_GENERATION: ContextVar[dict[str, Any] | None] = ContextVar(
    "structured_search_pending_generation",
    default=None,
)

_ORIGINAL_GENERATE: Any = None
_ORIGINAL_EVALUATE: Any = None
_ORIGINAL_COMPLETE: Any = None


class SearchNoveltyRejected(SearchLedgerError):
    """Raised before a repeated structured branch reaches the model provider."""


def _load_config(config_source: Any) -> dict[str, Any]:
    resolved = config_source.resolve()
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SearchLedgerError("Optimisation config must contain a JSON object")
    return value


def _structured_enabled(config: dict[str, Any]) -> bool:
    policy = config.get("search_policy")
    if isinstance(policy, dict) and isinstance(policy.get("mode"), str):
        return policy["mode"] == STRUCTURED_SEARCH_MODE
    return isinstance(config.get("track_a"), dict)


def _repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else generation.REPO_ROOT / path


def _output_dir(config: dict[str, Any]) -> Path:
    return _repo_path(str(config["output_dir"]))


def _load_optional(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _parent_index(
    output_dir: Path,
    candidate_index: int,
    strategy: dict[str, Any],
) -> int:
    decision = _load_optional(
        output_dir / f"candidate_{candidate_index:03d}_search_decision.json"
    )
    selected = decision.get("selected_parent_index")
    if isinstance(selected, int) and selected >= 0:
        return selected

    source = strategy.get("source_candidate_index")
    if isinstance(source, int) and source >= 0:
        return source

    feedback = _load_optional(
        output_dir / f"candidate_{candidate_index:03d}_feedback.json"
    )
    previous = feedback.get("previous_candidate_index")
    if isinstance(previous, int) and previous >= 0:
        return previous
    return 0


def _parent_source_path(
    config: dict[str, Any],
    output_dir: Path,
    parent_candidate_index: int,
) -> Path:
    if parent_candidate_index == 0:
        path = _repo_path(str(config["baseline"]["source"]))
    else:
        path = output_dir / f"candidate_{parent_candidate_index:03d}.cpp"
    if not path.is_file():
        raise SearchLedgerError(f"Search parent source not found: {path}")
    return path


def _search_plan(
    config: dict[str, Any],
    candidate_index: int,
) -> dict[str, Any]:
    output_dir = _output_dir(config)
    strategy = _load_optional(
        output_dir / f"candidate_{candidate_index:03d}_strategy.json"
    )
    decision = _load_optional(
        output_dir / f"candidate_{candidate_index:03d}_search_decision.json"
    )
    feedback = _load_optional(
        output_dir / f"candidate_{candidate_index:03d}_feedback.json"
    )

    parent_candidate_index = _parent_index(
        output_dir,
        candidate_index,
        strategy,
    )
    parent_path = _parent_source_path(
        config,
        output_dir,
        parent_candidate_index,
    )

    family = strategy.get("name")
    if not isinstance(family, str) or not family:
        family = decision.get("parent_reason")
    if not isinstance(family, str) or not family:
        family = feedback.get("strategy_family")
    if not isinstance(family, str) or not family:
        family = f"structured_slot_{candidate_index}"

    parameters = strategy.get("parameters")
    if not isinstance(parameters, dict):
        parameters = {}

    phase = strategy.get("phase")
    if not isinstance(phase, str) or not phase:
        phase = decision.get("phase")
    if not isinstance(phase, str) or not phase:
        phase = feedback.get("phase")
    if not isinstance(phase, str) or not phase:
        phase = None

    return {
        "candidate_index": candidate_index,
        "output_dir": output_dir,
        "parent_candidate_index": parent_candidate_index,
        "parent_source_hash": source_digest(
            parent_path.read_text(encoding="utf-8")
        ),
        "strategy_family": family,
        "parameters": parameters,
        "phase": phase,
    }


def _prompt_text(output_dir: Path, candidate_index: int) -> str | None:
    effective = output_dir / f"candidate_{candidate_index:03d}_effective_prompt.txt"
    if effective.is_file():
        return effective.read_text(encoding="utf-8")
    prompt = output_dir / f"candidate_{candidate_index:03d}_prompt.txt"
    return prompt.read_text(encoding="utf-8") if prompt.is_file() else None


def _write_rejection(
    plan: dict[str, Any],
    report: dict[str, Any],
) -> Path:
    path = plan["output_dir"] / (
        f"candidate_{plan['candidate_index']:03d}{NOVELTY_REJECTION_SUFFIX}"
    )
    path.write_text(
        json.dumps(
            {
                "candidate_index": plan["candidate_index"],
                "passed": False,
                "stage": "pre_model_novelty",
                "reason": report.get("reason"),
                "duplicate_of_candidate_index": report.get(
                    "duplicate_of_candidate_index"
                ),
                "branch_fingerprint": report.get("branch_fingerprint"),
                "effective_prompt_hash": report.get("effective_prompt_hash"),
                "strategy_family": plan["strategy_family"],
                "parameters": plan["parameters"],
                "parent_candidate_index": plan["parent_candidate_index"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _inspect_branch_before_generation(plan: dict[str, Any]) -> None:
    report = inspect_branch_novelty(
        plan["output_dir"],
        candidate_index=plan["candidate_index"],
        parent_source_hash=plan["parent_source_hash"],
        strategy_family=plan["strategy_family"],
        parameters=plan["parameters"],
        effective_prompt=None,
    )
    if report["allowed"] is True:
        return
    rejection = _write_rejection(plan, report)
    raise SearchNoveltyRejected(
        f"Structured search branch rejected before generation: "
        f"{report['reason']} (see {rejection})"
    )


def _register_exact_prompt(
    plan: dict[str, Any],
    effective_prompt: str,
) -> dict[str, Any]:
    record = register_search_attempt(
        plan["output_dir"],
        candidate_index=plan["candidate_index"],
        parent_candidate_index=plan["parent_candidate_index"],
        parent_source_hash=plan["parent_source_hash"],
        strategy_family=plan["strategy_family"],
        parameters=plan["parameters"],
        effective_prompt=effective_prompt,
        phase=plan["phase"],
    )
    if record.get("registration_allowed", True) is True:
        return record

    report = {
        "reason": record.get("registration_rejection_reason"),
        "duplicate_of_candidate_index": record.get(
            "duplicate_of_candidate_index"
        ),
        "branch_fingerprint": record.get("branch_fingerprint"),
        "effective_prompt_hash": record.get("effective_prompt_hash"),
    }
    rejection = _write_rejection(plan, report)
    raise SearchNoveltyRejected(
        f"Structured search prompt rejected before provider call: "
        f"{report['reason']} (see {rejection})"
    )


def _attempt_registered(output_dir: Path, candidate_index: int) -> bool:
    return any(
        item.get("candidate_index") == candidate_index
        for item in load_search_ledger(output_dir).get("attempts", [])
    )


def _guarded_complete(*args: Any, **kwargs: Any) -> Any:
    pending = _PENDING_GENERATION.get()
    if pending is None:
        return _ORIGINAL_COMPLETE(*args, **kwargs)

    user_prompt = kwargs.get("user_prompt")
    if not isinstance(user_prompt, str):
        raise SearchLedgerError("Structured provider call is missing user_prompt")
    _register_exact_prompt(pending, user_prompt)
    return _ORIGINAL_COMPLETE(*args, **kwargs)


def _guarded_generate(
    config_source: Any,
    candidate_index: int = 1,
    *,
    budget: Any = None,
) -> Path:
    config = _load_config(config_source)
    if not _structured_enabled(config):
        return _ORIGINAL_GENERATE(
            config_source,
            candidate_index,
            budget=budget,
        )

    plan = _search_plan(config, candidate_index)
    _inspect_branch_before_generation(plan)
    token = _PENDING_GENERATION.set(plan)
    try:
        candidate_path = _ORIGINAL_GENERATE(
            config_source,
            candidate_index,
            budget=budget,
        )
    finally:
        _PENDING_GENERATION.reset(token)

    if not _attempt_registered(plan["output_dir"], candidate_index):
        # Controlled deterministic generation does not call the model provider.
        # Register it after generation using the best persisted prompt evidence.
        prompt = _prompt_text(plan["output_dir"], candidate_index)
        _register_exact_prompt(plan, prompt or "")

    record_candidate_source(
        plan["output_dir"],
        candidate_index=candidate_index,
        source_text=candidate_path.read_text(encoding="utf-8"),
    )
    return candidate_path


def _guarded_evaluate(config_source: Any) -> dict[str, Any]:
    summary = _ORIGINAL_EVALUATE(config_source)
    config = _load_config(config_source)
    if not _structured_enabled(config):
        return summary

    output_dir = _output_dir(config)
    registered = {
        int(item["candidate_index"])
        for item in load_search_ledger(output_dir).get("attempts", [])
        if isinstance(item.get("candidate_index"), int)
    }
    for record in summary.get("candidates", []):
        index = record.get("candidate_index")
        verdict = record.get("verdict")
        if (
            not isinstance(index, int)
            or index not in registered
            or not isinstance(verdict, str)
            or verdict in {"incomplete", "awaiting_cosim"}
        ):
            continue
        record_search_outcome(
            output_dir,
            candidate_index=index,
            verdict=verdict,
        )
    return summary


def install_structured_search_ledger_runtime() -> None:
    """Install idempotent runtime guards around the preserved PPA functions."""

    global _ORIGINAL_GENERATE, _ORIGINAL_EVALUATE, _ORIGINAL_COMPLETE

    marker = "_structured_search_ledger_runtime_installed"
    if getattr(runner_legacy, marker, False):
        return

    _ORIGINAL_GENERATE = runner_legacy.generate_candidate
    _ORIGINAL_EVALUATE = runner_legacy.evaluate_experiment
    _ORIGINAL_COMPLETE = generation.complete

    runner_legacy.generate_candidate = _guarded_generate
    runner_legacy.evaluate_experiment = _guarded_evaluate
    generation.complete = _guarded_complete
    setattr(runner_legacy, marker, True)


# Direct imports of this module must observe the same guarded state as imports
# that arrive through ``search_policy``. Repeated calls remain no-ops.
install_structured_search_ledger_runtime()
