"""Structured-search wrapper around the preserved optimisation runner.

The established validation, budget, synthesis, archive and final-selection
implementation remains byte-for-byte in :mod:`agent.optimise.runner_legacy`.
This wrapper changes only search control when structured search is enabled:

* candidates 1-3 explore distinct strategy families from the verified baseline;
* candidate 4 exploits the best explicitly refinement-eligible exploration;
* candidate 5 performs one bounded recovery or an independent baseline fallback;
* no unstructured candidate 6+ retries are created.

The temporary hook mechanism is process-local and guarded by a lock because the
legacy runner resolves its collaborators from module globals. Track-A executes
one optimisation task per process, so no concurrent search is expected.
"""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from agent.optimise import runner_legacy as _legacy
from agent.optimise.config_source import ConfigInput, ConfigSource, as_config_source
from agent.optimise.search_policy import (
    MAX_STRUCTURED_CANDIDATES,
    build_structured_search_schedule,
)
from agent.optimise.structured_exploration import (
    prepare_structured_exploration_prompt,
)
from agent.optimise.structured_tail import (
    STRUCTURED_EXPLOIT_FALLBACK_REASON,
    STRUCTURED_EXPLOIT_REASON,
    STRUCTURED_RECOVERY_FALLBACK_REASON,
    STRUCTURED_RECOVERY_REASONS,
    baseline_fallback_parent,
    prepare_structured_baseline_fallback_prompt,
    select_structured_exploitation_parent,
    select_structured_recovery_parent,
    write_structured_search_decision,
)

STRUCTURED_SEARCH_MODE = "structured_v1"
STRUCTURED_PARENT_REASON = "structured_baseline_exploration"

# Re-export the preserved implementation so existing imports and tests keep the
# same surface. run_optimisation is defined below and intentionally excluded.
for _name in dir(_legacy):
    if _name != "run_optimisation" and not _name.startswith("__"):
        globals().setdefault(_name, getattr(_legacy, _name))

_RUN_LOCK = threading.RLock()
_DIRECT_PROMPT_HOOK_NAMES = (
    "is_resource_frequency_balance_reason",
    "is_resource_recovery_reason",
    "prepare_refinement_prompt",
    "prepare_resource_frequency_balance_prompt",
    "prepare_resource_recovery_prompt",
    "prepare_tradeoff_prompt",
    "resource_frequency_balance_trigger",
    "resource_limit_recovery_trigger",
)
_LEGACY_HOOK_NAMES = (
    "REPO_ROOT",
    "evaluate_experiment",
    "generate_candidate",
    "select_refinement_parent",
    "_record",
    "_candidate_indices",
    "_status_summary",
    "_initialise",
    "_run_cosim_stage",
    "_evaluate_candidate",
    "run_candidate_cosim",
    "run_candidate_csim",
    "run_candidate_synthesis",
    "validate_ppa_candidate",
    "check_candidate_duplicate",
    "ensure_baseline_synthesis",
    *_DIRECT_PROMPT_HOOK_NAMES,
)


def structured_search_enabled(config: dict[str, Any]) -> bool:
    """Return whether this PPA config opts into the structured rescue policy."""

    policy = config.get("search_policy")
    if isinstance(policy, dict) and isinstance(policy.get("mode"), str):
        return policy["mode"] == STRUCTURED_SEARCH_MODE

    # Track-A task adapters created before this field existed are treated as an
    # explicit compatibility opt-in. Standalone legacy PPA JSON remains unchanged.
    return isinstance(config.get("track_a"), dict)


def _structured_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return an isolated config capped at the five declared search slots."""

    copied = json.loads(json.dumps(config))
    budget = copied.setdefault("budget", {})
    configured = budget.get("max_candidates", MAX_STRUCTURED_CANDIDATES)
    if isinstance(configured, bool) or not isinstance(configured, (int, float)):
        configured = MAX_STRUCTURED_CANDIDATES
    budget["max_candidates"] = min(
        max(int(configured), 0),
        MAX_STRUCTURED_CANDIDATES,
    )
    copied["search_policy"] = {"mode": STRUCTURED_SEARCH_MODE}
    return copied


def _output_dir(config: dict[str, Any]) -> Path:
    path = Path(str(config["output_dir"])).expanduser()
    return path if path.is_absolute() else Path(REPO_ROOT) / path


def _exploration_attempt(candidate_index: int) -> dict[str, Any] | None:
    return next(
        (
            attempt
            for attempt in build_structured_search_schedule(max_candidates=3)
            if attempt["candidate_index"] == candidate_index
        ),
        None,
    )


def _load_strategy(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _structured_history_ready(
    config: dict[str, Any],
    next_candidate_index: int,
) -> bool:
    """Require all preceding exploration slots to carry matching metadata."""

    if next_candidate_index not in {2, 3, 4, 5}:
        return False
    output_dir = _output_dir(config)
    schedule = build_structured_search_schedule(max_candidates=3)
    required_count = min(next_candidate_index - 1, 3)
    for attempt in schedule[:required_count]:
        index = int(attempt["candidate_index"])
        strategy = _load_strategy(
            output_dir / f"candidate_{index:03d}_strategy.json"
        )
        if not (
            strategy.get("trigger") == STRUCTURED_PARENT_REASON
            and strategy.get("phase") == "explore"
            and strategy.get("source_candidate_index") == 0
            and strategy.get("schedule_slot") == index
            and strategy.get("name") == attempt["strategy_family"]
        ):
            return False
    if next_candidate_index == 5:
        return (output_dir / "candidate_004_search_decision.json").is_file()
    return True


def _baseline_exploration_parent(
    config: dict[str, Any],
    next_candidate_index: int,
) -> dict[str, Any]:
    attempt = _exploration_attempt(next_candidate_index)
    if attempt is None:
        raise ValueError("candidate is not a structured exploration slot")
    return {
        "candidate_index": 0,
        "candidate_file": config["baseline"]["source"],
        "fully_verified": True,
        "verdict": STRUCTURED_PARENT_REASON,
        "next_candidate_index": next_candidate_index,
        "strategy_family": attempt["strategy_family"],
    }


def _prepare_next_prompt(
    config_source: ConfigSource,
    previous: dict[str, Any],
    previous_index: int,
    next_index: int,
    summary: dict[str, Any],
    parent_reason: str,
) -> None:
    """Delegate direct calls while honouring wrapper-level monkeypatches."""

    saved: dict[str, Any] = {}
    for name in _DIRECT_PROMPT_HOOK_NAMES:
        if name in globals() and hasattr(_legacy, name):
            saved[name] = getattr(_legacy, name)
            setattr(_legacy, name, globals()[name])
    try:
        _legacy._prepare_next_prompt(
            config_source,
            previous,
            previous_index,
            next_index,
            summary,
            parent_reason,
        )
    finally:
        for name, value in saved.items():
            setattr(_legacy, name, value)


@contextmanager
def _legacy_execution_hooks(
    config_source: ConfigSource,
    config: dict[str, Any],
) -> Iterator[None]:
    """Temporarily install the five-slot policy into the preserved runner."""

    del config_source
    enabled = structured_search_enabled(config)
    saved: dict[str, Any] = {}

    # Synchronise common monkeypatch points from this compatibility module to
    # the preserved implementation before adding the structured wrappers.
    for name in _LEGACY_HOOK_NAMES:
        if name in globals() and hasattr(_legacy, name):
            saved[name] = getattr(_legacy, name)
            setattr(_legacy, name, globals()[name])

    base_generate = _legacy.generate_candidate
    base_select = _legacy.select_refinement_parent
    base_prepare = _legacy._prepare_next_prompt
    base_record = _legacy._record

    def structured_generate(
        source: ConfigSource,
        candidate_index: int = 1,
        *,
        budget: Any = None,
    ) -> Path:
        attempt = _exploration_attempt(candidate_index) if enabled else None
        if attempt is not None:
            prepare_structured_exploration_prompt(
                source,
                candidate_index=candidate_index,
                strategy_family=str(attempt["strategy_family"]),
            )
        return base_generate(source, candidate_index, budget=budget)

    def structured_select(
        records: Any,
        selection: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], str] | None:
        record_list = list(records)
        indexed = [
            int(record["candidate_index"])
            for record in record_list
            if isinstance(record.get("candidate_index"), int)
        ]
        next_index = max(indexed, default=0) + 1

        if enabled and next_index in {2, 3} and _structured_history_ready(
            config,
            next_index,
        ):
            return (
                _baseline_exploration_parent(config, next_index),
                STRUCTURED_PARENT_REASON,
            )

        if enabled and next_index == 4 and _structured_history_ready(config, 4):
            exploitation = select_structured_exploitation_parent(
                record_list,
                selection,
            )
            return exploitation or baseline_fallback_parent(
                config,
                candidate_index=4,
            )

        if enabled and next_index == 5 and _structured_history_ready(config, 5):
            recovery = select_structured_recovery_parent(
                record_list,
                selection,
                base_select,
            )
            return recovery or baseline_fallback_parent(
                config,
                candidate_index=5,
            )

        return base_select(record_list, selection)

    def structured_prepare(
        source: ConfigSource,
        previous: dict[str, Any],
        previous_index: int,
        next_index: int,
        summary: dict[str, Any],
        parent_reason: str,
    ) -> None:
        attempt = _exploration_attempt(next_index) if enabled else None
        if (
            attempt is not None
            and next_index in {2, 3}
            and parent_reason == STRUCTURED_PARENT_REASON
            and previous_index == 0
        ):
            prepare_structured_exploration_prompt(
                source,
                candidate_index=next_index,
                strategy_family=str(attempt["strategy_family"]),
            )
            return

        if enabled and next_index == 4:
            if (
                parent_reason == STRUCTURED_EXPLOIT_FALLBACK_REASON
                and previous_index == 0
            ):
                prepare_structured_baseline_fallback_prompt(
                    source,
                    candidate_index=4,
                )
            elif parent_reason == STRUCTURED_EXPLOIT_REASON:
                base_prepare(
                    source,
                    previous,
                    previous_index,
                    next_index,
                    summary,
                    parent_reason,
                )
            else:
                base_prepare(
                    source,
                    previous,
                    previous_index,
                    next_index,
                    summary,
                    parent_reason,
                )
            write_structured_search_decision(
                source,
                candidate_index=4,
                phase="exploit",
                parent=previous,
                reason=parent_reason,
            )
            return

        if enabled and next_index == 5:
            if (
                parent_reason == STRUCTURED_RECOVERY_FALLBACK_REASON
                and previous_index == 0
            ):
                prepare_structured_baseline_fallback_prompt(
                    source,
                    candidate_index=5,
                )
            elif parent_reason in STRUCTURED_RECOVERY_REASONS:
                base_prepare(
                    source,
                    previous,
                    previous_index,
                    next_index,
                    summary,
                    parent_reason,
                )
            else:
                base_prepare(
                    source,
                    previous,
                    previous_index,
                    next_index,
                    summary,
                    parent_reason,
                )
            write_structured_search_decision(
                source,
                candidate_index=5,
                phase="recover",
                parent=previous,
                reason=parent_reason,
            )
            return

        base_prepare(
            source,
            previous,
            previous_index,
            next_index,
            summary,
            parent_reason,
        )

    def structured_record(
        summary: dict[str, Any],
        candidate_index: int,
    ) -> dict[str, Any] | None:
        record = base_record(summary, candidate_index)
        if not (
            enabled
            and candidate_index in {1, 2, 3, 4}
            and isinstance(record, dict)
            and record.get("verdict") == "accept_dominates_baseline"
        ):
            return record

        # Keep the persisted summary truthful, but prevent the legacy early-stop
        # branch from ending before the declared structured slots complete.
        visible = dict(record)
        visible["verdict"] = "keep_pareto_candidate"
        visible["reason"] = (
            "Candidate dominates the baseline; structured search continues until "
            "the bounded five-slot schedule completes."
        )
        visible["structured_exploration_continue"] = True
        return visible

    if enabled:
        _legacy.generate_candidate = structured_generate
        _legacy.select_refinement_parent = structured_select
        _legacy._prepare_next_prompt = structured_prepare
        _legacy._record = structured_record

    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(_legacy, name, value)


def run_optimisation(
    config_input: ConfigInput,
    *,
    status_only: bool = False,
    max_steps: int | None = None,
    budget: Any = None,
) -> Any:
    """Run the preserved optimiser with the bounded structured policy."""

    config_source = as_config_source(config_input)
    config = _load_json(config_source)
    enabled = structured_search_enabled(config)
    effective_config = _structured_config(config) if enabled else config
    effective_input: ConfigInput = effective_config if enabled else config_input

    with _RUN_LOCK:
        with _legacy_execution_hooks(
            as_config_source(effective_input),
            effective_config,
        ):
            return _legacy.run_optimisation(
                effective_input,
                status_only=status_only,
                max_steps=max_steps,
                budget=budget,
            )


def __getattr__(name: str) -> Any:
    """Delegate any unlisted compatibility attribute to the preserved runner."""

    return getattr(_legacy, name)
