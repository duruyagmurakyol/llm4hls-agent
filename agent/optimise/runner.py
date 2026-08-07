"""Bounded structured-search wrapper around the preserved PPA runner.

The validation, budget, synthesis, archive and final-selection implementation
remains in :mod:`agent.optimise.runner_legacy`.  This module changes only search
control when ``structured_v1`` is enabled:

* C1-C3 explore distinct diagnosis-selected layer-one strategy families from the verified baseline;
* C4 exploits the best explicitly refinement-eligible exploration;
* C5 performs one bounded recovery or an independent baseline fallback;
* the candidate budget is capped at five, preventing unstructured C6+ retries.

All temporary hooks are process-local, protected by a re-entrant lock, and
restored after every call.
"""

from __future__ import annotations

import json
import shutil
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from agent.optimise import runner_legacy as _legacy
from agent.optimise.config_source import ConfigInput, ConfigSource, as_config_source
from agent.optimise.duplicate import normalise_source
from agent.optimise.refinement_strategy import check_strategy_compliance
from agent.optimise.search_policy import (
    DEFAULT_EXPLORATION_STRATEGY_FAMILIES,
    MAX_STRUCTURED_CANDIDATES,
    build_structured_search_schedule,
)
from agent.optimise.strategy_selector import resolve_exploration_strategy_families
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
STRUCTURED_GENERATION_RETRIES = 1
STRUCTURED_POST_SYNTHESIS_RETRIES = 1
STRUCTURED_POST_SYNTHESIS_RETRY_VERDICTS = {
    "reject_no_change",
    "reject_no_objective_gain",
    "reject_resource_limits",
}

# Re-export the preserved implementation so existing imports keep the same
# surface.  The two functions defined below intentionally replace the legacy
# prompt dispatcher and optimisation entry point.
for _name in dir(_legacy):
    if _name not in {"run_optimisation", "_prepare_next_prompt"} and not _name.startswith("__"):
        globals().setdefault(_name, getattr(_legacy, _name))

_RUN_LOCK = threading.RLock()
_ORIGINAL_LEGACY_PREPARE = _legacy._prepare_next_prompt

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
    """Return whether a PPA config opts into the structured rescue policy."""

    policy = config.get("search_policy")
    if isinstance(policy, dict) and isinstance(policy.get("mode"), str):
        return policy["mode"] == STRUCTURED_SEARCH_MODE

    # Track-A adapters created before this field existed are an explicit
    # compatibility opt-in. Standalone legacy PPA JSON remains unchanged.
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


def _exploration_strategy_families(
    config: dict[str, Any] | None = None,
) -> tuple[str, str, str]:
    """Return the immutable per-run exploration plan or the historical default."""

    if config is None:
        return DEFAULT_EXPLORATION_STRATEGY_FAMILIES
    return resolve_exploration_strategy_families(_output_dir(config))


def _exploration_attempt(
    candidate_index: int,
    config: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    families = _exploration_strategy_families(config)
    return next(
        (
            attempt
            for attempt in build_structured_search_schedule(
                max_candidates=3,
                exploration_strategy_families=families,
            )
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


def _implementation_parent_source(
    config: dict[str, Any],
    output_dir: Path,
    strategy: dict[str, Any],
) -> str | None:
    """Load the exact source that the strategy declares as its implementation parent."""

    source_index = strategy.get("source_candidate_index")
    if isinstance(source_index, int) and source_index > 0:
        parent_path = output_dir / f"candidate_{source_index:03d}.cpp"
    else:
        baseline_value = (config.get("baseline") or {}).get("source")
        if not isinstance(baseline_value, str) or not baseline_value:
            return None
        parent_path = Path(baseline_value)
        if not parent_path.is_absolute():
            parent_path = Path(REPO_ROOT) / parent_path

    if not parent_path.is_file():
        return None
    return parent_path.read_text(encoding="utf-8")


def _structured_generation_retry_reason(
    config: dict[str, Any],
    output_dir: Path,
    candidate_index: int,
) -> str | None:
    """Return why one structured generation should receive a same-slot retry."""

    strategy = _load_strategy(
        output_dir / f"candidate_{candidate_index:03d}_strategy.json"
    )
    if strategy.get("compliance_mode") != "advisory":
        return None

    candidate_path = output_dir / f"candidate_{candidate_index:03d}.cpp"
    if not candidate_path.is_file():
        return None

    parent_source = _implementation_parent_source(config, output_dir, strategy)
    if parent_source is None:
        return None

    candidate_source = candidate_path.read_text(encoding="utf-8")
    if normalise_source(candidate_source) == normalise_source(parent_source):
        return "no_semantic_change"

    compliance = check_strategy_compliance(
        candidate_source,
        strategy,
        baseline=parent_source,
    )
    if compliance.get("required") is True and compliance.get("passed") is False:
        reason = compliance.get("reason")
        return str(reason) if reason else "strategy_not_realised"
    return None


def _structured_retry_suffix(strategy: dict[str, Any], reason: str) -> str:
    required = strategy.get("required_changes") or []
    required_text = "\n".join(f"- {item}" for item in required)
    family = str(strategy.get("name") or "assigned_strategy")

    family_hint = ""
    if family == "bounded_unroll":
        family_hint = (
            "\n- Use an explicit #pragma HLS UNROLL factor=2 or factor=4 on the "
            "selected real loop when legal; comments or unchanged source are invalid."
        )
    elif family == "memory_parallelism":
        family_hint = (
            "\n- A bounded local row/tile buffer is allowed. Bank or partition the "
            "local buffer when useful; do not completely partition a top-level interface array."
        )
    elif family == "buffered_parallelism":
        family_hint = (
            "\n- Realise both halves of this family: introduce a bounded reused local "
            "buffer/tile with bounded banking, and apply a matching explicit UNROLL "
            "factor from the allowed set to the independent consumer loop."
        )
    elif family == "sliding_window_reuse":
        family_hint = (
            "\n- Introduce an actual bounded shift/window/line-buffer reuse structure; "
            "do not merely add a pragma to the original repeated memory accesses."
        )
    elif family == "dataflow_pipeline":
        family_hint = (
            "\n- Create genuine producer/consumer stages and add #pragma HLS DATAFLOW "
            "at their enclosing level; a DATAFLOW pragma on unchanged monolithic code is invalid."
        )

    return (
        "\n\nCORRECTIVE RETRY FOR THIS SAME CANDIDATE SLOT:\n"
        f"- The previous response was rejected before Vitis because: {reason}.\n"
        f"- Strategy family remains: {family}.\n"
        "- Do not return the implementation parent unchanged.\n"
        "- Make one material executable or HLS-directive change that realises this family.\n"
        "- Keep the change focused and preserve correctness, interfaces and bounds.\n"
        "Required strategy changes include:\n"
        + (required_text or "- Realise the assigned strategy with observable source evidence.")
        + family_hint
        + "\n- Return only the complete compilable C++ source file."
    )


def _preserve_generation_attempt(
    output_dir: Path,
    candidate_index: int,
    attempt_number: int,
) -> dict[str, Any]:
    """Preserve first-attempt artefacts before a same-slot model retry overwrites them."""

    prefix = f"candidate_{candidate_index:03d}"
    mappings = {
        output_dir / f"{prefix}.cpp": output_dir / f"{prefix}_generation_attempt_{attempt_number}.cpp",
        output_dir / f"{prefix}_model_response.txt": output_dir / f"{prefix}_generation_attempt_{attempt_number}_model_response.txt",
        output_dir / f"{prefix}_model_metadata.json": output_dir / f"{prefix}_generation_attempt_{attempt_number}_model_metadata.json",
        output_dir / f"{prefix}_effective_prompt.txt": output_dir / f"{prefix}_generation_attempt_{attempt_number}_effective_prompt.txt",
    }
    for source, destination in mappings.items():
        if source.is_file():
            destination.write_bytes(source.read_bytes())

    metadata_path = mappings[output_dir / f"{prefix}_model_metadata.json"]
    return _load_strategy(metadata_path)


def _merge_generation_retry_metadata(
    output_dir: Path,
    candidate_index: int,
    first_metadata: dict[str, Any],
    retry_reason: str,
) -> None:
    metadata_path = output_dir / f"candidate_{candidate_index:03d}_model_metadata.json"
    second_metadata = _load_strategy(metadata_path)
    if not second_metadata:
        return

    attempts = [dict(first_metadata), dict(second_metadata)]
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        values = [item.get(key) for item in attempts]
        numeric = [
            int(value)
            for value in values
            if isinstance(value, int) and not isinstance(value, bool)
        ]
        second_metadata[key] = sum(numeric)

    latencies = [
        float(item["latency_seconds"])
        for item in attempts
        if isinstance(item.get("latency_seconds"), (int, float))
        and not isinstance(item.get("latency_seconds"), bool)
    ]
    if latencies:
        second_metadata["latency_seconds"] = sum(latencies)

    second_metadata["generation_attempt_count"] = 2
    second_metadata["generation_retry_reason"] = retry_reason
    second_metadata["generation_attempts"] = attempts
    metadata_path.write_text(
        json.dumps(second_metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def _structured_post_synthesis_retry_reason(
    record: dict[str, Any] | None,
) -> str | None:
    """Return a one-shot refinement reason for a verified synthesis no-gain result."""

    if not isinstance(record, dict):
        return None
    verdict = record.get("verdict")
    if verdict not in STRUCTURED_POST_SYNTHESIS_RETRY_VERDICTS:
        return None

    # This is deliberately NOT the existing pre-Vitis no-change retry.
    # Only trigger after a real successful synthesis result.
    if record.get("synthesis") is not True:
        return None

    if verdict == "reject_resource_limits":
        # Resource-limit candidates cannot be fully_verified by definition,
        # but a correctness-preserving, frequency-valid performance gain is
        # valuable evidence for one bounded resource-recovery refinement.
        if record.get("csim") is not True:
            return None
        if record.get("meets_frequency_requirement") is not True:
            return None

        performance = record.get("performance_comparison") or {}
        gains = (
            performance.get("latency_delta_percent"),
            performance.get("throughput_delta_percent"),
        )
        if not any(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value < 0
            for value in gains
        ):
            return None

        return str(verdict)

    if record.get("fully_verified") is not True:
        return None

    return str(verdict)


def _format_measured_delta(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):+.2f}%"
    return "unavailable"


def _structured_post_synthesis_retry_suffix(
    strategy: dict[str, Any],
    record: dict[str, Any],
) -> str:
    """Build generic measured feedback for one same-strategy implementation retry."""

    family = str(strategy.get("name") or "assigned_strategy")
    performance = record.get("performance_comparison") or {}
    deltas = record.get("deltas_percent") or {}

    latency = _format_measured_delta(
        performance.get("latency_delta_percent")
    )
    throughput = _format_measured_delta(
        performance.get("throughput_delta_percent")
    )
    lut = _format_measured_delta(deltas.get("resources_lut_used"))
    ff = _format_measured_delta(deltas.get("resources_ff_used"))
    dsp = _format_measured_delta(deltas.get("resources_dsp_used"))
    bram = _format_measured_delta(deltas.get("resources_bram_used"))

    family_hint = ""
    if family == "memory_parallelism":
        family_hint = (
            "\n- For memory parallelism, added storage, banking, or partitioning must "
            "expose useful concurrent accesses. Avoid serial buffering that merely "
            "relocates the same sequential access pattern."
        )
    elif family == "buffered_parallelism":
        family_hint = (
            "\n- For buffered parallelism, the buffer must feed genuinely concurrent "
            "consumers; match buffering/banking with bounded compute parallelism."
        )
    elif family == "bounded_unroll":
        family_hint = (
            "\n- For bounded unrolling, make sure the selected loop has independent "
            "work and that memory access bandwidth can support the requested parallelism."
        )
    elif family == "critical_path_restructuring":
        family_hint = (
            "\n- For critical-path restructuring, materially shorten or overlap the "
            "reported limiting computation rather than only moving equivalent operations."
        )
    elif family == "sliding_window_reuse":
        family_hint = (
            "\n- For sliding-window reuse, the new local structure must eliminate "
            "repeated external accesses rather than add an extra serial copy stage."
        )
    elif family == "dataflow_pipeline":
        family_hint = (
            "\n- For dataflow, create genuinely overlapping producer/consumer stages "
            "rather than applying DATAFLOW to an effectively serial implementation."
        )

    previous_reason = str(record.get("reason") or "no objective gain")

    compliance = record.get("resource_limit_compliance") or {}
    violations = compliance.get("violations") or []
    violation_lines = []
    for item in violations:
        if not isinstance(item, dict):
            continue
        metric = item.get("metric")
        actual = item.get("actual")
        limit = item.get("limit")
        if metric is not None and actual is not None and limit is not None:
            violation_lines.append(
                f"  - {metric}: {actual} > hard limit {limit}"
            )

    if record.get("verdict") == "reject_resource_limits":
        outcome_text = (
            "- The previous implementation passed correctness checks and synthesis "
            "and produced a useful performance gain, but exceeded one or more hard "
            "resource ceilings.\n"
            "- Preserve as much of the measured performance gain as possible while "
            "substantially reducing the resources that exceeded their limits.\n"
            "- Resource violations were:\n"
            + ("\n".join(violation_lines) if violation_lines else "  - unavailable")
            + "\n"
        )
    else:
        outcome_text = (
            "- The previous implementation passed correctness checks and synthesis, "
            "but it did not realise a useful hardware improvement.\n"
        )

    return (
        "\n\nCORRECTIVE RETRY FOR THIS SAME CANDIDATE SLOT:\n"
        "POST-SYNTHESIS NO-GAIN RETRY:\n"
        + outcome_text
        + f"- Previous verdict: {record.get('verdict')}.\n"
        f"- Previous evaluation reason: {previous_reason}\n"
        f"- Strategy family remains: {family}.\n"
        "- Preserve the same strategy family, but implement it through a materially "
        "different hardware mechanism.\n"
        "- Do not simply repeat the previous implementation.\n"
        "- Measured candidate deltas versus the verified baseline were:\n"
        f"  - latency: {latency}\n"
        f"  - throughput period: {throughput}\n"
        f"  - LUT: {lut}\n"
        f"  - FF: {ff}\n"
        f"  - DSP: {dsp}\n"
        f"  - BRAM: {bram}\n"
        "- Use these measured results as evidence that the previous implementation "
        "was ineffective. Seek actual useful concurrency, reuse, critical-path "
        "reduction, or resource reduction appropriate to the assigned strategy."
        + family_hint
        + "\n- Preserve correctness, interfaces, bounds, and hard resource limits."
        "\n- Return only the complete compilable C++ source file."
    )


_POST_SYNTHESIS_FILE_SUFFIXES = (
    ".cpp",
    "_prompt.txt",
    "_effective_prompt.txt",
    "_model_response.txt",
    "_model_metadata.json",
    "_generation_retry_failure.json",
    "_static_validation.json",
    "_duplicate_check.json",
    "_csim_validation.json",
    "_synthesis.json",
    "_cosim.json",
    "_diff.patch",
)


def _preserve_post_synthesis_attempt(
    output_dir: Path,
    candidate_index: int,
    attempt_number: int = 1,
) -> dict[str, Any]:
    """Archive the synthesis-tested implementation before overwriting its slot."""

    prefix = f"candidate_{candidate_index:03d}"
    archive_prefix = (
        f"{prefix}_post_synthesis_attempt_{attempt_number}"
    )

    for suffix in _POST_SYNTHESIS_FILE_SUFFIXES:
        source = output_dir / f"{prefix}{suffix}"
        destination = output_dir / f"{archive_prefix}{suffix}"
        if source.is_file():
            destination.write_bytes(source.read_bytes())

    # Preserve Vitis logs/TCL as well. The actual temporary Vitis project is
    # intentionally not copied; the JSON report already contains its metrics.
    for stage in ("csim", "synthesis", "cosim"):
        source_dir = output_dir / f"{prefix}_{stage}"
        destination_dir = output_dir / f"{archive_prefix}_{stage}"
        if source_dir.is_dir():
            shutil.rmtree(destination_dir, ignore_errors=True)
            shutil.copytree(source_dir, destination_dir)

    metadata_path = output_dir / f"{archive_prefix}_model_metadata.json"
    return _load_strategy(metadata_path)


def _restore_post_synthesis_generation_artifacts(
    output_dir: Path,
    candidate_index: int,
    attempt_number: int = 1,
) -> None:
    """Restore attempt one when the refinement generation itself produces no change."""

    prefix = f"candidate_{candidate_index:03d}"
    archive_prefix = (
        f"{prefix}_post_synthesis_attempt_{attempt_number}"
    )
    for suffix in (
        ".cpp",
        "_prompt.txt",
        "_effective_prompt.txt",
        "_model_response.txt",
        "_model_metadata.json",
    ):
        archived = output_dir / f"{archive_prefix}{suffix}"
        active = output_dir / f"{prefix}{suffix}"
        if archived.is_file():
            active.write_bytes(archived.read_bytes())


def _preserve_post_synthesis_retry_generation(
    output_dir: Path,
    candidate_index: int,
) -> None:
    """Keep the retry's model-facing artefacts even if attempt one is restored."""

    prefix = f"candidate_{candidate_index:03d}"
    for suffix in (
        "_prompt.txt",
        "_effective_prompt.txt",
        "_model_response.txt",
        "_model_metadata.json",
        "_generation_retry_failure.json",
    ):
        source = output_dir / f"{prefix}{suffix}"
        destination = output_dir / f"{prefix}_post_synthesis_retry{suffix}"
        if source.is_file():
            destination.write_bytes(source.read_bytes())


def _atomic_generation_attempts(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten already-aggregated generation metadata without double-counting tokens."""

    nested = metadata.get("generation_attempts")
    if isinstance(nested, list):
        attempts = [
            dict(item)
            for item in nested
            if isinstance(item, dict)
        ]
        if attempts:
            return attempts
    return [dict(metadata)] if metadata else []


def _merge_post_synthesis_retry_metadata(
    output_dir: Path,
    candidate_index: int,
    previous_metadata: dict[str, Any],
    retry_reason: str,
) -> None:
    """Merge model cost/provenance from the original and post-synthesis calls."""

    metadata_path = output_dir / f"candidate_{candidate_index:03d}_model_metadata.json"
    latest = _load_strategy(metadata_path)
    if not latest:
        return

    attempts = (
        _atomic_generation_attempts(previous_metadata)
        + _atomic_generation_attempts(latest)
    )
    if not attempts:
        return

    merged = dict(latest)

    for key in ("input_tokens", "output_tokens", "total_tokens"):
        values = [
            int(item[key])
            for item in attempts
            if isinstance(item.get(key), int)
            and not isinstance(item.get(key), bool)
        ]
        if values:
            merged[key] = sum(values)

    latencies = [
        float(item["latency_seconds"])
        for item in attempts
        if isinstance(item.get("latency_seconds"), (int, float))
        and not isinstance(item.get("latency_seconds"), bool)
    ]
    if latencies:
        merged["latency_seconds"] = sum(latencies)

    if previous_metadata.get("generation_retry_reason") is not None:
        merged["generation_retry_reason"] = previous_metadata[
            "generation_retry_reason"
        ]

    merged["generation_attempt_count"] = len(attempts)
    merged["generation_attempts"] = attempts
    merged["post_synthesis_retry_count"] = 1
    merged["post_synthesis_retry_reason"] = retry_reason

    metadata_path.write_text(
        json.dumps(merged, indent=2) + "\n",
        encoding="utf-8",
    )


def _clear_candidate_evaluation_artifacts(
    output_dir: Path,
    candidate_index: int,
) -> None:
    """Remove stale pass/fail records before evaluating the replacement source."""

    prefix = f"candidate_{candidate_index:03d}"
    for suffix in (
        "_static_validation.json",
        "_duplicate_check.json",
        "_csim_validation.json",
        "_synthesis.json",
        "_cosim.json",
        "_diff.patch",
    ):
        (output_dir / f"{prefix}{suffix}").unlink(missing_ok=True)


def _post_synthesis_retry_budget_available(
    budget: Any,
    *,
    requires_cosim: bool,
) -> bool:
    """Reserve enough authoritative budget to evaluate the replacement source."""

    if budget is None:
        return True

    for resource in ("model_calls", "csim_calls", "synthesis_calls"):
        if not budget.can_consume(resource):
            return False

    if requires_cosim and not budget.can_consume("cosim_calls"):
        return False

    remaining_tokens = getattr(budget, "total_tokens_remaining", None)
    if remaining_tokens == 0:
        return False

    if hasattr(budget, "can_afford_track_a"):
        if not budget.can_afford_track_a("csim"):
            return False
        if not budget.can_afford_track_a("synthesis"):
            return False
        if requires_cosim and not budget.can_afford_track_a("cosim"):
            return False

    return True


def _write_post_synthesis_retry_record(
    output_dir: Path,
    candidate_index: int,
    payload: dict[str, Any],
) -> Path:
    path = output_dir / f"candidate_{candidate_index:03d}_post_synthesis_retry.json"
    path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    return path



def _structured_history_ready(
    config: dict[str, Any],
    next_candidate_index: int,
) -> bool:
    """Require all preceding structured slots to carry matching metadata."""

    if next_candidate_index not in {2, 3, 4, 5}:
        return False

    output_dir = _output_dir(config)
    families = _exploration_strategy_families(config)
    schedule = build_structured_search_schedule(
        max_candidates=3,
        exploration_strategy_families=families,
    )
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
    attempt = _exploration_attempt(next_candidate_index, config)
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


def _sync_legacy_globals(names: tuple[str, ...]) -> dict[str, Any]:
    """Copy wrapper-level monkeypatches into the preserved module."""

    saved: dict[str, Any] = {}
    for name in names:
        if name in globals() and hasattr(_legacy, name):
            saved[name] = getattr(_legacy, name)
            setattr(_legacy, name, globals()[name])
    return saved


def _restore_legacy_globals(saved: dict[str, Any]) -> None:
    for name, value in saved.items():
        setattr(_legacy, name, value)


def _prepare_next_prompt(
    config_source: ConfigSource,
    previous: dict[str, Any],
    previous_index: int,
    next_index: int,
    summary: dict[str, Any],
    parent_reason: str,
) -> None:
    """Dispatch a direct legacy prompt call with wrapper monkeypatch support."""

    saved = _sync_legacy_globals(_DIRECT_PROMPT_HOOK_NAMES)
    try:
        _ORIGINAL_LEGACY_PREPARE(
            config_source,
            previous,
            previous_index,
            next_index,
            summary,
            parent_reason,
        )
    finally:
        _restore_legacy_globals(saved)


@contextmanager
def _legacy_execution_hooks(
    config_source: ConfigSource,
    config: dict[str, Any],
) -> Iterator[None]:
    """Temporarily install the five-slot policy into the preserved runner."""

    del config_source
    enabled = structured_search_enabled(config)
    saved = _sync_legacy_globals(_LEGACY_HOOK_NAMES)

    # Capture the exact incoming function so nested contexts restore their
    # caller's state.  Always delegate prompt preparation to the immutable
    # original implementation, never to a previously installed wrapper.
    incoming_prepare = _legacy._prepare_next_prompt
    base_generate = _legacy.generate_candidate
    base_select = _legacy.select_refinement_parent
    base_record = _legacy._record
    base_evaluate_candidate = _legacy._evaluate_candidate
    base_prepare = _ORIGINAL_LEGACY_PREPARE

    def structured_generate(
        source: ConfigSource,
        candidate_index: int = 1,
        *,
        budget: Any = None,
    ) -> Path:
        families = _exploration_strategy_families(config) if enabled else None
        attempt = _exploration_attempt(candidate_index, config) if enabled else None
        if attempt is not None:
            prepare_structured_exploration_prompt(
                source,
                candidate_index=candidate_index,
                strategy_family=str(attempt["strategy_family"]),
                exploration_strategy_families=families,
            )

        candidate_path = base_generate(source, candidate_index, budget=budget)
        if attempt is None or STRUCTURED_GENERATION_RETRIES <= 0:
            return candidate_path

        output_dir = _output_dir(config)
        retry_reason = _structured_generation_retry_reason(
            config,
            output_dir,
            candidate_index,
        )
        if retry_reason is None:
            return candidate_path

        if budget is not None and not budget.can_consume("model_calls"):
            (output_dir / f"candidate_{candidate_index:03d}_generation_retry_skipped.json").write_text(
                json.dumps(
                    {
                        "candidate_index": candidate_index,
                        "reason": retry_reason,
                        "retry_skipped": "model_call_budget_unavailable",
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            return candidate_path

        strategy = _load_strategy(
            output_dir / f"candidate_{candidate_index:03d}_strategy.json"
        )
        first_metadata = _preserve_generation_attempt(
            output_dir,
            candidate_index,
            1,
        )
        prompt_path = output_dir / f"candidate_{candidate_index:03d}_prompt.txt"
        with prompt_path.open("a", encoding="utf-8") as handle:
            handle.write(_structured_retry_suffix(strategy, retry_reason) + "\n")

        print(
            f"Candidate {candidate_index:03d} generation retry: {retry_reason}",
            flush=True,
        )
        retry_path = base_generate(source, candidate_index, budget=budget)
        _merge_generation_retry_metadata(
            output_dir,
            candidate_index,
            first_metadata,
            retry_reason,
        )
        return retry_path


    def structured_evaluate_candidate(
        source: ConfigSource,
        candidate_index: int,
        trajectory: list[dict[str, Any]],
        budget: Any = None,
    ) -> dict[str, Any]:
        """Evaluate once, then refine the same C1-C3 strategy once after verified no-gain."""

        summary = base_evaluate_candidate(
            source,
            candidate_index,
            trajectory,
            budget,
        )

        if not enabled or STRUCTURED_POST_SYNTHESIS_RETRIES <= 0:
            return summary

        # Keep the five-slot controller structure intact: this refinement
        # belongs to the C1-C3 exploration strategy that just failed to
        # materialise useful hardware, not to a new search slot.
        attempt = _exploration_attempt(candidate_index, config)
        if attempt is None:
            return summary

        record = base_record(summary, candidate_index)
        retry_reason = _structured_post_synthesis_retry_reason(record)
        if retry_reason is None:
            return summary

        output_dir = _output_dir(config)
        retry_record_path = (
            output_dir
            / f"candidate_{candidate_index:03d}_post_synthesis_retry.json"
        )

        # One-shot and resumable: never repeatedly burn budget on the same
        # synthesis-equivalent implementation after a restart.
        if retry_record_path.is_file():
            return summary

        requires_cosim = bool(config.get("requires_cosim", True))
        if not _post_synthesis_retry_budget_available(
            budget,
            requires_cosim=requires_cosim,
        ):
            _write_post_synthesis_retry_record(
                output_dir,
                candidate_index,
                {
                    "candidate_index": candidate_index,
                    "attempted": False,
                    "status": "skipped_budget_unavailable",
                    "trigger_verdict": retry_reason,
                    "strategy_family": attempt["strategy_family"],
                },
            )
            trajectory.append(
                {
                    "candidate": candidate_index,
                    "stage": "post_synthesis_same_strategy_retry",
                    "passed": False,
                    "skipped": True,
                    "reason": "budget_unavailable",
                }
            )
            return summary

        strategy = _load_strategy(
            output_dir / f"candidate_{candidate_index:03d}_strategy.json"
        )
        if not strategy:
            return summary

        candidate_path = output_dir / f"candidate_{candidate_index:03d}.cpp"
        prompt_path = output_dir / f"candidate_{candidate_index:03d}_prompt.txt"
        if not candidate_path.is_file() or not prompt_path.is_file():
            return summary

        original_source = candidate_path.read_text(encoding="utf-8")
        previous_metadata = _preserve_post_synthesis_attempt(
            output_dir,
            candidate_index,
            1,
        )

        # Remove any earlier pre-Vitis corrective suffix. The measured
        # post-synthesis feedback becomes the authoritative retry instruction.
        prompt_text = prompt_path.read_text(encoding="utf-8")
        retry_marker = "\n\nCORRECTIVE RETRY FOR THIS SAME CANDIDATE SLOT:\n"
        base_prompt = prompt_text.split(retry_marker, 1)[0].rstrip()
        retry_suffix = _structured_post_synthesis_retry_suffix(
            strategy,
            record,
        )
        prompt_path.write_text(
            base_prompt + retry_suffix + "\n",
            encoding="utf-8",
        )

        # A stale pre-Vitis failure record must not be mistaken for failure
        # of this post-synthesis retry. Its original copy was archived above.
        (
            output_dir
            / f"candidate_{candidate_index:03d}_generation_retry_failure.json"
        ).unlink(missing_ok=True)

        _write_post_synthesis_retry_record(
            output_dir,
            candidate_index,
            {
                "candidate_index": candidate_index,
                "attempted": True,
                "status": "generation_started",
                "trigger_verdict": retry_reason,
                "trigger_reason": record.get("reason"),
                "strategy_family": strategy.get("name"),
                "trigger_metrics": record.get("metrics") or {},
                "trigger_deltas_percent": record.get("deltas_percent") or {},
            },
        )

        print(
            f"Candidate {candidate_index:03d} post-synthesis retry: "
            f"{retry_reason}; strategy={strategy.get('name')}",
            flush=True,
        )

        retry_path = base_generate(
            source,
            candidate_index,
            budget=budget,
        )

        _preserve_post_synthesis_retry_generation(
            output_dir,
            candidate_index,
        )

        retry_source = retry_path.read_text(encoding="utf-8")
        changed = (
            normalise_source(retry_source)
            != normalise_source(original_source)
        )

        retry_failure_path = (
            output_dir
            / f"candidate_{candidate_index:03d}_generation_retry_failure.json"
        )

        if retry_failure_path.is_file() or not changed:
            # The synthesis-tested first implementation remains the truthful
            # active candidate if regeneration itself failed or repeated it.
            _restore_post_synthesis_generation_artifacts(
                output_dir,
                candidate_index,
                1,
            )
            _write_post_synthesis_retry_record(
                output_dir,
                candidate_index,
                {
                    "candidate_index": candidate_index,
                    "attempted": True,
                    "status": (
                        "generation_failed"
                        if retry_failure_path.is_file()
                        else "generation_no_semantic_change"
                    ),
                    "trigger_verdict": retry_reason,
                    "strategy_family": strategy.get("name"),
                    "replacement_source_changed": False,
                },
            )
            trajectory.append(
                {
                    "candidate": candidate_index,
                    "stage": "post_synthesis_same_strategy_retry",
                    "passed": False,
                    "reason": (
                        "generation_failed"
                        if retry_failure_path.is_file()
                        else "generation_no_semantic_change"
                    ),
                }
            )
            return summary

        _merge_post_synthesis_retry_metadata(
            output_dir,
            candidate_index,
            previous_metadata,
            retry_reason,
        )

        _clear_candidate_evaluation_artifacts(
            output_dir,
            candidate_index,
        )

        trajectory.append(
            {
                "candidate": candidate_index,
                "stage": "post_synthesis_same_strategy_retry",
                "passed": True,
                "strategy_family": strategy.get("name"),
                "trigger_verdict": retry_reason,
            }
        )

        # Re-run the normal safety -> duplicate -> CSim -> synthesis pipeline.
        # Call the preserved evaluator directly so this cannot recurse into a
        # second post-synthesis retry.
        retry_summary = base_evaluate_candidate(
            source,
            candidate_index,
            trajectory,
            budget,
        )
        retry_record = base_record(retry_summary, candidate_index)

        _write_post_synthesis_retry_record(
            output_dir,
            candidate_index,
            {
                "candidate_index": candidate_index,
                "attempted": True,
                "status": "evaluated",
                "trigger_verdict": retry_reason,
                "strategy_family": strategy.get("name"),
                "replacement_source_changed": True,
                "final_verdict": (
                    retry_record.get("verdict")
                    if isinstance(retry_record, dict)
                    else None
                ),
                "final_reason": (
                    retry_record.get("reason")
                    if isinstance(retry_record, dict)
                    else None
                ),
                "final_metrics": (
                    retry_record.get("metrics") or {}
                    if isinstance(retry_record, dict)
                    else {}
                ),
                "final_deltas_percent": (
                    retry_record.get("deltas_percent") or {}
                    if isinstance(retry_record, dict)
                    else {}
                ),
            },
        )

        return retry_summary

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

        if (
            enabled
            and next_index in {2, 3}
            and _structured_history_ready(config, next_index)
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
        families = _exploration_strategy_families(config) if enabled else None
        attempt = _exploration_attempt(next_index, config) if enabled else None
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
                exploration_strategy_families=families,
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

        # Keep the persisted summary truthful while preventing the legacy
        # early-stop branch from ending before all bounded slots complete.
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
        _legacy._evaluate_candidate = structured_evaluate_candidate

    try:
        yield
    finally:
        # Restore the directly installed dispatcher first, then every synced
        # collaborator.  This prevents wrapper accumulation across test or task
        # contexts while preserving correct nested-context behaviour.
        _legacy._prepare_next_prompt = incoming_prepare
        _restore_legacy_globals(saved)


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
