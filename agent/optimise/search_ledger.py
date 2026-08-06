"""Deterministic novelty and branch-retirement ledger for structured PPA search.

This module records search attempts but does not call a model, modify prompts,
or run validation tools. Runtime enforcement is wired in a separate stage.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from agent.optimise.duplicate import source_digest

LEDGER_FILENAME = "structured_search_ledger.json"
SCHEMA_VERSION = 1

TERMINAL_RETIREMENT_VERDICTS = {
    "reject_duplicate",
    "reject_no_change",
    "reject_no_objective_gain",
    "reject_dominated_pre_cosim",
    "reject_no_change_pre_cosim",
    "reject_static",
    "reject_strategy_not_realised",
    "reject_synthesis_equivalent",
    "reject_resource_limits",
}


class SearchLedgerError(RuntimeError):
    """Raised when ledger data is malformed or an unknown candidate is updated."""


def _canonical_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, set):
        canonical = [_canonical_value(item) for item in value]
        return sorted(
            canonical,
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
        )
    raise TypeError(f"Unsupported ledger value type: {type(value).__name__}")


def canonical_parameters(parameters: Any) -> dict[str, Any]:
    """Return deterministic JSON-compatible strategy parameters."""

    if parameters in (None, {}):
        return {}
    if not isinstance(parameters, dict):
        raise TypeError("strategy parameters must be an object")
    return _canonical_value(parameters)


def text_digest(text: str) -> str:
    """Hash text after stable newline and trailing-whitespace normalisation."""

    normalised = "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").split("\n"))
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def branch_fingerprint(
    parent_source_hash: str,
    strategy_family: str,
    parameters: Any = None,
) -> str:
    """Return the canonical identity of one parent/strategy/parameter branch."""

    if not isinstance(parent_source_hash, str) or not parent_source_hash.strip():
        raise ValueError("parent_source_hash must be a non-empty string")
    if not isinstance(strategy_family, str) or not strategy_family.strip():
        raise ValueError("strategy_family must be a non-empty string")
    payload = {
        "parent_source_hash": parent_source_hash.strip(),
        "strategy_family": strategy_family.strip(),
        "parameters": canonical_parameters(parameters),
    }
    serialised = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


def _empty_ledger() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "attempts": [],
    }


def ledger_path(output_dir: Path) -> Path:
    return output_dir / LEDGER_FILENAME


def load_search_ledger(output_dir: Path) -> dict[str, Any]:
    """Load the ledger, returning an empty schema when it does not exist."""

    path = ledger_path(output_dir)
    if not path.is_file():
        return _empty_ledger()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SearchLedgerError(f"Could not read search ledger: {path}") from error
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise SearchLedgerError("Unsupported or malformed search ledger schema")
    attempts = value.get("attempts")
    if not isinstance(attempts, list) or not all(isinstance(item, dict) for item in attempts):
        raise SearchLedgerError("Search ledger attempts must be a list of objects")
    return value


def write_search_ledger(output_dir: Path, ledger: dict[str, Any]) -> Path:
    """Persist the ledger atomically."""

    output_dir.mkdir(parents=True, exist_ok=True)
    path = ledger_path(output_dir)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def _attempt_by_index(
    ledger: dict[str, Any],
    candidate_index: int,
) -> dict[str, Any] | None:
    return next(
        (
            item
            for item in ledger.get("attempts", [])
            if item.get("candidate_index") == candidate_index
        ),
        None,
    )


def inspect_branch_novelty(
    output_dir: Path,
    *,
    candidate_index: int,
    parent_source_hash: str,
    strategy_family: str,
    parameters: Any = None,
    effective_prompt: str | None = None,
) -> dict[str, Any]:
    """Report whether a proposed branch and effective prompt are novel."""

    if isinstance(candidate_index, bool) or not isinstance(candidate_index, int) or candidate_index <= 0:
        raise ValueError("candidate_index must be a positive integer")
    ledger = load_search_ledger(output_dir)
    fingerprint = branch_fingerprint(parent_source_hash, strategy_family, parameters)
    prompt_hash = text_digest(effective_prompt) if effective_prompt is not None else None

    duplicate_branch = next(
        (
            item
            for item in ledger["attempts"]
            if item.get("branch_fingerprint") == fingerprint
            and item.get("candidate_index") != candidate_index
        ),
        None,
    )
    duplicate_prompt = next(
        (
            item
            for item in ledger["attempts"]
            if prompt_hash is not None
            and item.get("effective_prompt_hash") == prompt_hash
            and item.get("candidate_index") != candidate_index
        ),
        None,
    )

    reason = None
    duplicate_of = None
    if duplicate_branch is not None:
        reason = "branch_already_attempted"
        duplicate_of = duplicate_branch.get("candidate_index")
    elif duplicate_prompt is not None:
        reason = "effective_prompt_already_attempted"
        duplicate_of = duplicate_prompt.get("candidate_index")

    return {
        "candidate_index": candidate_index,
        "allowed": reason is None,
        "reason": reason,
        "duplicate_of_candidate_index": duplicate_of,
        "branch_fingerprint": fingerprint,
        "effective_prompt_hash": prompt_hash,
    }


def register_search_attempt(
    output_dir: Path,
    *,
    candidate_index: int,
    parent_candidate_index: int,
    parent_source_hash: str,
    strategy_family: str,
    parameters: Any = None,
    effective_prompt: str | None = None,
    phase: str | None = None,
) -> dict[str, Any]:
    """Register one novel search attempt and return its persisted record.

    Duplicate proposals are reported without changing the ledger. Re-registering
    the same candidate index with identical data is idempotent.
    """

    ledger = load_search_ledger(output_dir)
    existing = _attempt_by_index(ledger, candidate_index)
    novelty = inspect_branch_novelty(
        output_dir,
        candidate_index=candidate_index,
        parent_source_hash=parent_source_hash,
        strategy_family=strategy_family,
        parameters=parameters,
        effective_prompt=effective_prompt,
    )
    parameters_value = canonical_parameters(parameters)

    record = {
        "candidate_index": candidate_index,
        "parent_candidate_index": parent_candidate_index,
        "parent_source_hash": parent_source_hash.strip(),
        "strategy_family": strategy_family.strip(),
        "parameters": parameters_value,
        "phase": phase,
        "branch_fingerprint": novelty["branch_fingerprint"],
        "effective_prompt_hash": novelty["effective_prompt_hash"],
        "candidate_source_hash": None,
        "source_duplicate_of_candidate_index": None,
        "verdict": None,
        "completed": False,
        "retired": False,
        "retirement_reason": None,
    }

    if existing is not None:
        comparable_keys = {
            "parent_candidate_index",
            "parent_source_hash",
            "strategy_family",
            "parameters",
            "phase",
            "branch_fingerprint",
            "effective_prompt_hash",
        }
        if all(existing.get(key) == record.get(key) for key in comparable_keys):
            return dict(existing)
        raise SearchLedgerError(
            f"candidate {candidate_index:03d} is already registered with different search data"
        )

    if not novelty["allowed"]:
        return {
            **record,
            "registration_allowed": False,
            "registration_rejection_reason": novelty["reason"],
            "duplicate_of_candidate_index": novelty["duplicate_of_candidate_index"],
        }

    ledger["attempts"].append(record)
    ledger["attempts"].sort(key=lambda item: int(item["candidate_index"]))
    write_search_ledger(output_dir, ledger)
    return dict(record)


def record_candidate_source(
    output_dir: Path,
    *,
    candidate_index: int,
    source_text: str,
) -> dict[str, Any]:
    """Record a generated source hash and identify earlier equivalent sources."""

    ledger = load_search_ledger(output_dir)
    attempt = _attempt_by_index(ledger, candidate_index)
    if attempt is None:
        raise SearchLedgerError(f"candidate {candidate_index:03d} is not registered")

    digest = source_digest(source_text)
    duplicate = next(
        (
            item
            for item in ledger["attempts"]
            if item.get("candidate_index") != candidate_index
            and item.get("candidate_source_hash") == digest
        ),
        None,
    )
    attempt["candidate_source_hash"] = digest
    attempt["source_duplicate_of_candidate_index"] = (
        duplicate.get("candidate_index") if duplicate is not None else None
    )
    if duplicate is not None:
        attempt["retired"] = True
        attempt["retirement_reason"] = "duplicate_generated_source"
    write_search_ledger(output_dir, ledger)
    return dict(attempt)


def record_search_outcome(
    output_dir: Path,
    *,
    candidate_index: int,
    verdict: str,
) -> dict[str, Any]:
    """Mark an attempt complete and retire terminal failed branches."""

    if not isinstance(verdict, str) or not verdict:
        raise ValueError("verdict must be a non-empty string")
    ledger = load_search_ledger(output_dir)
    attempt = _attempt_by_index(ledger, candidate_index)
    if attempt is None:
        raise SearchLedgerError(f"candidate {candidate_index:03d} is not registered")

    attempt["verdict"] = verdict
    attempt["completed"] = True
    if verdict in TERMINAL_RETIREMENT_VERDICTS:
        attempt["retired"] = True
        attempt["retirement_reason"] = verdict
    write_search_ledger(output_dir, ledger)
    return dict(attempt)


def attempted_branch_fingerprints(output_dir: Path) -> set[str]:
    """Return every previously registered branch identity."""

    return {
        str(item["branch_fingerprint"])
        for item in load_search_ledger(output_dir)["attempts"]
        if isinstance(item.get("branch_fingerprint"), str)
    }


def attempted_source_hashes(output_dir: Path) -> set[str]:
    """Return every generated candidate source identity."""

    return {
        str(item["candidate_source_hash"])
        for item in load_search_ledger(output_dir)["attempts"]
        if isinstance(item.get("candidate_source_hash"), str)
    }


def retired_candidate_indices(output_dir: Path) -> set[int]:
    """Return candidate indices whose branches are terminally retired."""

    return {
        int(item["candidate_index"])
        for item in load_search_ledger(output_dir)["attempts"]
        if item.get("retired") is True and isinstance(item.get("candidate_index"), int)
    }
