from __future__ import annotations

from pathlib import Path

import pytest

from agent.optimise.search_ledger import (
    CORRECTIVE_RETRY_MARKER,
    SearchLedgerError,
    load_search_ledger,
    register_search_attempt,
    text_digest,
)


def _register(output_dir: Path, prompt: str):
    return register_search_attempt(
        output_dir,
        candidate_index=2,
        parent_candidate_index=0,
        parent_source_hash="baseline-hash",
        strategy_family="bounded_unroll",
        parameters={"allowed_factors": [2, 4]},
        effective_prompt=prompt,
        phase="explore",
    )


def test_corrective_retry_keeps_branch_identity_and_records_prompt_history(
    tmp_path: Path,
) -> None:
    initial_prompt = "optimise with bounded unroll\n"
    first = _register(tmp_path, initial_prompt)

    retry_prompt = (
        initial_prompt
        + "\n"
        + CORRECTIVE_RETRY_MARKER
        + "\n- previous response was unchanged\n"
    )
    retry = _register(tmp_path, retry_prompt)

    assert retry["generation_retry_registered"] is True
    assert retry["branch_fingerprint"] == first["branch_fingerprint"]
    assert retry["effective_prompt_hash"] == text_digest(initial_prompt)
    assert retry["latest_effective_prompt_hash"] == text_digest(retry_prompt)
    assert retry["generation_attempt_count"] == 2

    ledger = load_search_ledger(tmp_path)
    attempt = ledger["attempts"][0]
    assert [item["kind"] for item in attempt["generation_prompt_attempts"]] == [
        "initial",
        "corrective_retry",
    ]
    assert [
        item["effective_prompt_hash"]
        for item in attempt["generation_prompt_attempts"]
    ] == [text_digest(initial_prompt), text_digest(retry_prompt)]


def test_corrective_retry_registration_is_idempotent(tmp_path: Path) -> None:
    initial_prompt = "optimise with bounded unroll\n"
    _register(tmp_path, initial_prompt)
    retry_prompt = initial_prompt + "\n" + CORRECTIVE_RETRY_MARKER + "\nretry\n"

    _register(tmp_path, retry_prompt)
    repeated = _register(tmp_path, retry_prompt)

    assert repeated["generation_retry_registered"] is True
    assert repeated["generation_retry_idempotent"] is True
    attempt = load_search_ledger(tmp_path)["attempts"][0]
    assert len(attempt["generation_prompt_attempts"]) == 2


def test_changed_prompt_without_corrective_marker_is_still_rejected(
    tmp_path: Path,
) -> None:
    _register(tmp_path, "optimise with bounded unroll\n")

    with pytest.raises(
        SearchLedgerError,
        match="already registered with different search data",
    ):
        _register(tmp_path, "different ordinary prompt\n")


def test_retry_cannot_change_search_branch(tmp_path: Path) -> None:
    initial_prompt = "optimise with bounded unroll\n"
    _register(tmp_path, initial_prompt)
    retry_prompt = initial_prompt + "\n" + CORRECTIVE_RETRY_MARKER + "\nretry\n"

    with pytest.raises(
        SearchLedgerError,
        match="already registered with different search data",
    ):
        register_search_attempt(
            tmp_path,
            candidate_index=2,
            parent_candidate_index=0,
            parent_source_hash="baseline-hash",
            strategy_family="memory_parallelism",
            parameters={},
            effective_prompt=retry_prompt,
            phase="explore",
        )
