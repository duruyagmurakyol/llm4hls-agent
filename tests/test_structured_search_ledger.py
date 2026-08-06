from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.optimise.search_ledger import (
    SearchLedgerError,
    attempted_branch_fingerprints,
    attempted_source_hashes,
    branch_fingerprint,
    canonical_parameters,
    inspect_branch_novelty,
    load_search_ledger,
    record_candidate_source,
    record_search_outcome,
    register_search_attempt,
    retired_candidate_indices,
    text_digest,
)


def _register(
    output_dir: Path,
    index: int,
    *,
    parent_hash: str = "baseline-hash",
    family: str = "bounded_unroll",
    parameters: dict | None = None,
    prompt: str | None = None,
):
    return register_search_attempt(
        output_dir,
        candidate_index=index,
        parent_candidate_index=0,
        parent_source_hash=parent_hash,
        strategy_family=family,
        parameters=parameters or {"factor": 2},
        effective_prompt=prompt or f"prompt for candidate {index}\n",
        phase="explore",
    )


def test_parameter_order_does_not_change_branch_identity() -> None:
    first = branch_fingerprint(
        "parent",
        "bounded_unroll",
        {"factor": 2, "options": {"pipeline": True, "tail": "guard"}},
    )
    second = branch_fingerprint(
        "parent",
        "bounded_unroll",
        {"options": {"tail": "guard", "pipeline": True}, "factor": 2},
    )
    assert first == second


def test_different_parent_family_or_parameter_changes_branch_identity() -> None:
    baseline = branch_fingerprint("parent-a", "bounded_unroll", {"factor": 2})
    assert branch_fingerprint("parent-b", "bounded_unroll", {"factor": 2}) != baseline
    assert branch_fingerprint("parent-a", "memory_parallelism", {"factor": 2}) != baseline
    assert branch_fingerprint("parent-a", "bounded_unroll", {"factor": 4}) != baseline


def test_set_parameters_are_canonical_and_deterministic() -> None:
    assert canonical_parameters({"factors": {4, 2}}) == {"factors": [2, 4]}


def test_prompt_digest_ignores_line_endings_and_trailing_whitespace() -> None:
    assert text_digest("alpha  \r\nbeta\r\n") == text_digest("alpha\nbeta\n")


def test_register_attempt_persists_deterministic_record(tmp_path: Path) -> None:
    record = _register(tmp_path, 1)
    ledger = load_search_ledger(tmp_path)

    assert record["candidate_index"] == 1
    assert record["parent_candidate_index"] == 0
    assert record["strategy_family"] == "bounded_unroll"
    assert record["parameters"] == {"factor": 2}
    assert record["completed"] is False
    assert record["retired"] is False
    assert ledger["schema_version"] == 1
    assert ledger["attempts"] == [record]


def test_identical_candidate_registration_is_idempotent(tmp_path: Path) -> None:
    first = _register(tmp_path, 1)
    second = _register(tmp_path, 1)

    assert second == first
    assert len(load_search_ledger(tmp_path)["attempts"]) == 1


def test_same_candidate_index_with_different_branch_is_rejected(tmp_path: Path) -> None:
    _register(tmp_path, 1)
    with pytest.raises(SearchLedgerError):
        _register(tmp_path, 1, family="memory_parallelism")


def test_repeated_branch_is_rejected_before_registration(tmp_path: Path) -> None:
    _register(tmp_path, 1, prompt="first prompt\n")
    proposal = _register(tmp_path, 2, prompt="different prompt\n")

    assert proposal["registration_allowed"] is False
    assert proposal["registration_rejection_reason"] == "branch_already_attempted"
    assert proposal["duplicate_of_candidate_index"] == 1
    assert len(load_search_ledger(tmp_path)["attempts"]) == 1


def test_repeated_effective_prompt_is_rejected_for_different_branch(
    tmp_path: Path,
) -> None:
    _register(tmp_path, 1, family="bounded_unroll", prompt="same prompt\n")
    proposal = _register(
        tmp_path,
        2,
        family="memory_parallelism",
        prompt="same prompt  \r\n",
    )

    assert proposal["registration_allowed"] is False
    assert proposal["registration_rejection_reason"] == "effective_prompt_already_attempted"
    assert proposal["duplicate_of_candidate_index"] == 1


def test_novel_branch_and_prompt_are_allowed(tmp_path: Path) -> None:
    _register(tmp_path, 1)
    report = inspect_branch_novelty(
        tmp_path,
        candidate_index=2,
        parent_source_hash="baseline-hash",
        strategy_family="memory_parallelism",
        parameters={},
        effective_prompt="new prompt\n",
    )

    assert report["allowed"] is True
    assert report["reason"] is None
    assert report["duplicate_of_candidate_index"] is None


def test_generated_source_duplicate_is_retired(tmp_path: Path) -> None:
    _register(tmp_path, 1, family="critical_path_restructuring")
    register_search_attempt(
        tmp_path,
        candidate_index=2,
        parent_candidate_index=0,
        parent_source_hash="baseline-hash",
        strategy_family="memory_parallelism",
        parameters={},
        effective_prompt="different prompt\n",
        phase="explore",
    )

    source = "int kernel(int x) { return x + 1; }\n"
    first = record_candidate_source(tmp_path, candidate_index=1, source_text=source)
    second = record_candidate_source(
        tmp_path,
        candidate_index=2,
        source_text="// comment\nint kernel(int x){return x+1;}\n",
    )

    assert first["source_duplicate_of_candidate_index"] is None
    assert second["source_duplicate_of_candidate_index"] == 1
    assert second["retired"] is True
    assert second["retirement_reason"] == "duplicate_generated_source"
    assert retired_candidate_indices(tmp_path) == {2}
    assert len(attempted_source_hashes(tmp_path)) == 1


@pytest.mark.parametrize(
    "verdict",
    [
        "reject_duplicate",
        "reject_static",
        "reject_no_objective_gain",
        "reject_synthesis_equivalent",
        "reject_strategy_not_realised",
        "reject_resource_limits",
    ],
)
def test_terminal_failure_verdict_retires_branch(
    tmp_path: Path,
    verdict: str,
) -> None:
    _register(tmp_path, 1)
    result = record_search_outcome(tmp_path, candidate_index=1, verdict=verdict)

    assert result["completed"] is True
    assert result["retired"] is True
    assert result["retirement_reason"] == verdict
    assert retired_candidate_indices(tmp_path) == {1}


def test_successful_or_archive_outcome_completes_without_retirement(
    tmp_path: Path,
) -> None:
    _register(tmp_path, 1)
    result = record_search_outcome(
        tmp_path,
        candidate_index=1,
        verdict="keep_pareto_candidate",
    )

    assert result["completed"] is True
    assert result["retired"] is False
    assert result["retirement_reason"] is None


def test_branch_and_source_sets_are_exposed(tmp_path: Path) -> None:
    registered = _register(tmp_path, 1)
    source_record = record_candidate_source(
        tmp_path,
        candidate_index=1,
        source_text="int kernel(int x) { return x + 1; }\n",
    )

    assert attempted_branch_fingerprints(tmp_path) == {
        registered["branch_fingerprint"]
    }
    assert attempted_source_hashes(tmp_path) == {
        source_record["candidate_source_hash"]
    }


def test_unknown_candidate_cannot_receive_source_or_outcome(tmp_path: Path) -> None:
    with pytest.raises(SearchLedgerError):
        record_candidate_source(
            tmp_path,
            candidate_index=7,
            source_text="int kernel() { return 0; }",
        )
    with pytest.raises(SearchLedgerError):
        record_search_outcome(tmp_path, candidate_index=7, verdict="reject_static")


def test_malformed_ledger_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "structured_search_ledger.json").write_text(
        json.dumps({"schema_version": 99, "attempts": []}),
        encoding="utf-8",
    )
    with pytest.raises(SearchLedgerError):
        load_search_ledger(tmp_path)
