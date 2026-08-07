from __future__ import annotations

from pathlib import Path

from agent.optimise import search_ledger_runtime as runtime
from agent.optimise.search_ledger import CORRECTIVE_RETRY_MARKER, load_search_ledger


def _plan(output_dir: Path) -> dict[str, object]:
    return {
        "candidate_index": 2,
        "output_dir": output_dir,
        "parent_candidate_index": 0,
        "parent_source_hash": "baseline-hash",
        "strategy_family": "bounded_unroll",
        "parameters": {"allowed_factors": [2, 4]},
        "phase": "explore",
    }


def test_corrective_retry_is_registered_then_reaches_provider(
    monkeypatch,
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    initial_prompt = "optimise with bounded unroll\n"
    runtime._register_exact_prompt(plan, initial_prompt)

    retry_prompt = (
        initial_prompt
        + "\n"
        + CORRECTIVE_RETRY_MARKER
        + "\n- previous response was unchanged\n"
    )
    calls: list[str] = []

    def fake_complete(*args, **kwargs):
        calls.append(kwargs["user_prompt"])
        return "provider-response"

    monkeypatch.setattr(runtime, "_ORIGINAL_COMPLETE", fake_complete)
    token = runtime._PENDING_GENERATION.set(plan)
    try:
        result = runtime._guarded_complete(user_prompt=retry_prompt)
    finally:
        runtime._PENDING_GENERATION.reset(token)

    assert result == "provider-response"
    assert calls == [retry_prompt]

    attempt = load_search_ledger(tmp_path)["attempts"][0]
    assert attempt["generation_attempt_count"] == 2
    assert [item["kind"] for item in attempt["generation_prompt_attempts"]] == [
        "initial",
        "corrective_retry",
    ]
