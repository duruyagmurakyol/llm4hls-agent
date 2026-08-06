from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.optimise import (
    runner,
    runner_legacy,
    structured_exploration,
    structured_tail,
)


def _config(tmp_path: Path) -> Path:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "candidate_001_prompt.txt").write_text(
        "BASELINE DIAGNOSIS\nComplete baseline source:\n"
        "int kernel(int x) { return x; }\n",
        encoding="utf-8",
    )
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "benchmark": "test",
                "top_function": "kernel",
                "output_dir": "run",
                "baseline": {"source": "baseline.cpp"},
                "budget": {"max_candidates": 5},
                "search_policy": {"mode": "structured_v1"},
            }
        ),
        encoding="utf-8",
    )
    return path


def _record(index: int) -> dict:
    return {
        "candidate_index": index,
        "candidate_file": f"candidate_{index:03d}.cpp",
        "verdict": "reject_no_objective_gain",
        "refinement_eligible": False,
        "fully_verified": True,
        "static_validation": True,
        "csim": True,
        "synthesis": True,
    }


def _prepare_history(config: Path) -> None:
    for index, family in enumerate(
        (
            "critical_path_restructuring",
            "bounded_unroll",
            "memory_parallelism",
        ),
        1,
    ):
        structured_exploration.prepare_structured_exploration_prompt(
            config,
            candidate_index=index,
            strategy_family=family,
        )


def _patch_roots(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(structured_exploration, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(structured_tail, "REPO_ROOT", tmp_path)


def test_c4_fallback_dispatch_writes_prompt_strategy_and_decision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_roots(monkeypatch, tmp_path)
    config = _config(tmp_path)
    config_data = json.loads(config.read_text(encoding="utf-8"))
    _prepare_history(config)
    records = [_record(index) for index in (1, 2, 3)]

    with runner._legacy_execution_hooks(config, config_data):
        selected = runner_legacy.select_refinement_parent(records)
        assert selected is not None
        parent, reason = selected
        runner_legacy._prepare_next_prompt(
            config,
            parent,
            int(parent["candidate_index"]),
            4,
            {"candidates": records},
            reason,
        )

    prompt = (tmp_path / "run" / "candidate_004_prompt.txt").read_text(
        encoding="utf-8"
    )
    assert "Strategy family: loop_schedule_restructuring." in prompt

    strategy = json.loads(
        (tmp_path / "run" / "candidate_004_strategy.json").read_text(
            encoding="utf-8"
        )
    )
    assert strategy["source_candidate_index"] == 0
    assert strategy["compliance_mode"] == "advisory"

    decision = json.loads(
        (tmp_path / "run" / "candidate_004_search_decision.json").read_text(
            encoding="utf-8"
        )
    )
    assert decision["selected_parent_index"] == 0
    assert decision["phase"] == "exploit"
    assert (
        decision["parent_reason"]
        == structured_tail.STRUCTURED_EXPLOIT_FALLBACK_REASON
    )


def test_c5_fallback_dispatch_is_final_independent_family(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_roots(monkeypatch, tmp_path)
    config = _config(tmp_path)
    config_data = json.loads(config.read_text(encoding="utf-8"))
    _prepare_history(config)
    (tmp_path / "run" / "candidate_004_search_decision.json").write_text(
        json.dumps({"candidate_index": 4, "phase": "exploit"}),
        encoding="utf-8",
    )
    records = [_record(index) for index in (1, 2, 3, 4)]

    with runner._legacy_execution_hooks(config, config_data):
        selected = runner_legacy.select_refinement_parent(records)
        assert selected is not None
        parent, reason = selected
        runner_legacy._prepare_next_prompt(
            config,
            parent,
            int(parent["candidate_index"]),
            5,
            {"candidates": records},
            reason,
        )

    prompt = (tmp_path / "run" / "candidate_005_prompt.txt").read_text(
        encoding="utf-8"
    )
    assert "Strategy family: pipeline_dataflow_restructuring." in prompt
    assert "original verified baseline (candidate 000)" in prompt

    decision = json.loads(
        (tmp_path / "run" / "candidate_005_search_decision.json").read_text(
            encoding="utf-8"
        )
    )
    assert decision["selected_parent_index"] == 0
    assert decision["phase"] == "recover"
    assert (
        decision["parent_reason"]
        == structured_tail.STRUCTURED_RECOVERY_FALLBACK_REASON
    )
