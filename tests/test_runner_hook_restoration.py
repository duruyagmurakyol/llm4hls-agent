from __future__ import annotations

from pathlib import Path

import pytest

from agent.optimise import runner, runner_legacy


def _structured_config(tmp_path: Path) -> dict:
    return {
        "search_policy": {"mode": "structured_v1"},
        "output_dir": str(tmp_path / "run"),
        "baseline": {"source": "baseline.cpp"},
        "budget": {"max_candidates": 5},
    }


def test_repeated_hook_contexts_restore_original_prepare(tmp_path: Path) -> None:
    original = runner._ORIGINAL_LEGACY_PREPARE
    runner_legacy._prepare_next_prompt = original
    config = _structured_config(tmp_path)

    for _ in range(8):
        with runner._legacy_execution_hooks(tmp_path / "config.json", config):
            assert runner_legacy._prepare_next_prompt is not original
        assert runner_legacy._prepare_next_prompt is original


def test_nested_hook_context_restores_outer_then_original(tmp_path: Path) -> None:
    original = runner._ORIGINAL_LEGACY_PREPARE
    runner_legacy._prepare_next_prompt = original
    config = _structured_config(tmp_path)

    with runner._legacy_execution_hooks(tmp_path / "config.json", config):
        outer = runner_legacy._prepare_next_prompt
        assert outer is not original

        with runner._legacy_execution_hooks(tmp_path / "config.json", config):
            inner = runner_legacy._prepare_next_prompt
            assert inner is not outer
            assert inner is not original

        assert runner_legacy._prepare_next_prompt is outer

    assert runner_legacy._prepare_next_prompt is original


def test_hook_context_restores_prepare_after_exception(tmp_path: Path) -> None:
    original = runner._ORIGINAL_LEGACY_PREPARE
    runner_legacy._prepare_next_prompt = original
    config = _structured_config(tmp_path)

    with pytest.raises(RuntimeError, match="deliberate"):
        with runner._legacy_execution_hooks(tmp_path / "config.json", config):
            raise RuntimeError("deliberate")

    assert runner_legacy._prepare_next_prompt is original


def test_direct_resource_recovery_dispatch_after_repeated_contexts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original = runner._ORIGINAL_LEGACY_PREPARE
    runner_legacy._prepare_next_prompt = original
    config = _structured_config(tmp_path)

    for _ in range(5):
        with runner._legacy_execution_hooks(tmp_path / "config.json", config):
            pass

    called: dict[str, int] = {}
    rejected = {"candidate_index": 3}

    monkeypatch.setattr(
        runner,
        "is_resource_frequency_balance_reason",
        lambda reason: False,
    )
    monkeypatch.setattr(
        runner,
        "is_resource_recovery_reason",
        lambda reason: reason == "test_resource_recovery",
    )
    monkeypatch.setattr(
        runner,
        "resource_limit_recovery_trigger",
        lambda records: rejected,
    )

    def fake_prepare(
        config_source: object,
        parent_index: int,
        rejected_index: int,
        next_index: int,
    ) -> Path:
        del config_source
        called.update(
            parent_index=parent_index,
            rejected_index=rejected_index,
            next_index=next_index,
        )
        return tmp_path / "prompt.txt"

    monkeypatch.setattr(runner, "prepare_resource_recovery_prompt", fake_prepare)

    parent = {
        "candidate_index": 2,
        "candidate_file": "candidate_002.cpp",
        "verdict": "keep_pareto_candidate",
    }
    runner._prepare_next_prompt(
        tmp_path / "missing-config.json",
        parent,
        2,
        4,
        {"candidates": [parent, rejected]},
        "test_resource_recovery",
    )

    assert called == {
        "parent_index": 2,
        "rejected_index": 3,
        "next_index": 4,
    }
    assert runner_legacy._prepare_next_prompt is original
