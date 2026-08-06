from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.optimise import runner, runner_legacy, structured_exploration


def _config(tmp_path: Path, *, structured: bool = True) -> Path:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "candidate_001_prompt.txt").write_text(
        "BASELINE DIAGNOSIS\nComplete baseline source:\n"
        "int kernel(int x) { return x; }\n",
        encoding="utf-8",
    )
    path = tmp_path / "config.json"
    payload = {
        "benchmark": "test",
        "top_function": "kernel",
        "output_dir": "run",
        "baseline": {"source": "baseline.cpp"},
    }
    if structured:
        payload["search_policy"] = {"mode": "structured_v1"}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _record(index: int, verdict: str = "reject_static") -> dict:
    return {
        "candidate_index": index,
        "candidate_file": f"candidate_{index:03d}.cpp",
        "verdict": verdict,
        "fully_verified": verdict == "accept_dominates_baseline",
        "static_validation": True,
        "csim": verdict == "accept_dominates_baseline",
        "synthesis": verdict == "accept_dominates_baseline",
    }


def _patch_roots(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(structured_exploration, "REPO_ROOT", tmp_path)


def test_candidate_one_generation_prepares_structured_baseline_prompt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_roots(monkeypatch, tmp_path)
    config = _config(tmp_path)
    generated: list[int] = []

    def fake_generate(source: object, candidate_index: int = 1, *, budget: object = None) -> Path:
        del source, budget
        generated.append(candidate_index)
        prompt = tmp_path / "run" / f"candidate_{candidate_index:03d}_prompt.txt"
        assert "Strategy family: critical_path_restructuring." in prompt.read_text(
            encoding="utf-8"
        )
        candidate = tmp_path / "run" / f"candidate_{candidate_index:03d}.cpp"
        candidate.write_text("int kernel(int x) { return x; }\n", encoding="utf-8")
        return candidate

    monkeypatch.setattr(runner, "generate_candidate", fake_generate)
    config_data = json.loads(config.read_text(encoding="utf-8"))

    with runner._legacy_execution_hooks(config, config_data):
        runner_legacy.generate_candidate(config, 1)

    assert generated == [1]
    strategy = json.loads(
        (tmp_path / "run" / "candidate_001_strategy.json").read_text(
            encoding="utf-8"
        )
    )
    assert strategy["name"] == "critical_path_restructuring"
    assert strategy["source_candidate_index"] == 0


def test_candidates_two_and_three_use_baseline_and_distinct_prompts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_roots(monkeypatch, tmp_path)
    config = _config(tmp_path)
    config_data = json.loads(config.read_text(encoding="utf-8"))

    # Materialise the completed earlier exploration metadata exactly as live
    # candidate generation would do.
    structured_exploration.prepare_structured_exploration_prompt(
        config,
        candidate_index=1,
        strategy_family="critical_path_restructuring",
    )

    with runner._legacy_execution_hooks(config, config_data):
        selected_two = runner_legacy.select_refinement_parent([_record(1)])
        assert selected_two is not None
        parent_two, reason_two = selected_two
        assert parent_two["candidate_index"] == 0
        assert reason_two == runner.STRUCTURED_PARENT_REASON
        runner_legacy._prepare_next_prompt(
            config,
            parent_two,
            0,
            2,
            {"candidates": [_record(1)]},
            reason_two,
        )

    prompt_two = (tmp_path / "run" / "candidate_002_prompt.txt").read_text(
        encoding="utf-8"
    )
    assert "Strategy family: bounded_unroll." in prompt_two

    with runner._legacy_execution_hooks(config, config_data):
        selected_three = runner_legacy.select_refinement_parent(
            [_record(1), _record(2)]
        )
        assert selected_three is not None
        parent_three, reason_three = selected_three
        assert parent_three["candidate_index"] == 0
        assert reason_three == runner.STRUCTURED_PARENT_REASON
        runner_legacy._prepare_next_prompt(
            config,
            parent_three,
            0,
            3,
            {"candidates": [_record(1), _record(2)]},
            reason_three,
        )

    prompt_three = (tmp_path / "run" / "candidate_003_prompt.txt").read_text(
        encoding="utf-8"
    )
    assert "Strategy family: memory_parallelism." in prompt_three
    assert prompt_two != prompt_three


def test_dominating_exploration_candidate_does_not_stop_before_slot_four(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_roots(monkeypatch, tmp_path)
    config = _config(tmp_path)
    config_data = json.loads(config.read_text(encoding="utf-8"))

    def fake_record(summary: dict, candidate_index: int) -> dict:
        del summary
        return _record(candidate_index, "accept_dominates_baseline")

    monkeypatch.setattr(runner, "_record", fake_record)

    with runner._legacy_execution_hooks(config, config_data):
        first = runner_legacy._record({}, 1)
        third = runner_legacy._record({}, 3)
        fourth = runner_legacy._record({}, 4)

    assert first["verdict"] == "keep_pareto_candidate"
    assert third["verdict"] == "keep_pareto_candidate"
    assert first["structured_exploration_continue"] is True
    assert fourth["verdict"] == "accept_dominates_baseline"


def test_candidate_four_delegates_to_existing_parent_selection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_roots(monkeypatch, tmp_path)
    config = _config(tmp_path)
    config_data = json.loads(config.read_text(encoding="utf-8"))
    delegated: list[int] = []

    def fake_select(records: object, selection: dict | None = None):
        del selection
        values = list(records)
        delegated.append(len(values))
        return _record(2, "keep_pareto_candidate"), "pareto_candidate"

    monkeypatch.setattr(runner, "select_refinement_parent", fake_select)

    with runner._legacy_execution_hooks(config, config_data):
        selected = runner_legacy.select_refinement_parent(
            [_record(1), _record(2), _record(3)]
        )

    assert delegated == [3]
    assert selected is not None
    parent, reason = selected
    assert parent["candidate_index"] == 2
    assert reason == "pareto_candidate"


def test_legacy_config_does_not_activate_structured_generation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_roots(monkeypatch, tmp_path)
    config = _config(tmp_path, structured=False)
    config_data = json.loads(config.read_text(encoding="utf-8"))

    def fake_generate(source: object, candidate_index: int = 1, *, budget: object = None) -> Path:
        del source, budget
        candidate = tmp_path / "run" / f"candidate_{candidate_index:03d}.cpp"
        candidate.write_text("int kernel(int x) { return x; }\n", encoding="utf-8")
        return candidate

    monkeypatch.setattr(runner, "generate_candidate", fake_generate)

    with runner._legacy_execution_hooks(config, config_data):
        runner_legacy.generate_candidate(config, 1)

    assert not (tmp_path / "run" / "candidate_001_strategy.json").exists()
    original_prompt = (tmp_path / "run" / "candidate_001_prompt.txt").read_text(
        encoding="utf-8"
    )
    assert "Structured exploration contract" not in original_prompt


def test_track_a_config_is_compatibility_opt_in() -> None:
    assert runner.structured_search_enabled({"track_a": {}}) is True
    assert runner.structured_search_enabled({}) is False
