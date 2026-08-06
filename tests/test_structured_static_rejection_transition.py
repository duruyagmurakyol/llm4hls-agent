from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.optimise import runner, runner_legacy, structured_exploration
from agent.optimise import structured_transition_runtime as runtime


def _config(tmp_path: Path) -> tuple[Path, dict]:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (tmp_path / "baseline.cpp").write_text(
        "#include <stdint.h>\nint kernel(int x) { return x; }\n",
        encoding="utf-8",
    )
    (output_dir / "baseline_optimisation_prompt.txt").write_text(
        "BASELINE DIAGNOSIS\nComplete baseline source:\n"
        "#include <stdint.h>\nint kernel(int x) { return x; }\n",
        encoding="utf-8",
    )
    (output_dir / "candidate_001_prompt.txt").write_text(
        "C1 structured prompt\n",
        encoding="utf-8",
    )
    payload = {
        "benchmark": "test",
        "top_function": "kernel",
        "output_dir": "run",
        "baseline": {"source": "baseline.cpp"},
        "budget": {"max_candidates": 5},
        "search_policy": {"mode": "structured_v1"},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload


def _record(tmp_path: Path, index: int) -> dict:
    candidate = tmp_path / "run" / f"candidate_{index:03d}.cpp"
    candidate.write_text(
        "#include <stdint.h>\nint kernel(int x) { return x; }\n",
        encoding="utf-8",
    )
    return {
        "candidate_index": index,
        "candidate_file": str(candidate),
        "verdict": "reject_static",
        "refinement_eligible": False,
        "fully_verified": False,
        "static_validation": False,
        "csim": False,
        "synthesis": False,
    }


def _write_partial_strategy(
    tmp_path: Path,
    index: int,
    family: str,
) -> None:
    # Intentionally omit schedule_slot. The wrapper's strict audit-ready check
    # must fail, while persisted structured evidence remains sufficient for the
    # compatibility guard to preserve the declared C1->C2->C3 schedule.
    payload = {
        "name": family,
        "source_candidate_index": 0,
        "next_candidate_index": index,
        "trigger": "structured_baseline_exploration",
        "phase": "explore",
        "compliance_mode": "advisory",
        "parameters": {},
    }
    (tmp_path / "run" / f"candidate_{index:03d}_strategy.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def _patch_roots(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(runner_legacy, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(structured_exploration, "REPO_ROOT", tmp_path)


def test_static_rejected_c1_still_prepares_structured_c2(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_roots(monkeypatch, tmp_path)
    config_path, config = _config(tmp_path)
    first = _record(tmp_path, 1)
    _write_partial_strategy(
        tmp_path,
        1,
        "critical_path_restructuring",
    )

    assert runner._structured_history_ready(config, 2) is False

    with runner._legacy_execution_hooks(config_path, config):
        selected = runner_legacy.select_refinement_parent([first])
        assert selected is not None
        parent, reason = selected
        assert parent["candidate_index"] == 0
        assert reason == runner.STRUCTURED_PARENT_REASON

        runner_legacy._prepare_next_prompt(
            config_path,
            parent,
            0,
            2,
            {"candidates": [first]},
            reason,
        )

    prompt = (tmp_path / "run" / "candidate_002_prompt.txt").read_text(
        encoding="utf-8"
    )
    assert "Strategy family: bounded_unroll." in prompt
    assert not (tmp_path / "run" / "candidate_002_feedback.json").read_text(
        encoding="utf-8"
    ).startswith("PPA baseline-restart")


def test_static_rejected_c2_still_selects_baseline_for_c3(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_roots(monkeypatch, tmp_path)
    config_path, config = _config(tmp_path)
    first = _record(tmp_path, 1)
    second = _record(tmp_path, 2)
    _write_partial_strategy(tmp_path, 1, "critical_path_restructuring")
    _write_partial_strategy(tmp_path, 2, "bounded_unroll")

    with runner._legacy_execution_hooks(config_path, config):
        selected = runner_legacy.select_refinement_parent([first, second])

    assert selected is not None
    parent, reason = selected
    assert parent["candidate_index"] == 0
    assert parent["strategy_family"] == "memory_parallelism"
    assert reason == runner.STRUCTURED_PARENT_REASON


def test_guard_delegates_when_no_structured_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    record = _record(tmp_path, 1)
    expected = (record, "legacy_result")
    called: list[tuple[list[dict], dict | None]] = []

    def fake_select(records, selection=None):
        materialised = list(records)
        called.append((materialised, selection))
        return expected

    monkeypatch.setattr(runtime, "_ORIGINAL_SELECT", fake_select)

    assert runtime._guarded_select([record], {"mode": "test"}) == expected
    assert called == [([record], {"mode": "test"})]


def test_transition_runtime_installation_is_idempotent() -> None:
    selected = runner_legacy.select_refinement_parent
    runtime.install_structured_transition_runtime()
    runtime.install_structured_transition_runtime()
    assert runner_legacy.select_refinement_parent is selected
