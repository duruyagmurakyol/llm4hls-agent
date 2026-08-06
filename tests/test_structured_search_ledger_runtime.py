from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.optimise import generate as generation
from agent.optimise import runner_legacy
from agent.optimise import search_ledger_runtime as runtime
from agent.optimise.search_ledger import load_search_ledger


def _config(tmp_path: Path, *, structured: bool = True) -> Path:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    baseline = tmp_path / "baseline.cpp"
    baseline.write_text(
        "#include <stdint.h>\nint kernel(int x) { return x; }\n",
        encoding="utf-8",
    )
    payload = {
        "output_dir": str(output_dir),
        "baseline": {"source": str(baseline)},
        "top_function": "kernel",
        "model": {
            "provider": "siliconflow",
            "name": "test-model",
            "temperature": 0.0,
            "max_tokens": 128,
        },
    }
    if structured:
        payload["search_policy"] = {"mode": "structured_v1"}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _slot(
    tmp_path: Path,
    index: int,
    *,
    family: str,
    prompt: str,
    parameters: dict | None = None,
) -> None:
    output_dir = tmp_path / "run"
    (output_dir / f"candidate_{index:03d}_prompt.txt").write_text(
        prompt,
        encoding="utf-8",
    )
    (output_dir / f"candidate_{index:03d}_strategy.json").write_text(
        json.dumps(
            {
                "name": family,
                "parameters": parameters or {},
                "source_candidate_index": 0,
                "next_candidate_index": index,
                "phase": "explore",
            }
        ),
        encoding="utf-8",
    )


def _source(index: int) -> str:
    return (
        "#include <stdint.h>\n"
        f"int kernel(int x) {{ return x + {index}; }}\n"
    )


def test_runtime_installation_is_idempotent() -> None:
    generated = runner_legacy.generate_candidate
    evaluated = runner_legacy.evaluate_experiment
    provider = generation.complete

    runtime.install_structured_search_ledger_runtime()
    runtime.install_structured_search_ledger_runtime()

    assert runner_legacy.generate_candidate is generated
    assert runner_legacy.evaluate_experiment is evaluated
    assert generation.complete is provider
    assert getattr(
        runner_legacy,
        "_structured_search_ledger_runtime_installed",
    ) is True


def test_repeated_branch_is_blocked_before_generation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(generation, "REPO_ROOT", tmp_path)
    config = _config(tmp_path)
    _slot(
        tmp_path,
        1,
        family="bounded_unroll",
        prompt="first prompt\n",
        parameters={"factor": 2},
    )
    _slot(
        tmp_path,
        2,
        family="bounded_unroll",
        prompt="different prompt\n",
        parameters={"factor": 2},
    )
    calls: list[int] = []

    def fake_generate(source, candidate_index=1, *, budget=None):
        del source, budget
        calls.append(candidate_index)
        path = tmp_path / "run" / f"candidate_{candidate_index:03d}.cpp"
        path.write_text(_source(candidate_index), encoding="utf-8")
        return path

    monkeypatch.setattr(runtime, "_ORIGINAL_GENERATE", fake_generate)

    runtime._guarded_generate(config, 1)
    with pytest.raises(runtime.SearchNoveltyRejected):
        runtime._guarded_generate(config, 2)

    assert calls == [1]
    rejection = json.loads(
        (tmp_path / "run" / "candidate_002_novelty_rejection.json").read_text(
            encoding="utf-8"
        )
    )
    assert rejection["reason"] == "branch_already_attempted"
    assert rejection["duplicate_of_candidate_index"] == 1


def test_repeated_exact_prompt_is_blocked_before_provider_call(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(generation, "REPO_ROOT", tmp_path)
    config = _config(tmp_path)
    _slot(
        tmp_path,
        1,
        family="bounded_unroll",
        prompt="identical effective prompt\n",
    )
    _slot(
        tmp_path,
        2,
        family="memory_parallelism",
        prompt="identical effective prompt\n",
    )
    provider_calls: list[str] = []

    def fake_complete(*args, **kwargs):
        del args
        provider_calls.append(kwargs["user_prompt"])
        return SimpleNamespace(content=_source(len(provider_calls)))

    def fake_generate(source, candidate_index=1, *, budget=None):
        del source, budget
        prompt = (
            tmp_path / "run" / f"candidate_{candidate_index:03d}_prompt.txt"
        ).read_text(encoding="utf-8")
        response = runtime._guarded_complete(
            model="test-model",
            system_prompt="system",
            user_prompt=prompt,
            temperature=0.0,
            max_tokens=128,
        )
        path = tmp_path / "run" / f"candidate_{candidate_index:03d}.cpp"
        path.write_text(response.content, encoding="utf-8")
        return path

    monkeypatch.setattr(runtime, "_ORIGINAL_COMPLETE", fake_complete)
    monkeypatch.setattr(runtime, "_ORIGINAL_GENERATE", fake_generate)

    runtime._guarded_generate(config, 1)
    with pytest.raises(runtime.SearchNoveltyRejected):
        runtime._guarded_generate(config, 2)

    assert provider_calls == ["identical effective prompt\n"]
    rejection = json.loads(
        (tmp_path / "run" / "candidate_002_novelty_rejection.json").read_text(
            encoding="utf-8"
        )
    )
    assert rejection["reason"] == "effective_prompt_already_attempted"


def test_generated_duplicate_source_is_recorded_and_retired(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(generation, "REPO_ROOT", tmp_path)
    config = _config(tmp_path)
    _slot(tmp_path, 1, family="bounded_unroll", prompt="prompt one\n")
    _slot(tmp_path, 2, family="memory_parallelism", prompt="prompt two\n")

    def fake_generate(source, candidate_index=1, *, budget=None):
        del source, budget
        path = tmp_path / "run" / f"candidate_{candidate_index:03d}.cpp"
        text = (
            "#include <stdint.h>\nint kernel(int x) { return x + 1; }\n"
            if candidate_index == 1
            else "// same design\n#include <stdint.h>\nint kernel(int x){return x+1;}\n"
        )
        path.write_text(text, encoding="utf-8")
        return path

    monkeypatch.setattr(runtime, "_ORIGINAL_GENERATE", fake_generate)

    runtime._guarded_generate(config, 1)
    runtime._guarded_generate(config, 2)

    attempts = load_search_ledger(tmp_path / "run")["attempts"]
    second = next(item for item in attempts if item["candidate_index"] == 2)
    assert second["source_duplicate_of_candidate_index"] == 1
    assert second["retired"] is True
    assert second["retirement_reason"] == "duplicate_generated_source"


def test_evaluation_verdict_retires_registered_branch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(generation, "REPO_ROOT", tmp_path)
    config = _config(tmp_path)
    _slot(tmp_path, 1, family="bounded_unroll", prompt="prompt one\n")

    def fake_generate(source, candidate_index=1, *, budget=None):
        del source, budget
        path = tmp_path / "run" / f"candidate_{candidate_index:03d}.cpp"
        path.write_text(_source(candidate_index), encoding="utf-8")
        return path

    monkeypatch.setattr(runtime, "_ORIGINAL_GENERATE", fake_generate)
    runtime._guarded_generate(config, 1)

    monkeypatch.setattr(
        runtime,
        "_ORIGINAL_EVALUATE",
        lambda source: {
            "candidates": [
                {
                    "candidate_index": 1,
                    "verdict": "reject_static",
                }
            ]
        },
    )
    runtime._guarded_evaluate(config)

    attempt = load_search_ledger(tmp_path / "run")["attempts"][0]
    assert attempt["completed"] is True
    assert attempt["retired"] is True
    assert attempt["retirement_reason"] == "reject_static"


def test_incomplete_evaluation_does_not_complete_branch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(generation, "REPO_ROOT", tmp_path)
    config = _config(tmp_path)
    _slot(tmp_path, 1, family="bounded_unroll", prompt="prompt one\n")

    def fake_generate(source, candidate_index=1, *, budget=None):
        del source, budget
        path = tmp_path / "run" / f"candidate_{candidate_index:03d}.cpp"
        path.write_text(_source(candidate_index), encoding="utf-8")
        return path

    monkeypatch.setattr(runtime, "_ORIGINAL_GENERATE", fake_generate)
    runtime._guarded_generate(config, 1)
    monkeypatch.setattr(
        runtime,
        "_ORIGINAL_EVALUATE",
        lambda source: {
            "candidates": [
                {
                    "candidate_index": 1,
                    "verdict": "awaiting_cosim",
                }
            ]
        },
    )

    runtime._guarded_evaluate(config)
    attempt = load_search_ledger(tmp_path / "run")["attempts"][0]
    assert attempt["completed"] is False
    assert attempt["verdict"] is None


def test_legacy_config_bypasses_ledger(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(generation, "REPO_ROOT", tmp_path)
    config = _config(tmp_path, structured=False)
    calls: list[int] = []

    def fake_generate(source, candidate_index=1, *, budget=None):
        del source, budget
        calls.append(candidate_index)
        path = tmp_path / "run" / f"candidate_{candidate_index:03d}.cpp"
        path.write_text(_source(candidate_index), encoding="utf-8")
        return path

    monkeypatch.setattr(runtime, "_ORIGINAL_GENERATE", fake_generate)
    result = runtime._guarded_generate(config, 1)

    assert result.is_file()
    assert calls == [1]
    assert not (tmp_path / "run" / "structured_search_ledger.json").exists()
