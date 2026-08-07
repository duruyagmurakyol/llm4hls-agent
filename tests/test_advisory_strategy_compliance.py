from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.optimise import structured_exploration
from agent.optimise.refinement_strategy import check_strategy_compliance


def test_explicit_advisory_strategy_without_baseline_remains_non_rejecting() -> None:
    report = check_strategy_compliance(
        "int kernel(int x) { return x + 1; }\n",
        {
            "name": "critical_path_restructuring",
            "parameters": {},
            "compliance_mode": "advisory",
        },
    )

    assert report == {
        "required": False,
        "passed": True,
        "strategy": "critical_path_restructuring",
        "reason": "structured_strategy_requires_baseline_for_source_audit",
    }


def test_unknown_strategy_without_advisory_marker_remains_rejected() -> None:
    report = check_strategy_compliance(
        "int kernel(int x) { return x + 1; }\n",
        {"name": "unknown_architecture", "parameters": {}},
    )

    assert report["required"] is True
    assert report["passed"] is False
    assert report["reason"] == "unsupported_strategy"


@pytest.mark.parametrize(
    ("candidate_index", "strategy_family"),
    [
        (1, "critical_path_restructuring"),
        (2, "bounded_unroll"),
        (3, "memory_parallelism"),
    ],
)
def test_structured_prompt_metadata_marks_strategy_advisory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    candidate_index: int,
    strategy_family: str,
) -> None:
    monkeypatch.setattr(structured_exploration, "REPO_ROOT", tmp_path)
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "candidate_001_prompt.txt").write_text(
        "BASELINE DIAGNOSIS\nComplete baseline source:\n"
        "int kernel(int x) { return x; }\n",
        encoding="utf-8",
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "benchmark": "test",
                "top_function": "kernel",
                "output_dir": "run",
                "baseline": {"source": "baseline.cpp"},
            }
        ),
        encoding="utf-8",
    )

    structured_exploration.prepare_structured_exploration_prompt(
        config,
        candidate_index=candidate_index,
        strategy_family=strategy_family,
    )

    strategy = json.loads(
        (output_dir / f"candidate_{candidate_index:03d}_strategy.json").read_text(
            encoding="utf-8"
        )
    )
    assert strategy["compliance_mode"] == "advisory"
    report = check_strategy_compliance(
        "int kernel(int x) { return x + 1; }\n",
        strategy,
    )
    assert report["passed"] is True
    assert report["required"] is False
