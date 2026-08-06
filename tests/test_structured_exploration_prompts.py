from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.optimise import structured_exploration
from agent.optimise.search_policy import EXPLORATION_STRATEGY_FAMILIES


def _config(tmp_path: Path) -> Path:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "candidate_001_prompt.txt").write_text(
        "BASELINE DIAGNOSIS\nComplete baseline source:\nint kernel(int x) { return x; }\n",
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
            }
        ),
        encoding="utf-8",
    )
    return path


def test_three_exploration_prompts_are_distinct_and_baseline_rooted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(structured_exploration, "REPO_ROOT", tmp_path)
    config = _config(tmp_path)

    paths = [
        structured_exploration.prepare_structured_exploration_prompt(
            config,
            candidate_index=index,
            strategy_family=family,
        )
        for index, family in enumerate(EXPLORATION_STRATEGY_FAMILIES, 1)
    ]

    prompts = [path.read_text(encoding="utf-8") for path in paths]
    assert len(set(prompts)) == 3
    assert all("BASELINE DIAGNOSIS" in prompt for prompt in prompts)
    assert all(
        "Implementation parent: original verified baseline (candidate 000)."
        in prompt
        for prompt in prompts
    )

    for index, family in enumerate(EXPLORATION_STRATEGY_FAMILIES, 1):
        prompt = prompts[index - 1]
        assert f"Strategy family: {family}." in prompt
        for other in EXPLORATION_STRATEGY_FAMILIES:
            if other != family:
                assert f"Strategy family: {other}." not in prompt

        strategy = json.loads(
            (tmp_path / "run" / f"candidate_{index:03d}_strategy.json").read_text(
                encoding="utf-8"
            )
        )
        assert strategy["name"] == family
        assert strategy["source_candidate_index"] == 0
        assert strategy["next_candidate_index"] == index
        assert strategy["trigger"] == "structured_baseline_exploration"
        assert strategy["phase"] == "explore"

        feedback = json.loads(
            (tmp_path / "run" / f"candidate_{index:03d}_feedback.json").read_text(
                encoding="utf-8"
            )
        )
        assert feedback["previous_candidate_index"] == 0
        assert feedback["selected_parent"] == "verified_baseline"
        assert feedback["strategy_family"] == family
        assert feedback["structured_schedule"] is True


def test_baseline_prompt_template_is_immutable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(structured_exploration, "REPO_ROOT", tmp_path)
    config = _config(tmp_path)

    first = structured_exploration.prepare_structured_exploration_prompt(
        config,
        candidate_index=1,
        strategy_family="critical_path_restructuring",
    )
    template = tmp_path / "run" / structured_exploration.BASELINE_PROMPT_TEMPLATE
    original = template.read_text(encoding="utf-8")

    # The generated candidate prompt may be replaced or modified later, but the
    # baseline template used by the other exploration slots must not change.
    first.write_text("MODIFIED CANDIDATE PROMPT\n", encoding="utf-8")
    structured_exploration.prepare_structured_exploration_prompt(
        config,
        candidate_index=2,
        strategy_family="bounded_unroll",
    )

    assert template.read_text(encoding="utf-8") == original
    assert "MODIFIED CANDIDATE PROMPT" not in (
        tmp_path / "run" / "candidate_002_prompt.txt"
    ).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("candidate_index", "strategy_family"),
    [
        (1, "bounded_unroll"),
        (2, "memory_parallelism"),
        (3, "critical_path_restructuring"),
        (4, "critical_path_restructuring"),
    ],
)
def test_mismatched_schedule_slot_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    candidate_index: int,
    strategy_family: str,
) -> None:
    monkeypatch.setattr(structured_exploration, "REPO_ROOT", tmp_path)
    config = _config(tmp_path)

    with pytest.raises(ValueError):
        structured_exploration.prepare_structured_exploration_prompt(
            config,
            candidate_index=candidate_index,
            strategy_family=strategy_family,
        )
