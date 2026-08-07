from __future__ import annotations

import json
from pathlib import Path

from agent.optimise.search_policy import (
    DEFAULT_EXPLORATION_STRATEGY_FAMILIES,
    build_structured_search_schedule,
)
from agent.optimise.strategy_selector import select_exploration_strategy_families


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_schedule_default_is_unchanged() -> None:
    schedule = build_structured_search_schedule(max_candidates=5)

    assert [item["strategy_family"] for item in schedule[:3]] == list(
        DEFAULT_EXPLORATION_STRATEGY_FAMILIES
    )
    assert schedule[3]["phase"] == "exploit"
    assert schedule[3]["parent_selector"] == "best_refinement_eligible_candidate"
    assert schedule[4]["phase"] == "recover"
    assert schedule[4]["parent_selector"] == "best_recoverable_candidate_or_baseline"


def test_no_evidence_preserves_historical_exploration(tmp_path: Path) -> None:
    selected, audit = select_exploration_strategy_families(tmp_path)

    assert selected == DEFAULT_EXPLORATION_STRATEGY_FAMILIES
    assert audit["fallback_default"] is True


def test_nested_ii1_memory_pressure_selects_buffered_parallelism(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "baseline_hierarchical_diagnosis.json",
        {
            "primary_target": {
                "primary_diagnosis": {"category": "dominant_latency_region"}
            },
            "ranked_targets": [
                {
                    "primary_diagnosis": {
                        "category": "near_sequential_lower_bound"
                    }
                }
            ],
        },
    )
    _write_json(
        tmp_path / "baseline_source_cause.json",
        {
            "primary_hypothesis": {
                "category": "memory_access_or_port_pressure",
                "confidence": 0.7,
            }
        },
    )

    selected, audit = select_exploration_strategy_families(tmp_path)

    assert selected == (
        "buffered_parallelism",
        "critical_path_restructuring",
        "memory_parallelism",
    )
    assert "bounded_unroll" not in selected
    assert audit["fallback_default"] is False


def test_explicit_memory_port_contention_keeps_memory_first(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "baseline_hierarchical_diagnosis.json",
        {
            "primary_target": {
                "primary_diagnosis": {"category": "memory_port_contention"}
            },
            "ranked_targets": [],
        },
    )

    selected, _ = select_exploration_strategy_families(tmp_path)

    assert selected == (
        "memory_parallelism",
        "buffered_parallelism",
        "critical_path_restructuring",
    )


def test_custom_layer_one_schedule_keeps_baseline_parents_and_tail() -> None:
    schedule = build_structured_search_schedule(
        max_candidates=5,
        exploration_strategy_families=(
            "buffered_parallelism",
            "critical_path_restructuring",
            "memory_parallelism",
        ),
    )

    assert [item["parent_candidate_index"] for item in schedule[:3]] == [0, 0, 0]
    assert schedule[3]["parent_candidate_index"] is None
    assert schedule[4]["parent_candidate_index"] is None
