from __future__ import annotations

import json
from pathlib import Path

from agent.optimise.structured_transition_runtime import _exploration_family


def test_transition_guard_preserves_old_family_mapping_without_plan(tmp_path: Path) -> None:
    assert _exploration_family(tmp_path, 1) == "critical_path_restructuring"
    assert _exploration_family(tmp_path, 2) == "bounded_unroll"
    assert _exploration_family(tmp_path, 3) == "memory_parallelism"


def test_transition_guard_follows_persisted_diagnosis_plan(tmp_path: Path) -> None:
    (tmp_path / "structured_exploration_plan.json").write_text(
        json.dumps(
            {
                "selected_strategy_families": [
                    "buffered_parallelism",
                    "critical_path_restructuring",
                    "memory_parallelism",
                ]
            }
        ),
        encoding="utf-8",
    )

    assert _exploration_family(tmp_path, 1) == "buffered_parallelism"
    assert _exploration_family(tmp_path, 2) == "critical_path_restructuring"
    assert _exploration_family(tmp_path, 3) == "memory_parallelism"
