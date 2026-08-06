from __future__ import annotations

import copy
import json
from pathlib import Path

from agent.config import TaskManifest, load_task
from agent.optimise.config_source import as_config_source, ppa_config_from_task
from agent.optimise.runner import structured_search_enabled


REPO_ROOT = Path(__file__).resolve().parents[1]
VECTOR_TASK = REPO_ROOT / "configs" / "tasks" / "vector_add_track_a.json"


def test_vector_add_manifest_declares_structured_search() -> None:
    task = load_task(VECTOR_TASK)
    assert task.data["search_policy"] == {"mode": "structured_v1"}


def test_ppa_adapter_preserves_explicit_search_policy() -> None:
    task = load_task(VECTOR_TASK)
    config = ppa_config_from_task(task)

    assert config["search_policy"] == {"mode": "structured_v1"}
    assert config["budget"]["max_candidates"] == 5
    assert structured_search_enabled(config) is True


def test_in_memory_config_carries_search_policy_to_runner() -> None:
    task = load_task(VECTOR_TASK)
    source = as_config_source(task)
    config = json.loads(source.read_text(encoding="utf-8"))

    assert config["search_policy"]["mode"] == "structured_v1"
    assert structured_search_enabled(config) is True


def test_adapter_does_not_invent_policy_for_legacy_manifest() -> None:
    task = load_task(VECTOR_TASK)
    data = copy.deepcopy(task.data)
    data.pop("search_policy")
    legacy = TaskManifest(path=task.path, data=data)

    config = ppa_config_from_task(legacy)

    assert "search_policy" not in config
    assert structured_search_enabled(config) is False
