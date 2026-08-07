from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "maintenance" / "map_siliconflow_model_ids.py"


def _module():
    name = "map_siliconflow_model_ids"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_requested_model_ids_map_to_valid_provider_ids_with_provenance(
    tmp_path: Path,
) -> None:
    module = _module()
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "id": "deepseek-ai/DeepSeek-V4-Pro",
                        "slug": "deepseek",
                        "provider": "siliconflow",
                    },
                    {
                        "id": "cyankiwi/Qwen3.5-122B-A10B-AWQ-4bit",
                        "slug": "qwen35",
                        "provider": "siliconflow",
                    },
                    {
                        "id": "Qwen/Qwen3.6-27B-FP8",
                        "slug": "qwen36",
                        "provider": "siliconflow",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    changed, backup = module.migrate(suite)
    result = json.loads(suite.read_text(encoding="utf-8"))

    assert changed == 2
    assert backup is not None and backup.is_file()
    assert result["models"][0]["id"] == "deepseek-ai/DeepSeek-V4-Pro"

    qwen35 = result["models"][1]
    assert qwen35["requested_id"] == "cyankiwi/Qwen3.5-122B-A10B-AWQ-4bit"
    assert qwen35["id"] == "Qwen/Qwen3.5-122B-A10B"
    assert qwen35["requested_precision"] == "AWQ-4bit"
    assert qwen35["provider_precision"] == "FP8"
    assert qwen35["model_equivalence"] == "provider_substitute_precision_variant"

    qwen36 = result["models"][2]
    assert qwen36["requested_id"] == "Qwen/Qwen3.6-27B-FP8"
    assert qwen36["id"] == "Qwen/Qwen3.6-27B"
    assert qwen36["requested_precision"] == qwen36["provider_precision"] == "FP8"
    assert qwen36["model_equivalence"] == "provider_alias_same_precision"


def test_mapping_is_idempotent(tmp_path: Path) -> None:
    module = _module()
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "id": "Qwen/Qwen3.6-27B-FP8",
                        "slug": "qwen36",
                        "provider": "siliconflow",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    first, _ = module.migrate(suite)
    second, second_backup = module.migrate(suite)

    assert first == 1
    assert second == 0
    assert second_backup is None
