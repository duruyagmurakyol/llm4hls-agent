#!/usr/bin/env python3

"""Map requested benchmark model names to valid SiliconFlow API identifiers.

The experiment suite originally used model artifact names containing precision
or quantisation suffixes. SiliconFlow exposes shorter API identifiers. This
migration keeps the requested identity as provenance while replacing only the
provider-facing ``id`` used for API calls.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


MODEL_MAPPINGS: dict[str, dict[str, str]] = {
    "cyankiwi/Qwen3.5-122B-A10B-AWQ-4bit": {
        "provider_id": "Qwen/Qwen3.5-122B-A10B",
        "equivalence": "provider_substitute_precision_variant",
        "requested_precision": "AWQ-4bit",
        "provider_precision": "FP8",
        "note": (
            "The requested AWQ-4bit artifact is not exposed by SiliconFlow. "
            "SiliconFlow's available Qwen3.5-122B-A10B endpoint is documented as FP8."
        ),
    },
    "Qwen/Qwen3.6-27B-FP8": {
        "provider_id": "Qwen/Qwen3.6-27B",
        "equivalence": "provider_alias_same_precision",
        "requested_precision": "FP8",
        "provider_precision": "FP8",
        "note": (
            "SiliconFlow omits the -FP8 suffix from its API identifier while "
            "documenting the hosted Qwen3.6-27B endpoint as FP8."
        ),
    },
}


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read {path}: {error}") from error
    if not isinstance(value, dict) or not isinstance(value.get("models"), list):
        raise RuntimeError(f"Suite definition has no models list: {path}")
    return value


def _backup_path(path: Path) -> Path:
    return path.with_name(path.stem + ".pre_provider_mapping" + path.suffix)


def migrate(path: Path) -> tuple[int, Path | None]:
    path = path.expanduser().resolve()
    suite = _load(path)
    changed = 0

    for raw in suite["models"]:
        if not isinstance(raw, dict):
            continue
        current = str(raw.get("id", "")).strip()
        mapping = MODEL_MAPPINGS.get(current)
        if mapping is None:
            continue

        raw["requested_id"] = current
        raw["id"] = mapping["provider_id"]
        raw["provider_model_id"] = mapping["provider_id"]
        raw["model_equivalence"] = mapping["equivalence"]
        raw["requested_precision"] = mapping["requested_precision"]
        raw["provider_precision"] = mapping["provider_precision"]
        raw["provider_mapping_note"] = mapping["note"]
        changed += 1

    if changed == 0:
        return 0, None

    backup = _backup_path(path)
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(json.dumps(suite, indent=2) + "\n", encoding="utf-8")
    return changed, backup


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replace unavailable requested model artifact names with valid "
            "SiliconFlow API IDs while retaining provenance metadata."
        )
    )
    parser.add_argument("suite", type=Path, nargs="+")
    args = parser.parse_args()

    total = 0
    for path in args.suite:
        changed, backup = migrate(path)
        total += changed
        print(f"{path}: updated {changed} model entr{'y' if changed == 1 else 'ies'}")
        if backup is not None:
            print(f"  original preserved at {backup}")

    if total == 0:
        print("No requested model IDs required migration.")


if __name__ == "__main__":
    main()
