#!/usr/bin/env python3

"""Estimate prompt savings from existing repair and optimisation artefacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.prompt_compaction import compact_user_prompt

PROMPT_PATTERNS = ("candidate_*_prompt.txt", "prompt.txt")


def prompt_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    found: set[Path] = set()
    for pattern in PROMPT_PATTERNS:
        found.update(path for path in root.rglob(pattern) if path.is_file())
    return sorted(found)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit conservative HLS prompt compaction without model calls."
    )
    parser.add_argument("path", type=Path)
    parser.add_argument("--show-files", action="store_true")
    args = parser.parse_args()

    files = prompt_files(args.path.expanduser().resolve())
    if not files:
        raise FileNotFoundError(f"No prompt artefacts found under {args.path}")

    original_characters = 0
    compacted_characters = 0
    estimated_original_tokens = 0
    estimated_compacted_tokens = 0

    for path in files:
        _, stats = compact_user_prompt(path.read_text(encoding="utf-8"))
        original_characters += int(stats["original_characters"])
        compacted_characters += int(stats["compacted_characters"])
        estimated_original_tokens += int(stats["estimated_original_tokens"])
        estimated_compacted_tokens += int(stats["estimated_compacted_tokens"])
        if args.show_files:
            print(
                f"{stats['reduction_percent']:6.1f}%  "
                f"{stats['estimated_original_tokens']:6} -> "
                f"{stats['estimated_compacted_tokens']:6} est. tokens  {path}"
            )

    saved_characters = original_characters - compacted_characters
    saved_tokens = estimated_original_tokens - estimated_compacted_tokens
    reduction = (
        100.0 * saved_characters / original_characters
        if original_characters
        else 0.0
    )

    print("\nPrompt compaction audit")
    print("Files:", len(files))
    print("Characters:", original_characters, "->", compacted_characters)
    print(f"Character reduction: {saved_characters} ({reduction:.1f}%)")
    print(
        "Estimated prompt tokens:",
        estimated_original_tokens,
        "->",
        estimated_compacted_tokens,
    )
    print("Estimated tokens saved:", saved_tokens)
    print("Note: token estimates use characters/4; API usage remains authoritative.")


if __name__ == "__main__":
    main()
