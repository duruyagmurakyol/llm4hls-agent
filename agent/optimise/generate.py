"""Compatibility adapter for the existing PPA candidate generator."""

from __future__ import annotations

import sys
from pathlib import Path

from agent.tools.command_runner import CommandResult, run_command


def run_candidate_generator(
    *,
    repository_root: Path,
    config_path: Path,
    extra_arguments: list[str] | None = None,
) -> CommandResult:
    command = [
        sys.executable,
        str(repository_root / "scripts" / "generate_ppa_candidate.py"),
        str(config_path),
        *(extra_arguments or []),
    ]
    return run_command(command, cwd=repository_root)
