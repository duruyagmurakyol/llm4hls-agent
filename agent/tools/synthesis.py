"""Generic adapter for existing synthesis entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from agent.tools.command_runner import CommandResult, run_command


def run_synthesis_adapter(
    command: Sequence[str | Path],
    *,
    repository_root: Path,
) -> CommandResult:
    """Run an existing synthesis script while the implementation is migrated."""
    return run_command(command, cwd=repository_root)
