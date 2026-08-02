"""Run external tools consistently and capture their output."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class CommandResult:
    command: tuple[str, ...]
    return_code: int
    output: str

    @property
    def passed(self) -> bool:
        return self.return_code == 0


def run_command(
    command: Sequence[str | Path],
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
    echo: bool = True,
) -> CommandResult:
    rendered = tuple(str(item) for item in command)
    if echo:
        print("Command:", " ".join(rendered), flush=True)

    completed = subprocess.run(
        rendered,
        cwd=cwd,
        env=dict(env) if env is not None else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output = completed.stdout or ""
    if echo and output:
        print(output, end="", flush=True)
    return CommandResult(rendered, completed.returncode, output)
