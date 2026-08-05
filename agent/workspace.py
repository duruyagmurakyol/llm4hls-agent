"""Create isolated task workspaces without benchmark-specific assumptions."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Workspace:
    root: Path

    def resolve(self, relative_path: str | Path) -> Path:
        return self.root / Path(relative_path)


def create_workspace(
    *,
    repository_root: Path,
    run_dir: Path,
    artifact_paths: Iterable[str | Path],
) -> Workspace:
    """Copy declared task artefacts into one isolated workspace."""
    workspace_root = run_dir / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=False)

    for item in artifact_paths:
        relative = Path(item)
        source = relative if relative.is_absolute() else repository_root / relative
        if not source.exists():
            raise FileNotFoundError(f"Task artefact not found: {source}")

        destination_relative = relative.name if relative.is_absolute() else relative
        destination = workspace_root / destination_relative
        destination.parent.mkdir(parents=True, exist_ok=True)

        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(source, destination)

    return Workspace(workspace_root)
