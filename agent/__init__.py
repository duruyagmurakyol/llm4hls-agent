"""Stable public import surface for the organised LLM4HLS agent package.

Implementation modules are grouped by responsibility under ``core/``,
``competition/`` and ``support/``.  Historical imports such as
``agent.config`` remain valid so experiment manifests, tests and downstream
scripts do not need to know about the physical package layout.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _normalise_repo_root(module: ModuleType) -> None:
    """Preserve repository-relative behaviour after moving a module deeper."""

    old_root = getattr(module, "REPO_ROOT", None)
    if not isinstance(old_root, Path):
        return

    module.REPO_ROOT = _REPO_ROOT

    # A small number of reporting helpers capture REPO_ROOT in function
    # defaults at import time.  Replace only defaults that are exactly the old
    # module root; unrelated Path defaults are left untouched.
    for value in vars(module).values():
        if not callable(value):
            continue
        defaults = getattr(value, "__defaults__", None)
        if defaults:
            value.__defaults__ = tuple(
                _REPO_ROOT if item == old_root else item for item in defaults
            )
        kwdefaults = getattr(value, "__kwdefaults__", None)
        if isinstance(kwdefaults, dict):
            for key, item in list(kwdefaults.items()):
                if item == old_root:
                    kwdefaults[key] = _REPO_ROOT


def _publish(name: str, target: str) -> ModuleType:
    module = import_module(target)
    _normalise_repo_root(module)
    sys.modules[f"{__name__}.{name}"] = module
    globals()[name] = module
    return module


# Foundational modules first.  This order keeps imports deterministic while the
# compatibility surface is being populated.
config = _publish("config", "agent.core.config")
state = _publish("state", "agent.core.state")
budget = _publish("budget", "agent.core.budget")
failures = _publish("failures", "agent.core.failures")
prompt_compaction = _publish("prompt_compaction", "agent.support.prompt_compaction")
onboarding = _publish("onboarding", "agent.support.onboarding")

# Selection must be available before the optimisation runner imports archive.
track_a_scoring = _publish("track_a_scoring", "agent.competition.track_a_scoring")
track_a_selection = _publish("track_a_selection", "agent.competition.track_a_selection")
archive = _publish("archive", "agent.optimise.archive")

baseline = _publish("baseline", "agent.core.baseline")
controller = _publish("controller", "agent.core.controller")
execution_mode = _publish("execution_mode", "agent.core.execution_mode")
final_cosim = _publish("final_cosim", "agent.competition.final_cosim")
stage_aware = _publish("stage_aware", "agent.competition.stage_aware")
track_a = _publish("track_a", "agent.competition.track_a")
onboarding_safe = _publish("onboarding_safe", "agent.support.onboarding_safe")
resume = _publish("resume", "agent.core.resume")
reporting = _publish("reporting", "agent.support.reporting")
terminal_reporting = _publish("terminal_reporting", "agent.support.terminal_reporting")
workspace = _publish("workspace", "agent.support.workspace")

__all__ = [
    "archive",
    "baseline",
    "budget",
    "config",
    "controller",
    "execution_mode",
    "failures",
    "final_cosim",
    "onboarding",
    "onboarding_safe",
    "prompt_compaction",
    "reporting",
    "resume",
    "stage_aware",
    "state",
    "terminal_reporting",
    "track_a",
    "track_a_scoring",
    "track_a_selection",
    "workspace",
]
