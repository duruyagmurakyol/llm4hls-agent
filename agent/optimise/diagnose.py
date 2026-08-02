"""Expose the existing generic Vitis analysis through the clean agent API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.analysis.hierarchical_hls_analyzer import analyse_hierarchy


def diagnose_reports(report_root: Path, *, interface_frozen: bool = False) -> dict[str, Any]:
    return analyse_hierarchy(report_root, interface_frozen=interface_frozen)
