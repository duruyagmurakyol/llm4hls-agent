from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
U55C_PART = "xcu55c-fsvh2892-2L-e"
PART_PATTERN = re.compile(r"\bxc[a-z0-9]+-[a-z0-9]+(?:-[a-z0-9]+)+\b", re.IGNORECASE)


def _tracked_files() -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=REPO_ROOT
    )
    return [
        REPO_ROOT / entry.decode("utf-8")
        for entry in output.split(b"\0")
        if entry
    ]


def test_no_tracked_file_retains_legacy_zu3eg_part() -> None:
    # Build the legacy identifier in pieces so this guard does not match itself.
    legacy = ("xczu3eg-" + "sfvc784-2-e").encode("ascii")
    offenders: list[str] = []
    for path in _tracked_files():
        if path.is_file() and legacy in path.read_bytes():
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, "legacy ZU3EG target remains in: " + ", ".join(offenders)


def test_all_tracked_xilinx_part_tokens_are_u55c() -> None:
    offenders: list[str] = []
    for path in _tracked_files():
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for part in sorted(set(PART_PATTERN.findall(text))):
            if part.lower() != U55C_PART.lower():
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {part}")
    assert not offenders, "non-U55C Xilinx parts remain in tracked files: " + ", ".join(offenders)


def test_all_benchmark_cfg_parts_are_u55c() -> None:
    offenders: list[str] = []
    for path in (REPO_ROOT / "benchmarks").rglob("*.cfg"):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip().startswith("part="):
                part = line.split("=", 1)[1].strip()
                if part != U55C_PART:
                    offenders.append(f"{path.relative_to(REPO_ROOT)}: {part}")
    assert not offenders, "non-U55C benchmark cfg targets: " + ", ".join(offenders)


def test_all_benchmark_tcl_set_part_targets_are_u55c() -> None:
    offenders: list[str] = []
    pattern = re.compile(r"\bset_part\s+\{?([^\s}]+)")
    for path in (REPO_ROOT / "benchmarks").rglob("*.tcl"):
        text = path.read_text(encoding="utf-8", errors="replace")
        for part in pattern.findall(text):
            if part != U55C_PART:
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {part}")
    assert not offenders, "non-U55C benchmark TCL targets: " + ", ".join(offenders)


def _walk_json(value: object, path: str = "") -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else key
            if key in {"part", "target_part"} and isinstance(child, str):
                found.append((child_path, child))
            found.extend(_walk_json(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_walk_json(child, f"{path}[{index}]"))
    return found


def test_all_config_json_part_fields_are_u55c() -> None:
    offenders: list[str] = []
    for path in (REPO_ROOT / "configs").rglob("*.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for field, part in _walk_json(value):
            if part.startswith("xc") and part != U55C_PART:
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{field}={part}")
    assert not offenders, "non-U55C JSON target fields: " + ", ".join(offenders)
