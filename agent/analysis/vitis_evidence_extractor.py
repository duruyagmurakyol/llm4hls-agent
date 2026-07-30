#!/usr/bin/env python3

"""Extract benchmark-independent optimisation evidence from Vitis HLS outputs."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _number(text: str | None) -> int | float | None:
    if text is None:
        return None
    cleaned = text.strip().replace(",", "")
    if not cleaned or cleaned in {"-", "?", "N/A", "n/a"}:
        return None
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return int(value) if value.is_integer() else value


def _first(root: ET.Element, names: Iterable[str]) -> int | float | None:
    wanted = {name.lower() for name in names}
    for element in root.iter():
        if _local(element.tag).lower() in wanted:
            value = _number(element.text)
            if value is not None:
                return value
    return None


def _text_first(root: ET.Element, names: Iterable[str]) -> str | None:
    wanted = {name.lower() for name in names}
    for element in root.iter():
        if _local(element.tag).lower() in wanted and element.text:
            text = element.text.strip()
            if text:
                return text
    return None


def find_csynth_xml(root: Path) -> Path:
    if root.is_file():
        if root.name.endswith("csynth.xml"):
            return root
        raise FileNotFoundError(f"Not a csynth XML report: {root}")
    candidates = sorted(root.glob("**/*csynth.xml"))
    if not candidates:
        raise FileNotFoundError(f"No *csynth.xml report found under {root}")
    preferred = [p for p in candidates if not any(part.startswith(".") for part in p.parts)]
    return (preferred or candidates)[-1]


def _loop_nodes(root: ET.Element) -> list[ET.Element]:
    nodes: list[ET.Element] = []
    for element in root.iter():
        tag = _local(element.tag).lower()
        if tag in {"loop", "looplatency", "loop_latency", "loopperformance"}:
            nodes.append(element)
            continue
        child_names = {_local(child.tag).lower() for child in element}
        if {"tripcount", "pipelineii"} & child_names and {"latency", "loopname", "name"} & child_names:
            nodes.append(element)
    return nodes


def _child_value(node: ET.Element, names: Iterable[str]) -> int | float | None:
    wanted = {name.lower() for name in names}
    for child in node.iter():
        if child is node:
            continue
        if _local(child.tag).lower() in wanted:
            value = _number(child.text)
            if value is not None:
                return value
    return None


def _child_text(node: ET.Element, names: Iterable[str]) -> str | None:
    wanted = {name.lower() for name in names}
    for child in node.iter():
        if child is node:
            continue
        if _local(child.tag).lower() in wanted and child.text:
            text = child.text.strip()
            if text:
                return text
    return None


def extract_loops(root: ET.Element, top_latency: int | float | None) -> list[dict[str, Any]]:
    loops: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for index, node in enumerate(_loop_nodes(root), start=1):
        name = (
            node.attrib.get("name")
            or node.attrib.get("label")
            or _child_text(node, ("LoopName", "Name", "Label"))
            or f"loop_{index}"
        )
        latency = _child_value(node, ("Latency", "LoopLatency", "Worst-caseLatency", "WorstLatency"))
        trip = _child_value(node, ("TripCount", "TripCountMax", "MaxTripCount"))
        achieved_ii = _child_value(node, ("PipelineII", "AchievedII", "II", "Interval-min", "MinInterval"))
        target_ii = _child_value(node, ("PipelineTargetII", "TargetII"))
        pipelined_text = _child_text(node, ("Pipelined", "PipelineType"))
        pipelined: bool | None = None
        if pipelined_text is not None:
            lowered = pipelined_text.lower()
            if lowered in {"yes", "true", "1", "pipeline", "pipelined"}:
                pipelined = True
            elif lowered in {"no", "false", "0", "none", "not pipelined"}:
                pipelined = False
        if pipelined is None and achieved_ii is not None:
            pipelined = True

        key = (name, latency, trip, achieved_ii, target_ii)
        if key in seen:
            continue
        seen.add(key)
        contribution = None
        if latency is not None and top_latency not in (None, 0):
            contribution = min(1.0, float(latency) / float(top_latency))
        loops.append(
            {
                "name": str(name),
                "trip_count": trip,
                "latency_cycles": latency,
                "achieved_ii": achieved_ii,
                "target_ii": target_ii,
                "pipelined": pipelined,
                "contribution_to_total_latency": contribution,
            }
        )
    return loops


def collect_warnings(search_root: Path, limit: int = 100) -> list[str]:
    if search_root.is_file():
        search_root = search_root.parent
    patterns = ("*.log", "*.rpt")
    warnings: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        for path in sorted(search_root.glob(f"**/{pattern}")):
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for line in text.splitlines():
                if not re.search(r"\b(?:warning|critical warning|deadlock|stall|dependenc|memory port|limited port)\b", line, re.I):
                    continue
                normalized = " ".join(line.strip().split())
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    warnings.append(normalized)
                if len(warnings) >= limit:
                    return warnings
    return warnings


def extract_evidence(report_root: Path, *, interface_frozen: bool = False) -> dict[str, Any]:
    xml_path = find_csynth_xml(report_root)
    root = ET.parse(xml_path).getroot()

    latency_best = _first(root, ("Best-caseLatency", "BestLatency"))
    latency_worst = _first(root, ("Worst-caseLatency", "WorstLatency"))
    interval_min = _first(root, ("Interval-min", "MinInterval"))
    interval_max = _first(root, ("Interval-max", "MaxInterval"))
    target_period = _first(root, ("TargetClockPeriod", "ClockPeriod"))
    estimated_period = _first(root, ("EstimatedClockPeriod", "EstimatedPeriod"))

    resources = {
        "lut_used": _first(root, ("LUT",)),
        "ff_used": _first(root, ("FF",)),
        "dsp_used": _first(root, ("DSP", "DSP48E")),
        "bram_used": _first(root, ("BRAM_18K", "BRAM")),
    }

    available_aliases = {
        "lut_available": ("AvailableLUT", "LUTAvailable"),
        "ff_available": ("AvailableFF", "FFAvailable"),
        "dsp_available": ("AvailableDSP", "DSPAvailable", "AvailableDSP48E"),
        "bram_available": ("AvailableBRAM", "BRAMAvailable", "AvailableBRAM_18K"),
    }
    for key, aliases in available_aliases.items():
        resources[key] = _first(root, aliases)

    top_latency = latency_worst if latency_worst is not None else latency_best
    evidence = {
        "schema_version": 1,
        "source_report": str(xml_path),
        "top_function": {
            "name": _text_first(root, ("TopModelName", "TopFunction", "FunctionName")),
            "latency_best_cycles": latency_best,
            "latency_worst_cycles": latency_worst,
            "latency_cycles": top_latency,
            "interval_min_cycles": interval_min,
            "interval_max_cycles": interval_max,
            "interval_cycles": interval_max if interval_max is not None else interval_min,
        },
        "loops": extract_loops(root, top_latency),
        "resources": resources,
        "clock": {
            "target_period_ns": target_period,
            "estimated_period_ns": estimated_period,
        },
        "warnings": collect_warnings(report_root),
        "constraints": {
            "interface_frozen": interface_frozen,
        },
        "extraction": {
            "loop_count": 0,
            "warning_count": 0,
            "missing_fields": [],
        },
    }
    evidence["extraction"]["loop_count"] = len(evidence["loops"])
    evidence["extraction"]["warning_count"] = len(evidence["warnings"])

    missing: list[str] = []
    if top_latency is None:
        missing.append("top_function.latency_cycles")
    if not evidence["loops"]:
        missing.append("loops")
    if target_period is None:
        missing.append("clock.target_period_ns")
    evidence["extraction"]["missing_fields"] = missing
    return evidence
