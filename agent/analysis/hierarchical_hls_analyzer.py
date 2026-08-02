#!/usr/bin/env python3

"""Combine diagnoses from all Vitis HLS synthesis reports in one design."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.analysis.hls_bottleneck_analyzer import analyse
from agent.analysis.vitis_evidence_extractor import extract_evidence


@dataclass
class ReportAnalysis:
    report: str
    function: str
    evidence: dict[str, Any]
    diagnosis: dict[str, Any]
    is_top: bool = False


def discover_reports(root: Path) -> list[Path]:
    reports = [p for p in root.glob("**/*_csynth.xml") if p.name != "csynth.xml"]
    if not reports:
        reports = list(root.glob("**/*csynth.xml"))
    return sorted(set(reports))


def _report_score(path: Path, function: str) -> tuple[int, int, int]:
    name = path.stem.lower()
    function_l = function.lower()
    generated = int(any(token in name for token in ("pipeline_", "loop_", "entry_proc", "exit_proc")))
    depth = len(path.parts)
    return (generated, len(function_l), depth)


def _select_top(analyses: list[ReportAnalysis]) -> ReportAnalysis:
    if not analyses:
        raise ValueError("No report analyses available")
    return min(analyses, key=lambda item: _report_score(Path(item.report), item.function))


def _severity(primary: dict[str, Any], evidence: dict[str, Any], *, is_top: bool) -> float:
    category = primary.get("category", "insufficient_evidence")
    base = {
        "dataflow_stall": 100,
        "external_memory_bandwidth_contention": 96,
        "memory_port_contention": 94,
        "loop_carried_dependency": 92,
        "pipeline_ii_violation": 88,
        "critical_path": 84,
        "resource_pressure": 75,
        "sequential_loop": 70,
        "dominant_latency_region": 40,
        "near_sequential_lower_bound": 5,
        "insufficient_evidence": 1,
    }.get(category, 20)

    loops = evidence.get("loops", []) or []
    max_ii = 1.0
    max_contribution = 0.0
    for loop in loops:
        try:
            max_ii = max(max_ii, float(loop.get("achieved_ii") or 1.0))
        except (TypeError, ValueError):
            pass
        try:
            max_contribution = max(max_contribution, float(loop.get("contribution_to_total_latency") or 0.0))
        except (TypeError, ValueError):
            pass

    top = evidence.get("top_function", {}) or {}
    try:
        latency = float(top.get("latency_cycles") or 0.0)
    except (TypeError, ValueError):
        latency = 0.0

    score = float(base)
    score += min(20.0, max(0.0, max_ii - 1.0) * 5.0)
    score += max_contribution * 5.0
    score += min(5.0, latency / 2000.0)
    if is_top:
        score += 1.0
    if category == "near_sequential_lower_bound":
        score -= 10.0
    return round(score, 3)


def analyse_hierarchy(report_root: Path, *, interface_frozen: bool = False) -> dict[str, Any]:
    reports = discover_reports(report_root)
    if not reports:
        raise FileNotFoundError(f"No synthesis reports found under {report_root}")

    analyses: list[ReportAnalysis] = []
    for report in reports:
        evidence = extract_evidence(report, interface_frozen=interface_frozen)
        diagnosis = analyse(evidence)
        function = str(evidence.get("top_function", {}).get("name") or report.stem.removesuffix("_csynth"))
        analyses.append(
            ReportAnalysis(
                report=str(report.resolve()),
                function=function,
                evidence=evidence,
                diagnosis=diagnosis,
            )
        )

    top = _select_top(analyses)
    top.is_top = True

    ranked: list[dict[str, Any]] = []
    for item in analyses:
        primary = item.diagnosis["primary_diagnosis"]
        loops = item.evidence.get("loops", []) or []
        max_ii = max((float(loop.get("achieved_ii") or 1) for loop in loops), default=1.0)
        ranked.append(
            {
                "function": item.function,
                "report": item.report,
                "is_top": item.is_top,
                "score": _severity(primary, item.evidence, is_top=item.is_top),
                "primary_diagnosis": primary,
                "stop_recommended": item.diagnosis.get("stop_recommended", False),
                "latency_cycles": item.evidence.get("top_function", {}).get("latency_cycles"),
                "interval_cycles": item.evidence.get("top_function", {}).get("interval_cycles"),
                "max_achieved_ii": max_ii,
                "loop_count": len(loops),
            }
        )

    ranked.sort(key=lambda item: item["score"], reverse=True)
    actionable = [
        item for item in ranked
        if item["primary_diagnosis"]["category"] not in {"near_sequential_lower_bound", "insufficient_evidence"}
    ]
    primary = actionable[0] if actionable else ranked[0]

    protected = [
        {
            "function": item["function"],
            "reason": item["primary_diagnosis"]["category"],
        }
        for item in ranked
        if item["primary_diagnosis"]["category"] == "near_sequential_lower_bound"
    ]

    return {
        "schema_version": 1,
        "report_root": str(report_root.resolve()),
        "top_function": top.function,
        "report_count": len(analyses),
        "primary_target": primary,
        "ranked_targets": ranked,
        "protected_regions": protected,
        "recommended_focus": {
            "target": primary["function"],
            "category": primary["primary_diagnosis"]["category"],
            "recommended_transformations": primary["primary_diagnosis"].get("recommended_transformations", []),
            "forbidden_transformations": primary["primary_diagnosis"].get("forbidden_transformations", []),
        },
    }
