#!/usr/bin/env python3

"""Benchmark-independent diagnosis of common HLS performance bottlenecks."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Diagnosis:
    category: str
    target: str
    confidence: float
    evidence: list[str] = field(default_factory=list)
    recommended_transformations: list[str] = field(default_factory=list)
    forbidden_transformations: list[str] = field(default_factory=list)
    expected_tradeoffs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _number(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _warning_text(evidence: dict[str, Any]) -> str:
    warnings = evidence.get("warnings", [])
    if isinstance(warnings, str):
        warnings = [warnings]
    return " ".join(str(item).lower() for item in warnings)


def _loops(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    loops = evidence.get("loops", [])
    return [item for item in loops if isinstance(item, dict)]


def _bus_ports(warnings: str) -> list[str]:
    return sorted(set(re.findall(r"port ['\"]([^'\"]+)['\"]", warnings, re.I)))


def _ram_arrays(warnings: str) -> list[str]:
    """Extract RAM-backed arrays named by Vitis scheduling warnings."""

    arrays = set(
        re.findall(
            r"(?:on array\s+['\"]?|accessing core:ram:)([A-Za-z_]\w*)",
            warnings,
            re.I,
        )
    )
    return sorted(arrays)


CATEGORY_PRIORITY = {
    "dataflow_stall": 100,
    "critical_path": 95,
    "external_memory_bandwidth_contention": 92,
    "memory_port_contention": 90,
    "loop_carried_dependency": 88,
    "pipeline_ii_violation": 80,
    "sequential_loop": 75,
    "resource_pressure": 70,
    "near_sequential_lower_bound": 65,
    "dominant_latency_region": 20,
    "insufficient_evidence": 0,
}


def _ranking_key(item: Diagnosis) -> tuple[int, float]:
    """Rank causal diagnoses above contextual observations."""
    return (CATEGORY_PRIORITY.get(item.category, 10), item.confidence)


def analyse(evidence: dict[str, Any]) -> dict[str, Any]:
    """Return ranked diagnoses from generic synthesis/source evidence."""

    diagnoses: list[Diagnosis] = []
    warnings = _warning_text(evidence)
    loops = _loops(evidence)
    constraints = evidence.get("constraints", {}) or {}
    interface_frozen = bool(constraints.get("interface_frozen", False))
    top = evidence.get("top_function", {}) or {}
    top_interval = _number(top.get("interval_cycles"))

    # Vitis can expose decisive bus/interface scheduling evidence even when the
    # csynth XML does not contain a per-loop table (for example after aggressive
    # unrolling or function-level pipelining). Treat this as causal evidence.
    bus_lower_bound = re.search(
        r"lower bound of ii is\s+([0-9]+).*?(?:multiple bus|m_axi)",
        warnings,
        re.I | re.S,
    )
    bus_contention = bool(bus_lower_bound) or (
        "lower bound of ii" in warnings
        and ("bus read operation" in warnings or "bus write operation" in warnings)
    )
    if bus_contention:
        lower_bound = _number(bus_lower_bound.group(1)) if bus_lower_bound else top_interval
        ports = _bus_ports(warnings)
        evidence_items = ["Vitis reports an II lower bound caused by repeated AXI bus accesses"]
        if lower_bound is not None:
            evidence_items.append(f"reported_ii_lower_bound={lower_bound:g}")
        if top_interval is not None:
            evidence_items.append(f"top_interval={top_interval:g} cycles")
        if ports:
            evidence_items.append("contended_ports=" + ",".join(ports))

        recommendations = [
            "remove accidental full unrolling or function-level pipelining that exposes all accesses concurrently",
            "restore a sequential or bounded loop pipeline matched to interface bandwidth",
            "use local buffering or burst-friendly access only when the interface contract permits it",
        ]
        forbidden = []
        if interface_frozen:
            forbidden.append("change external AXI interface architecture")

        diagnoses.append(
            Diagnosis(
                category="external_memory_bandwidth_contention",
                target=",".join(ports) if ports else "external_memory_interfaces",
                confidence=0.98,
                evidence=evidence_items,
                recommended_transformations=recommendations,
                forbidden_transformations=forbidden,
                expected_tradeoffs=[
                    "more requested parallel accesses cannot improve throughput when AXI service rate is the bound"
                ],
            )
        )

    # Vitis also reports on-chip RAM port pressure using messages such as:
    # "Lower bound of II is 19 due to multiple 'load' ... on array 'A' ...
    # accessing core:RAM:A". This is distinct from external AXI bandwidth and
    # should drive local banking/buffering recommendations rather than generic
    # pipeline advice.
    ram_lower_bound = re.search(
        r"lower bound of ii is\s+([0-9]+).*?(?:multiple\s+['\"]?(?:load|store)|accessing core:ram:)",
        warnings,
        re.I | re.S,
    )
    ram_contention = bool(ram_lower_bound) and (
        "accessing core:ram:" in warnings or " on array " in warnings
    )
    if ram_contention:
        lower_bound = _number(ram_lower_bound.group(1))
        arrays = _ram_arrays(warnings)
        evidence_items = [
            "Vitis reports an II lower bound caused by repeated accesses to a RAM-backed array"
        ]
        if lower_bound is not None:
            evidence_items.append(f"reported_ii_lower_bound={lower_bound:g}")
        if arrays:
            evidence_items.append("contended_arrays=" + ",".join(arrays))

        diagnoses.append(
            Diagnosis(
                category="memory_port_contention",
                target=",".join(arrays) if arrays else "local_memory",
                confidence=0.98,
                evidence=evidence_items,
                recommended_transformations=[
                    "match controlled unrolling to available memory ports",
                    "bank or partition a bounded local buffer for the contended access pattern",
                    "restructure accesses or introduce local row/tile buffering before parallel compute",
                ],
                forbidden_transformations=(
                    ["change external interface architecture"] if interface_frozen else []
                ),
                expected_tradeoffs=[
                    "lower II may increase BRAM, LUT, routing and buffering latency"
                ],
            )
        )

    for loop in loops:
        name = str(loop.get("name") or loop.get("label") or "unknown_loop")
        trip = _number(loop.get("trip_count_max", loop.get("trip_count")))
        latency = _number(loop.get("latency_cycles", loop.get("latency")))
        achieved_ii = _number(loop.get("achieved_ii", loop.get("ii")))
        target_ii = _number(loop.get("target_ii"))
        pipelined = loop.get("pipelined")
        contribution = _number(loop.get("contribution_to_total_latency"))

        if achieved_ii is not None and achieved_ii > 1:
            if "memory port" in warnings or "limited port" in warnings or loop.get("memory_port_limited"):
                diagnoses.append(
                    Diagnosis(
                        category="memory_port_contention",
                        target=name,
                        confidence=0.95,
                        evidence=[
                            f"achieved_ii={achieved_ii:g}",
                            "synthesis evidence reports limited memory-port availability",
                        ],
                        recommended_transformations=[
                            "match controlled unrolling to available memory ports",
                            "partition or reshape the contended local array",
                            "restructure accesses or introduce local tiling",
                        ],
                        forbidden_transformations=(
                            ["change external interface architecture"] if interface_frozen else []
                        ),
                        expected_tradeoffs=["lower II may increase BRAM, LUT and routing cost"],
                    )
                )
            elif "depend" in warnings or loop.get("loop_carried_dependency"):
                diagnoses.append(
                    Diagnosis(
                        category="loop_carried_dependency",
                        target=name,
                        confidence=0.92,
                        evidence=[
                            f"achieved_ii={achieved_ii:g}",
                            "dependency evidence prevents the requested pipeline rate",
                        ],
                        recommended_transformations=[
                            "use partial accumulators or a tree reduction",
                            "restructure the recurrence",
                            "interchange or split loops where semantics permit",
                        ],
                        expected_tradeoffs=["more parallel arithmetic may increase DSP/LUT use"],
                    )
                )
            else:
                target_text = f", target_ii={target_ii:g}" if target_ii is not None else ""
                diagnoses.append(
                    Diagnosis(
                        category="pipeline_ii_violation",
                        target=name,
                        confidence=0.72,
                        evidence=[f"achieved_ii={achieved_ii:g}{target_text}"],
                        recommended_transformations=[
                            "inspect scheduling constraints and loop dependencies",
                            "inspect memory access conflicts",
                            "pipeline or restructure the limiting loop body",
                        ],
                        expected_tradeoffs=["lower II can require duplicated operators or memories"],
                    )
                )

        if pipelined is False and trip is not None and trip >= 8:
            diagnoses.append(
                Diagnosis(
                    category="sequential_loop",
                    target=name,
                    confidence=0.86,
                    evidence=[f"trip_count={trip:g}", "loop is reported as not pipelined"],
                    recommended_transformations=[
                        "pipeline the loop",
                        "normalize control flow that blocks pipelining",
                    ],
                    expected_tradeoffs=["pipelining may increase registers and operator duplication"],
                )
            )

        if achieved_ii == 1 and trip is not None and latency is not None:
            ratio = latency / max(trip, 1.0)
            if 0.9 <= ratio <= 1.2:
                forbidden = ["add another pipeline pragma as the primary optimisation"]
                if interface_frozen:
                    forbidden.append("change external interface architecture")
                diagnoses.append(
                    Diagnosis(
                        category="near_sequential_lower_bound",
                        target=name,
                        confidence=0.9,
                        evidence=[
                            "achieved_ii=1",
                            f"latency={latency:g} cycles is close to trip_count={trip:g}",
                        ],
                        recommended_transformations=[
                            "test bounded parallelism only if memory access concurrency exists",
                            "otherwise stop and report limited optimisation headroom",
                        ],
                        forbidden_transformations=forbidden,
                        expected_tradeoffs=[
                            "unrolling without matching memory bandwidth may increase resources with no speedup"
                        ],
                    )
                )

        if contribution is not None and contribution >= 0.6:
            diagnoses.append(
                Diagnosis(
                    category="dominant_latency_region",
                    target=name,
                    confidence=min(0.95, 0.65 + contribution / 3),
                    evidence=[f"loop contributes approximately {contribution:.0%} of total latency"],
                    recommended_transformations=["prioritise this loop before modifying minor regions"],
                )
            )

    clock = evidence.get("clock", {}) or {}
    target_period = _number(clock.get("target_period_ns"))
    estimated_period = _number(clock.get("estimated_period_ns"))
    if target_period is not None and estimated_period is not None and estimated_period > target_period:
        diagnoses.append(
            Diagnosis(
                category="critical_path",
                target=str(clock.get("critical_region") or "top_function"),
                confidence=0.93,
                evidence=[
                    f"estimated_period_ns={estimated_period:g}",
                    f"target_period_ns={target_period:g}",
                ],
                recommended_transformations=[
                    "split long expressions into pipeline stages",
                    "use balanced arithmetic trees",
                    "reduce fan-out or operator chaining",
                ],
                expected_tradeoffs=["extra pipeline stages can increase latency cycles and FF use"],
            )
        )

    resources = evidence.get("resources", {}) or {}
    for key in ("lut", "ff", "dsp", "bram"):
        used = _number(resources.get(f"{key}_used"))
        available = _number(resources.get(f"{key}_available"))
        if used is not None and available and used / available >= 0.8:
            diagnoses.append(
                Diagnosis(
                    category="resource_pressure",
                    target=key.upper(),
                    confidence=0.9,
                    evidence=[f"{key}_utilisation={used / available:.1%}"],
                    recommended_transformations=[
                        "reduce excessive unrolling or partitioning",
                        "share operators where throughput permits",
                        "narrow data types where functionally safe",
                    ],
                    expected_tradeoffs=["resource reduction may increase latency or II"],
                )
            )

    if "dataflow" in warnings and ("stall" in warnings or "deadlock" in warnings):
        diagnoses.append(
            Diagnosis(
                category="dataflow_stall",
                target="dataflow_region",
                confidence=0.94,
                evidence=["synthesis/runtime evidence reports dataflow stalls or deadlock"],
                recommended_transformations=[
                    "identify the slow producer or consumer stage",
                    "adjust FIFO depth",
                    "rebalance stage latency or stream access order",
                ],
                expected_tradeoffs=["deeper FIFOs consume additional BRAM/LUTRAM"],
            )
        )

    if not diagnoses:
        diagnoses.append(
            Diagnosis(
                category="insufficient_evidence",
                target="design",
                confidence=0.35,
                evidence=["available aggregate evidence does not identify a specific bottleneck"],
                recommended_transformations=[
                    "extract per-loop latency and II",
                    "collect scheduling warnings and memory-port evidence",
                    "avoid speculative source edits until evidence improves",
                ],
            )
        )

    diagnoses.sort(key=_ranking_key, reverse=True)
    primary = diagnoses[0]
    return {
        "schema_version": 1,
        "primary_diagnosis": primary.to_dict(),
        "ranked_diagnoses": [item.to_dict() for item in diagnoses],
        "stop_recommended": primary.category in {
            "near_sequential_lower_bound",
            "insufficient_evidence",
        },
    }
