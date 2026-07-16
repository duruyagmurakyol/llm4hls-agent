#!/usr/bin/env python3

from dataclasses import dataclass


@dataclass
class Design:
    name: str
    csim_pass: bool
    synth_pass: bool
    latency_cycles: int
    clock_period_ns: float
    lut: int
    ff: int
    dsp: int


BASELINE = Design(
    name="baseline",
    csim_pass=True,
    synth_pass=True,
    latency_cycles=949,
    clock_period_ns=7.643,
    lut=9366,
    ff=24983,
    dsp=56,
)

DESIGNS = [
    BASELINE,
    Design("candidate_1", True, True, 1954, 37.989, 32484, 38244, 238),
    Design("candidate_2", True, True, 5145, 37.989, 3786, 2566, 31),
    Design("candidate_3a", True, True, 8938, 10.565, 1818, 1627, 14),
    Design("candidate_3b", True, True, 5936, 10.565, 1868, 1823, 14),
    Design("candidate_3c", True, True, 8185, 10.565, 3011, 2204, 28),
]


def is_feasible(design: Design) -> bool:
    return (
        design.csim_pass
        and design.synth_pass
        and design.clock_period_ns <= 10
        and design.dsp <= 80
        and design.lut <= 15000
    )


def score(design: Design) -> float:
    return (
        0.60 * design.latency_cycles / BASELINE.latency_cycles
        + 0.20 * design.lut / BASELINE.lut
        + 0.10 * design.ff / BASELINE.ff
        + 0.10 * design.dsp / BASELINE.dsp
    )


print(
    f"{'Design':<15}"
    f"{'Feasible':<12}"
    f"{'Score':<12}"
    f"{'Latency':<12}"
    f"{'Clock':<12}"
    f"{'LUT':<10}"
    f"{'FF':<10}"
    f"{'DSP':<8}"
)

print("-" * 91)

for design in DESIGNS:
    feasible = is_feasible(design)
    design_score = score(design)

    print(
        f"{design.name:<15}"
        f"{str(feasible):<12}"
        f"{design_score:<12.4f}"
        f"{design.latency_cycles:<12}"
        f"{design.clock_period_ns:<12.3f}"
        f"{design.lut:<10}"
        f"{design.ff:<10}"
        f"{design.dsp:<8}"
    )
