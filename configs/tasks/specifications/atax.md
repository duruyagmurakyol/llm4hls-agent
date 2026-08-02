# ATAX HLS task specification

## Objective

Repair or optimise the supplied AMD Vitis HLS implementation of the ATAX kernel while preserving functional correctness. Correctness must be established before PPA optimisation.

## Interface contract

Top function: `kernel_atax`

The exact function signature and C linkage in the supplied source must be preserved. The implementation operates on fixed ATAX dimensions represented by `m = 38` and `n = 42` in the supplied design and testbench.

## Correctness

The supplied public testbench is the first correctness gate. A candidate must compile and pass C simulation before synthesis results may influence the optimisation policy. All array accesses must remain within the declared dimensions.

## Target

- Tool: AMD Vitis HLS 2025.2
- FPGA part: `xczu3eg-sfvc784-2-e`
- Requested clock: 10 ns

## Optimisation objective

Minimise latency and initiation interval while controlling LUT, FF, DSP, and BRAM usage. Preserve non-dominated solutions rather than collapsing the task into a single hard-coded scalar score.

## Budget policy

The agent must terminate within the task manifest's model, C simulation, co-simulation, synthesis, and iteration budgets. Duplicate candidates must be rejected before expensive tool calls. Failed correctness candidates must not be synthesized.
