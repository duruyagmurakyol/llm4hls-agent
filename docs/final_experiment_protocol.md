# Final experiment protocol

## Status

This document freezes the evaluation definition used for the final repair-to-optimisation experiment. Changes to benchmarks, model settings, budgets, constraints, ranking, validation or success criteria require a new protocol revision and a new run identifier.

Reference suite index: `configs/tasks/combined_full_agent/index.json`  
Completed reference run: `combined_full_agent_20260805_051650`  
Protocol branch: `refactor/unified-agent`

The completed run contains 12 cases with three independent repetitions per case, giving 36 runs in total.

## Research question

Can one budgeted LLM-based HLS agent:

1. detect whether a submitted design requires repair;
2. repair functional or compile-time faults;
3. establish a fully verified baseline;
4. search for hardware-relevant PPA alternatives under hard FPGA constraints; and
5. return the best fully verified design, including retaining the baseline when no candidate ranks above it?

## Benchmarks and injected faults

| Benchmark | Fault | Top function | Numerical equivalence |
|---|---|---|---|
| dot product | accumulator overwrite | `dot_product` | exact integer comparison |
| dot product | shifted indexing of `b` | `dot_product` | exact integer comparison |
| dot product | missing final loop iteration | `dot_product` | exact integer comparison |
| dot product | staged compile error then shifted indexing | `dot_product` | exact integer comparison |
| GEMM | wrong accumulation sign | `kernel_gemm` | absolute tolerance `1e-8` |
| GEMM | shifted indexing of `B` | `kernel_gemm` | absolute tolerance `1e-8` |
| GEMM | missing final `k` iteration | `kernel_gemm` | absolute tolerance `1e-8` |
| GEMM | staged compile error then shifted indexing | `kernel_gemm` | absolute tolerance `1e-8` |
| GEMVER | transposed-access fault | `kernel_gemver` | absolute tolerance `1e-8` |
| GESUMMV | accumulator overwrite | `kernel_gesummv` | absolute tolerance defined by its task manifest and self-checking testbench |
| MVT | shifted second-vector access | `kernel_mvt` | absolute tolerance defined by its task manifest and self-checking testbench |
| SYRK | missing diagonal scaling | `kernel_syrk` | absolute tolerance defined by its task manifest and self-checking testbench |

The four imported HLS-Eval cases are generated from `sharc-lab/hls-eval` at upstream commit `adea9ff46ab3dea51a8e1790b9d8c4da7899275b`. Imported benchmark material remains local; the tracked manifests preserve provenance and regeneration information.

## Repetitions

Each case is executed three times from a fresh derived task manifest and fresh output directory. Runs are sequential to avoid Vitis workspace and licence interference.

- Cases: 12
- Repetitions per case: 3
- Total runs: 36
- Cross-run candidate or synthesis state reuse: prohibited

## Agent workflow

Every automatic task follows this sequence:

1. initial C simulation;
2. route to repair if initial validation fails;
3. host repair validation;
4. independent Vitis C simulation;
5. post-repair synthesis;
6. post-repair C/RTL co-simulation;
7. promotion of the fully verified repaired or initial source to the active baseline;
8. iterative PPA candidate generation and validation;
9. deterministic selection of the best fully verified design.

A failed or inferior optimisation attempt must not replace a verified baseline.

## Model configuration

| Setting | Value |
|---|---|
| Provider | SiliconFlow |
| Model | `Qwen/Qwen3.5-122B-A10B` |
| Temperature | `0.0` |
| Maximum output tokens per call | `2048` |
| API timeout | `180` seconds |
| Thinking mode | disabled |
| Prompt compaction in the completed reference run | disabled; evaluated separately after the reference sweep |

Temperature zero reduces sampling variation but does not guarantee bit-for-bit determinism from the hosted service. Repetitions therefore remain necessary.

## HLS target

| Setting | Value |
|---|---|
| Tool | AMD Vitis HLS |
| Tool version | `2025.2` |
| FPGA part | `xczu3eg-sfvc784-2-e` |
| Target clock period | `10.0 ns` |
| Minimum accepted estimated frequency | `100 MHz` |

The part above is the target used by the completed reference experiment. It is not yet a claim that this is the final competition-required target. U55C or other competition-target requirements must be resolved separately before submission; a target change requires a new protocol revision and representative reruns.

## Hard constraints

Every selected design must satisfy all of the following:

- preserve the exact top-level function signature;
- preserve all testbench-observed functional behaviour;
- modify only the declared editable source file;
- pass static validation;
- pass Vitis C simulation;
- produce a valid top-level synthesis report;
- pass C/RTL co-simulation;
- achieve an estimated frequency of at least `100 MHz`;
- remain within the configured resource ceilings;
- avoid complete partitioning of top-level interface arrays;
- use a hardware-relevant change rather than comments, renaming or a no-op rewrite.

Resource ceilings:

| Resource | Maximum |
|---|---:|
| LUT | 70,560 |
| FF | 141,120 |
| DSP | 360 |
| BRAM18K | 432 |

## Shared per-run budget

One authoritative budget is shared across repair and optimisation.

| Resource | Maximum per run |
|---|---:|
| Agent iterations | 8 |
| Model calls | 8 |
| C simulation calls | 10 |
| Synthesis calls | 10 |
| C/RTL co-simulation calls | 10 |
| Total model tokens | 32,768 |

The authoritative consumed values are written to each task's `budget_summary.json`.

## Candidate eligibility

An optimisation candidate is fully verified only when all of these are true:

- static validation passed;
- C simulation passed;
- synthesis passed and produced usable metrics;
- C/RTL co-simulation passed;
- the frequency threshold passed; and
- resource limits passed.

Candidates with missing objective metrics, duplicate source, failed verification, constraint violations or no useful objective gain may remain in the trajectory for analysis but are not eligible to replace the baseline.

## PPA objectives and trade-offs

The measured objectives are:

- latency in nanoseconds;
- throughput period in nanoseconds;
- LUT usage;
- FF usage;
- DSP usage; and
- BRAM usage.

A candidate that improves at least one objective without worsening another dominates the baseline. A fully verified candidate that improves one objective while worsening another is retained as a Pareto trade-off. Therefore, an "optimised selection" does not necessarily mean that every PPA metric improved.

## Final selection policy

Selection is deterministic and lexicographic. The default ranking is:

1. fully verified status;
2. frequency compliance;
3. resource-limit compliance;
4. latency in nanoseconds;
5. scalar resource cost;
6. throughput period in nanoseconds;
7. total model tokens;
8. tool-call count;
9. tool execution time; and
10. candidate index as the final stable tie-breaker.

For resource cost, task resource ceilings are used to normalise usage when limits are configured. Only fully verified, frequency-compliant and resource-compliant Pareto records are eligible for the best-PPA selection. If no candidate ranks above the baseline, the baseline is retained.

## Definitions of success

### Repair success

Repair succeeds when the resulting source:

- passes the host test;
- passes independent Vitis C simulation;
- passes post-repair synthesis; and
- passes post-repair C/RTL co-simulation.

### Optimisation success

An optimisation candidate is successful only if it is fully verified and satisfies all hard constraints. It may be either a dominating design or a valid Pareto trade-off.

### Complete run success

A complete agent run succeeds when it terminates with a fully verified final design. The final design may be:

- an eligible optimisation candidate; or
- the verified baseline retained as a safe fallback.

Baseline retention is not classified as a run failure.

### Reported outcome categories

- `optimised_candidate`: selected candidate index greater than zero;
- `baseline_retained`: selected candidate index zero;
- `unselected`: no valid final selection, treated as a failed final outcome.

## Authoritative evidence

For each run, the authoritative records are:

- `unified_agent_result.json` for run status, termination and trajectory;
- `budget_summary.json` for model, token and tool-call accounting;
- `experiment_summary.json` for baseline metrics, candidate records, Pareto state and final selection;
- `candidate_state.json` and `candidate_archive/` for materialised selected and best-so-far sources;
- model metadata and validation artefacts for candidate-level evidence.

Suite-level reporting must read these task records rather than relying on stale compatibility fields in the suite summary.

## Frozen reference result

The completed reference sweep produced:

- 36/36 runs with a successful fully verified final design;
- 16 optimised candidate selections;
- 20 verified baseline fallbacks;
- 204 evaluated candidates;
- 116 fully verified optimisation candidates;
- 240 model calls;
- 263,248 input tokens;
- 83,827 output tokens;
- 347,075 total tokens;
- 205 C simulation calls;
- 168 synthesis calls;
- 152 C/RTL co-simulation calls; and
- 4 hours, 4 minutes and 28 seconds total runtime.

These figures describe the reference suite, not the expected cost of one normal user task. Per-run means and individual run records must be reported alongside suite totals.

## Freeze rule

The reference evaluation is frozen. Do not change benchmark-specific recovery logic, prompts, budgets, ranking, validation criteria, target settings or task manifests and then combine the new results with this reference sweep. Any such change starts a separately named experiment and must retain its own protocol revision, run identifier and result archive.
