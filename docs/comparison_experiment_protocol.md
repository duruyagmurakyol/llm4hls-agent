# Controlled comparison experiment protocol

## Purpose

This experiment compares three levels of LLM4HLS assistance under the same model, FPGA target, verification rules and maximum budget:

1. **One-shot**: one model response with no iterative tool feedback.
2. **Generic iterative feedback**: repeated model calls using only the latest compiler, C simulation, synthesis or co-simulation feedback, without the full agent's structured diagnosis, Pareto archive, parent selection, duplicate detection or specialised recovery policies.
3. **Full agent**: the complete repair-to-optimisation framework with structured diagnosis, staged verification, budget accounting, candidate history, deterministic selection and safe baseline fallback.

The experiment is intended to measure whether the full agent improves repair reliability, verification success, optimisation quality and cost efficiency over simpler prompting baselines.

## Status

This protocol is prepared while the Alveo U55C device package is being installed. No comparison run should begin until the U55C synthesis smoke test passes for:

```text
xcu55c-fsvh2892-2L-e
```

The comparison results must remain separate from the frozen 36-run reference sweep, which targeted `xczu3eg-sfvc784-2-e`.

## Representative benchmark subset

The comparison uses five structurally and behaviourally distinct tasks:

| Benchmark | Case | Reason for inclusion |
|---|---|---|
| vector add | simple arithmetic or indexing fault | Small sanity case and low-cost end-to-end check |
| dot product | accumulator overwrite | Repeatable repair and optimisation success in the reference experiments |
| GEMM | missing final `k` iteration | Larger nested-loop design where the verified baseline was often retained |
| GESUMMV | accumulator overwrite | Imported HLS-Eval kernel with successful optimisation in some repetitions |
| MVT | shifted second-vector access | Difficult imported case with no fully verified optimisation candidate in the reference sweep |

The exact vector-add manifest must be fixed before execution and recorded in the run plan. The other four cases should be derived from the existing U55C representative manifests.

## Compared methods

### One-shot

The model receives the task specification, editable source and fixed context once. It returns one complete source file.

The candidate then passes through the same independent validation sequence as every other method:

1. host validation where configured;
2. Vitis C simulation;
3. synthesis;
4. C/RTL co-simulation;
5. frequency and resource checks.

There is no retry after failure and no optimisation refinement loop.

### Generic iterative feedback

The model starts from the same initial prompt as the one-shot method. After each failed or non-selected candidate, it receives a compact generic message containing:

- the previous candidate source;
- the latest failing stage;
- concise tool evidence;
- current baseline and candidate PPA metrics when available; and
- the remaining budget.

It may retry until the common model-call or token budget is exhausted. It must not use:

- hierarchical diagnosis;
- benchmark-specific recovery rules;
- specialised resource-frequency balancing;
- Pareto parent selection;
- duplicate escape strategies;
- synthesis-equivalence rejection as a prompt strategy; or
- full-agent trajectory-derived strategy names.

The same static validation and independent Vitis tools still apply.

### Full agent

The full agent uses the current frozen framework, including:

- automatic repair routing;
- structured diagnosis;
- verified-baseline promotion;
- static validation;
- duplicate and synthesis-equivalence detection;
- candidate history and Pareto state;
- deterministic parent and final-design selection;
- hard frequency and resource constraints;
- safe baseline fallback; and
- authoritative budget accounting.

No new benchmark-specific rule may be added after the comparison begins.

## Controlled settings

All three methods use the following shared settings:

| Setting | Value |
|---|---|
| Provider | SiliconFlow |
| Model | `Qwen/Qwen3.5-122B-A10B` |
| Temperature | `0.0` |
| Thinking mode | disabled |
| Maximum generated tokens per call | `2048` |
| FPGA board | AMD Alveo U55C |
| FPGA part | `xcu55c-fsvh2892-2L-e` |
| Vitis HLS version | `2025.2` |
| Target clock | `10 ns` |
| Minimum frequency | `100 MHz` |
| Maximum model calls per run | `8` |
| Maximum total model tokens per run | `32,768` |
| Maximum C simulation calls | `10` |
| Maximum synthesis calls | `10` |
| Maximum co-simulation calls | `10` |
| Prompt compaction | enabled consistently for all three methods, or disabled consistently for all three |

The prompt-compaction setting must be fixed before the first run and recorded in the suite metadata.

## Fair-budget interpretation

The maximum budget is identical across methods, but methods are charged only for calls they actually make.

- One-shot normally consumes one model call.
- Generic iterative feedback may consume up to eight calls.
- Full agent may consume up to eight calls shared across repair and optimisation.

A method is not required to spend the full budget. Lower cost is a valid advantage when final quality is comparable.

## Verification and selection policy

All generated designs are evaluated by the same independent tools and hard constraints. A final design is accepted only if it:

- preserves the top-level interface;
- passes static checks;
- passes C simulation;
- synthesises successfully;
- passes C/RTL co-simulation;
- achieves at least 100 MHz; and
- remains within U55C resource limits.

For one-shot and generic feedback, final selection must use the same deterministic ranking function as the full agent. This prevents the comparison from using a weaker evaluator for the baseline methods.

If no generated candidate is eligible but a fully verified repaired or original baseline exists, the baseline may be retained. Baseline retention must be reported separately from optimisation success.

## Repetitions

Recommended initial experiment:

- 5 benchmark cases;
- 3 methods;
- 1 repetition;
- 15 total runs.

This is sufficient for a first controlled comparison while limiting model and Vitis cost. If time and budget allow, repeat all 15 conditions three times for 45 runs total.

Hosted-model non-determinism means a single repetition should be described as a controlled representative comparison, not a definitive statistical result.

## Primary outcomes

For each method and case, record:

- successful fully verified final design;
- repair success;
- optimised candidate selected or baseline retained;
- number of generated candidates;
- number of fully verified candidates;
- final latency and throughput period;
- LUT, FF, DSP and BRAM use;
- frequency and constraint compliance;
- input, output and total tokens;
- model, CSim, synthesis and co-simulation calls;
- runtime; and
- termination reason.

## Aggregate metrics

Report these across the five cases:

1. fully verified final-design rate;
2. repair success rate;
3. optimisation-selection rate;
4. mean and median latency change for selected optimisation candidates;
5. resource trade-offs for selected candidates;
6. mean tokens and model calls per run;
7. mean synthesis and co-simulation calls per run;
8. mean runtime per run;
9. baseline-fallback rate; and
10. failure-stage distribution.

Do not average PPA deltas from baseline-retained runs as zero-valued improvements. Report fallbacks separately.

## Hypotheses

The expected findings are:

- one-shot has the lowest cost but the lowest recovery rate on difficult faults;
- generic iterative feedback improves repair success but may repeat ineffective or equivalent candidates;
- the full agent achieves the highest fully verified final-design rate and produces better-controlled optimisation trade-offs;
- the full agent may use more calls than one-shot, but structured validation and safe fallback should reduce invalid final outputs;
- difficult cases such as MVT should expose whether the extra controller structure provides real benefit rather than only additional token use.

These are hypotheses, not claims, until the comparison is executed.

## Execution order

After U55C installation completes:

1. verify `get_parts` returns `xcu55c-fsvh2892-2L-e`;
2. rerun the no-model-token U55C synthesis smoke test;
3. complete the four-case U55C representative validation;
4. freeze the exact vector-add case;
5. generate method-specific manifests with separate output directories;
6. dry-run the 15-condition comparison suite;
7. execute runs sequentially;
8. extract authoritative results from each task's `budget_summary.json`, `experiment_summary.json` and `unified_agent_result.json`; and
9. produce one method-by-case table and one aggregate comparison table.

## Naming and isolation

Use a new run identifier, for example:

```text
u55c_method_comparison_<UTC timestamp>
```

Every method/case combination must have a unique task ID and output directory. Do not reuse cached candidate, synthesis or co-simulation state across methods.

## Completion criteria

This comparison step is complete when:

- all planned conditions have a recorded final outcome;
- all methods used the same model, target, hard constraints and maximum budget;
- tool and token costs come from authoritative per-task budget files;
- final design selection used the same evaluator;
- failures and baseline fallbacks are reported explicitly; and
- the comparison archive is kept separate from the frozen reference and U55C target-validation runs.
