# `results/`

This directory contains the curated experimental evidence used for the final LLM4HLS-Agent evaluation. The dissertation and the two-page competition paper report the same underlying result set; the files here differ only in level of detail and machine-readability.

The frozen controller revision for the reported experiments is commit `30b8107`. Later commits on `main` are documentation and repository-cleanup changes.

## Final reported results

Start with [`final_results.md`](final_results.md). It is the canonical human-readable summary and contains:

- the pre-specified 20-task × 3-model breadth experiment: 54/60 completed overall, including 39/42 generation/repair/structural rows and 15/18 PPA rows;
- the complete optimisation table used in the two-page paper, including cross-model speed/area trade-offs;
- the BICG infeasible-to-recovered trajectory and configured resource ceilings;
- the GEMM timing-regression negative control;
- the staged-feedback repair ablation and repeated full-agent sweep as secondary controlled studies.

Supporting files provide the same result set at finer granularity:

- [`optimisation_results.csv`](optimisation_results.csv) — machine-readable form of the complete optimisation table;
- [`overnight_60_matrix.md`](overnight_60_matrix.md) — detailed human-readable 20-task × 3-model breadth matrix;
- [`overnight_60_matrix.csv`](overnight_60_matrix.csv) — machine-readable 60-row breadth matrix;
- [`../configs/suites/overnight_60.json`](../configs/suites/overnight_60.json) — canonical experiment definition for the breadth study.

For PPA rows, `completed` means that the controller reached a valid terminal state. It does not imply that an improved candidate displaced the verified baseline.

## Provenance and interpretation

Retained results should remain traceable to the benchmark/task, source revision, configuration, Vitis version and target, model/provider, original reports, and the acceptance or rejection decision. Compare candidates under identical synthesis settings, report timing regressions as well as improvements, and keep meaningful rejected candidates when they explain controller behaviour.

Generated working runs and rejected/debugging artefacts live under `experiments/`; `results/` is the smaller curated reporting surface.
