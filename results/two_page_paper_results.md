# Results reported in the two-page LLM4HLS-Agent paper

This file collects the experimental results that were explicitly reported in the two-page LLM4HLS-Agent competition paper and its appended supporting pages. It is intended as a human-readable index to the retained result evidence in this repository.

The frozen controller revision for the reported competition experiment is commit `30b8107`. The current `main` branch may contain later documentation-only commits.

## 1. Principal 60-run breadth result

The pre-specified breadth experiment used 20 tasks and three fixed models, producing 60 isolated model–task runs under the same controller.

| Group | Rows | Completed | Completion rate |
|---|---:|---:|---:|
| Generation / repair / structural | 42 | 39 | 92.9% |
| PPA run completion | 18 | 15 | 83.3% |
| **Overall** | **60** | **54** | **90.0%** |

The six non-completions were confined to:

- 3 × ATAX PPA rows with an invalid starting optimisation baseline;
- 3 × wrong-top interface rows.

The full task × model matrix is retained separately in [`overnight_60_matrix.md`](overnight_60_matrix.md) and [`overnight_60_matrix.csv`](overnight_60_matrix.csv).

For PPA rows, `Completed` means that the controller reached a valid terminal state. It may therefore include returning the unchanged verified baseline and does **not** imply that an improved candidate displaced the baseline.

## 2. Representative verified optimisation outcomes

The following table reproduces the optimisation result table reported in the paper. Negative latency or throughput deltas indicate improvement. The BICG infeasible row is intentionally retained because the paper used it to demonstrate hard-resource rejection and subsequent recovery.

| Kernel | Model | Δ latency | Δ throughput period | Δ LUT | Δ FF | Δ DSP | Status |
|---|---|---:|---:|---:|---:|---:|---|
| Vector add | Qwen | -73.39% | -66.73% | +182.83% | -40.0% | — | verified |
| ATAX | Qwen | -5.93% | -5.93% | -0.62% | -2.96% | +6.82% | verified |
| BICG recovery | Qwen | -61.84% | -61.77% | +227.01% | +158.05% | +300% | verified and selected |
| BICG Pareto | Qwen | -40.7% | -40.7% | +47.5% | +17.2% | +100% | verified Pareto point |
| BICG infeasible | Qwen | -55.94% | -55.88% | +891.3% | +545.5% | +904.5% | rejected: resource infeasible |
| Vector add | DeepSeek | -35.1% | -45.9% | +31.3% | 0.0% | — | verified |
| Vector add | Kimi | -35.1% | -45.9% | +31.3% | 0.0% | — | verified |
| Dot product | DeepSeek | -69.3% | -69.1% | +182.5% | +219.6% | — | verified |
| Dot product | Qwen | -62.4% | -62.3% | +420.8% | +480.4% | — | verified |
| FIR-small | DeepSeek | -64.8% | -64.6% | +50.3% | +69.4% | — | verified |
| Stencil2D | DeepSeek | -57.6% | — | +190.4% | — | — | verified |
| Stencil2D | Qwen | -41.2% | — | +99.8% | — | — | verified |
| Stencil2D | Kimi | -23.4% | — | +39.1% | — | — | verified |
| Transpose | DeepSeek | -81.2% | — | +220.6% | — | — | verified |
| Transpose | Kimi | -81.2% | — | +97.9% | — | — | verified |
| Transpose | Qwen | -49.3% | — | +51.5% | — | — | verified |
| Prefix | DeepSeek | -25.3% | — | +23.7% | — | — | verified |
| Prefix | Kimi | -25.3% | — | +23.7% | — | — | verified |
| Prefix | Qwen | -5.4% | — | +193.9% | — | — | verified |
| Histogram | DeepSeek | -9.1% | — | +15.4% | — | — | verified |
| Histogram | Kimi | -9.1% | — | +15.4% | — | — | verified |
| Conv2D | Qwen | -0.89% | — | — | — | — | verified |

The machine-readable form of this table is [`two_page_paper_optimisation_table.csv`](two_page_paper_optimisation_table.csv).

## 3. BICG resource-aware recovery

The paper used BICG as its central recovery example.

| State | HLS-estimated latency | LUT | FF | DSP | Outcome |
|---|---:|---:|---:|---:|---|
| Verified baseline | 7420.56 ns | 7,626 | 20,191 | 44 | feasible baseline |
| Aggressive candidate | 3269.76 ns | 75,596 | 130,326 | 442 | rejected |
| Recovered candidate | 2831.816 ns | 24,938 | 52,102 | 176 | selected |
| Configured ceiling | — | 34,964 | 86,908 | 224 | hard limit |

The recovered candidate reduced HLS-estimated latency by **61.84%** relative to the verified baseline and had an estimated Fmax of **131.72 MHz**, satisfying the 100 MHz requirement and all configured LUT, FF and DSP ceilings.

The paper also retained an earlier verified BICG Pareto point at approximately **-40.7% latency**, **+47.5% LUT**, **+17.2% FF** and **+100% DSP**. This point and the faster recovered design represent different speed/area operating choices.

The BICG run exhausted its five-call model budget only after the 2831.816 ns candidate had already been verified and selected, so budget exhaustion stopped further exploration rather than invalidating the result.

## 4. GEMM negative-control result

The paper reported GEMM as a negative control for cycle-only ranking. A generated candidate reduced scheduled cycle count but degraded achievable clock sufficiently that realised post-synthesis latency increased by approximately **137%** in the competition observation. The candidate was therefore rejected rather than promoted on cycle count alone.

This result supports the policy that optimisation is ranked using realised HLS timing and feasibility evidence rather than source-level intent or cycle count in isolation.

## 5. Secondary repair and repeated-run evidence

The paper's supporting material also reported two secondary studies that informed the controller design.

### Staged-feedback repair ablation

Across Qwen, DeepSeek and Kimi:

- iterative repair: **90/90** successful staged-fault cases;
- corresponding one-shot repair: **1/90** successful cases.

Iterative repair receives newly exposed tool feedback and additional model calls after earlier faults are corrected, so this is not an equal-call or equal-token comparison.

### Repeated full-agent sweep

A 12-task sweep with three repetitions per task produced 36 complete runs:

| Measurement | Result |
|---|---:|
| Completed runs | 36/36 |
| Generated candidates | 204 |
| Fully verified candidates | 116 |
| Model calls | 240 |
| Total tokens | 347,075 |

These values correspond to approximately 6.67 model calls and 9.64k tokens per completed run, with 56.9% of generated candidates reaching full verification.

## 6. Scope of this summary

This file reproduces the results that were explicitly displayed or numerically discussed in the two-page paper and its supporting appended pages. The architecture figure is not duplicated here because it describes controller structure rather than an experimental result. The repository's underlying experiment definitions, source artefacts and retained reports remain the authoritative provenance for individual measurements.
