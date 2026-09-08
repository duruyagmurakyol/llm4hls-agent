# Final reported results

This directory contains one consolidated result set for LLM4HLS-Agent. The same experimental evidence is used across the dissertation and the two-page competition paper; the files below differ only in level of detail and machine-readability, not in experimental provenance.

The frozen controller revision for the reported experiments is commit `30b8107`. Later commits on `main` are documentation and repository-cleanup changes.

## Breadth across tasks and models

The pre-specified breadth experiment used 20 tasks and three fixed models, producing 60 model–task runs under the same controller.

| Group | Rows | Completed | Completion rate |
|---|---:|---:|---:|
| Generation / repair / structural | 42 | 39 | 92.9% |
| PPA run completion | 18 | 15 | 83.3% |
| **Overall** | **60** | **54** | **90.0%** |

The six non-completions were confined to three ATAX PPA rows with an invalid starting optimisation baseline and three wrong-top interface rows. For PPA rows, `Completed` means the controller reached a valid terminal state; it does not imply that an improved candidate displaced the verified baseline.

The detailed task × model breakdown is retained in [`overnight_60_matrix.md`](overnight_60_matrix.md) and [`overnight_60_matrix.csv`](overnight_60_matrix.csv). The canonical experiment definition is [`../configs/suites/overnight_60.json`](../configs/suites/overnight_60.json).

## Optimisation outcomes

The optimisation table below is the complete Table I result set used in the two-page paper. Negative latency or throughput deltas indicate improvement; positive resource deltas indicate increased resource use. The BICG infeasible point is intentionally included because it demonstrates hard-resource rejection before recovery.

| Kernel | Model | Δ latency | Δ throughput period | Δ LUT | Δ FF | Δ DSP |
|---|---|---:|---:|---:|---:|---:|
| Vector add | Qwen | -73.39% | -66.73% | +182.83% | -40.0% | — |
| ATAX | Qwen | -5.93% | -5.93% | -0.62% | -2.96% | +6.82% |
| BICG recovery | Qwen | -61.84% | -61.77% | +227.01% | +158.05% | +300% |
| BICG Pareto | Qwen | -40.7% | -40.7% | +47.5% | +17.2% | +100% |
| BICG infeasible* | Qwen | -55.94% | -55.88% | +891.3% | +545.5% | +904.5% |
| Vector add | DeepSeek | -35.1% | -45.9% | +31.3% | 0.0% | — |
| Vector add | Kimi | -35.1% | -45.9% | +31.3% | 0.0% | — |
| Dot product | DeepSeek | -69.3% | -69.1% | +182.5% | +219.6% | — |
| Dot product | Qwen | -62.4% | -62.3% | +420.8% | +480.4% | — |
| FIR-small | DeepSeek | -64.8% | -64.6% | +50.3% | +69.4% | — |
| Stencil2D | DeepSeek | -57.6% | — | +190.4% | — | — |
| Stencil2D | Qwen | -41.2% | — | +99.8% | — | — |
| Stencil2D | Kimi | -23.4% | — | +39.1% | — | — |
| Transpose | DeepSeek | -81.2% | — | +220.6% | — | — |
| Transpose | Kimi | -81.2% | — | +97.9% | — | — |
| Transpose | Qwen | -49.3% | — | +51.5% | — | — |
| Prefix | DeepSeek | -25.3% | — | +23.7% | — | — |
| Prefix | Kimi | -25.3% | — | +23.7% | — | — |
| Prefix | Qwen | -5.4% | — | +193.9% | — | — |
| Histogram | DeepSeek | -9.1% | — | +15.4% | — | — |
| Histogram | Kimi | -9.1% | — | +15.4% | — | — |
| Conv2D | Qwen | -0.89% | — | — | — | — |

`*` Rejected by the hard resource gates. The candidate reduced BICG latency substantially but exceeded LUT, FF and DSP ceilings; measured resource feedback then produced the compliant BICG recovery point.

The machine-readable version of this table is [`optimisation_results.csv`](optimisation_results.csv).

## BICG resource-aware recovery

The central constrained case study starts from a verified baseline at 7420.560 ns, 7,626 LUT, 20,191 FF and 44 DSP. An aggressive candidate reached 3269.760 ns but expanded to 75,596 LUT, 130,326 FF and 442 DSP, exceeding the configured ceilings of 34,964 LUT, 86,908 FF and 224 DSP. Resource-aware recovery then produced the selected 2831.816 ns candidate using 24,938 LUT, 52,102 FF and 176 DSP at an estimated 131.72 MHz. This is a 61.84% reduction in HLS-estimated latency relative to the baseline while remaining within all configured hard limits.

An earlier verified BICG Pareto point at approximately -40.7% latency with +47.5% LUT, +17.2% FF and +100% DSP is retained as a distinct speed/area operating point.

## GEMM negative control

A GEMM candidate reduced scheduled cycle count but degraded achievable clock sufficiently that HLS-estimated latency increased by approximately 137%. It was therefore rejected. This is the main negative-control example showing why realised post-synthesis timing and feasibility, rather than cycle count alone, govern selection.

## Additional controlled studies

The same project evaluation also includes two secondary studies used in the dissertation and supporting pages of the competition paper:

- staged-feedback repair: iterative repair succeeded in 90/90 tested model–task cases, versus 1/90 for the corresponding one-shot configuration. Iterative repair receives newly exposed feedback and additional calls, so this is not an equal-budget comparison;
- repeated full-agent sweep: 36/36 completed runs, 204 generated candidates, 116 fully verified candidates, 240 model calls and 347,075 tokens.

These studies are part of the same evaluation package, but are secondary to the 60-row breadth experiment and the optimisation evidence above.
