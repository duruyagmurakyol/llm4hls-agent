# Results reported in the two-page LLM4HLS-Agent paper

This file reproduces the experimental results that appear in the **two-page main paper** of *LLM4HLS-Agent: Budgeted Autonomous Repair and Multi-Objective Optimisation for HLS* (pages 1–2 of the competition report).

The frozen controller revision used for the reported competition experiment is commit `30b8107`. The current `main` branch contains later documentation-only commits.

> Scope note: the report also contains references and supporting appendices after the two-page main paper. Results that appear only in those appendices (for example the 90/90 staged-feedback ablation and the 36-run repeated sweep) are not counted here as two-page-paper results.

## 1. Principal breadth result

The main paper reports a pre-specified matrix of 20 tasks × 3 models = 60 isolated model–task runs under the same controller.

| Group | Rows | Completed | Completion rate |
|---|---:|---:|---:|
| Generation / repair / structural | 42 | 39 | 92.9% |
| PPA run completion | 18 | 15 | 83.3% |
| **Overall** | **60** | **54** | **90.0%** |

The six non-completions reported in the paper were confined to:

- 3 × ATAX PPA rows with an invalid starting optimisation baseline;
- 3 × wrong-top interface rows.

The paper states that the successful non-PPA rows span the remaining 13 generation/repair/structural tasks. The full task × model matrix is retained separately in [`overnight_60_matrix.md`](overnight_60_matrix.md) and [`overnight_60_matrix.csv`](overnight_60_matrix.csv); that detailed matrix corresponds to the supporting benchmark matrix rather than the two-page main-paper table.

For PPA rows, `Completed` means the controller reached a valid terminal state. It does **not** imply that an improved candidate necessarily displaced the verified baseline.

## 2. Table I — representative verified optimisation outcomes

Negative latency or throughput deltas indicate improvement. `*` marks the BICG candidate rejected by hard resource gates.

| Kernel | Model | Δ latency | Δ throughput | Δ LUT | Δ FF | Δ DSP | Status |
|---|---|---:|---:|---:|---:|---:|---|
| Vector add | Qwen | -73.39% | -66.73% | +182.83% | -40.0% | — | verified |
| ATAX | Qwen | -5.93% | -5.93% | -0.62% | -2.96% | +6.82% | verified |
| BICG recovery | Qwen | -61.84% | -61.77% | +227.01% | +158.05% | +300% | verified and selected |
| BICG Pareto | Qwen | -40.7% | -40.7% | +47.5% | +17.2% | +100% | verified Pareto point |
| BICG infeasible* | Qwen | -55.94% | -55.88% | +891.3% | +545.5% | +904.5% | rejected: resource infeasible |
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

The machine-readable form is [`two_page_paper_optimisation_table.csv`](two_page_paper_optimisation_table.csv).

The paper uses these rows to illustrate several distinct optimisation regimes: compute-dominated vector-add, dot-product and FIR speed-ups; memory/locality trade-offs in Stencil2D and Transpose; dependency-constrained Prefix and Histogram cases; and explicit hard-resource recovery in BICG.

## 3. Cross-model trade-offs explicitly discussed in the paper

The paper calls out the following comparisons from Table I:

- **Dot product:** DeepSeek reaches -69.3% latency versus Qwen at -62.4%, while also using substantially less added LUT/FF.
- **Stencil2D:** DeepSeek (-57.6% latency, +190.4% LUT), Qwen (-41.2%, +99.8%), and Kimi (-23.4%, +39.1%) form an area–latency ladder.
- **Transpose:** DeepSeek and Kimi both reach -81.2% latency, but Kimi uses +97.9% LUT versus DeepSeek's +220.6%.
- **Prefix:** DeepSeek and Kimi reach -25.3% latency with +23.7% LUT, whereas Qwen reaches only -5.4% while using +193.9% LUT.

These comparisons motivate measured Pareto retention rather than selecting candidates from model identity or transformation description alone.

## 4. GEMM negative-control result

The main paper reports GEMM as a negative control for cycle-only ranking. One generated candidate reduced cycle count but degraded achievable clock enough that realised post-synthesis latency increased by approximately **137%**. The candidate was therefore rejected.

This result is used to support ranking by realised HLS timing and feasibility evidence rather than cycle count or source-level intent alone.

## 5. BICG resource-aware recovery

BICG is the main recovery case study in the two-page paper.

| State | HLS-estimated latency | Throughput | LUT | FF | DSP | Outcome |
|---|---:|---:|---:|---:|---:|---|
| Verified baseline | 7420.56 ns | 7428.132 ns | 7,626 | 20,191 | 44 | feasible baseline |
| Aggressive candidate | 3269.76 ns | — | 75,596 | 130,326 | 442 | rejected: resource infeasible |
| Recovered candidate | 2831.816 ns | — | 24,938 | 52,102 | 176 | selected |
| Configured ceiling | — | — | 34,964 | 86,908 | 224 | hard limit |

The recovered candidate:

- reduces HLS-estimated latency by **61.84%** relative to the verified baseline;
- remains within the configured LUT, FF and DSP ceilings;
- reaches an estimated Fmax of **131.72 MHz**, above the 100 MHz minimum requirement.

The paper also reports that the BICG run exhausted its five-call model budget only after the 2831.816 ns candidate had already been verified and selected, so budget exhaustion stopped further exploration rather than compromising correctness.

An earlier verified BICG Pareto point is retained at approximately **-40.7% latency**, **+47.5% LUT**, **+17.2% FF** and **+100% DSP**, preserving a lower-area operating point alongside the faster recovered design.

## 6. Headline quantitative results from the abstract/conclusion

The two-page paper foregrounds the following quantities:

- **54/60 (90.0%)** overall breadth completion;
- **39/42 (92.9%)** non-PPA completion;
- **15/18 (83.3%)** PPA run completion;
- **73.4%** vector-add latency reduction;
- **5.9%** ATAX latency reduction;
- **61.8%** BICG latency reduction, from 7420.56 ns to 2831.816 ns, while satisfying the configured resource ceilings and 100 MHz requirement.

These values are repetitions of the same evidence above rather than separate experiments.

## 7. Scope

This file is intentionally limited to results reported on **pages 1–2 of the two-page main paper**. The supporting appendices contain additional evidence, including the detailed 20-task benchmark matrix, the staged-feedback repair ablation, and the repeated 36-run sweep; those are retained separately and should not be described as results displayed in the two-page main paper itself.
