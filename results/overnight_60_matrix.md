# Frozen 60-run breadth evaluation matrix

This table records the terminal-status matrix used for the dissertation breadth evaluation.
The canonical experiment definition is `configs/suites/overnight_60.json`: 20 tasks × 3 fixed models = 60 model–task runs.
The frozen controller revision for the reported experiment is commit `30b8107`.

**Interpretation.** `Completed` means the controller reached a valid terminal state for the task contract. For PPA rows, completion may mean returning the unchanged verified baseline; it does **not** imply that an improved candidate displaced the baseline. `Not completed` is reserved for rows that did not reach a valid terminal state.

| # | Task | Role / subtype | DeepSeek-V4-Pro | Qwen3.5-122B-A10B | Qwen3.6-27B |
|---:|---|---|---|---|---|
| 1 | `vector_add_generate` | generation / `specification_to_kernel` | Completed | Completed | Completed |
| 2 | `synth_fix_dynamic_buffer` | synthesis_repair / `unsupported_dynamic_allocation` | Completed | Completed | Completed |
| 3 | `projection_bugfix` | repair / `functional` | Completed | Completed | Completed |
| 4 | `residual_stream_deadlock` | structural_repair / `cosim_stream_deadlock` | Completed | Completed | Completed |
| 5 | `structural_blind_stream` | structural_repair / `blind_cosim_stream_deadlock` | Completed | Completed | Completed |
| 6 | `dotProduct_optimize` | optimisation / `ppa` | Completed | Completed | Completed |
| 7 | `atax` | optimisation / `ppa` | Not completed | Not completed | Not completed |
| 8 | `bicg` | optimisation / `ppa` | Completed | Completed | Completed |
| 9 | `gemm` | optimisation / `ppa` | Completed | Completed | Completed |
| 10 | `syntax_missing_semicolon` | repair / `syntax` | Completed | Completed | Completed |
| 11 | `indexing_off_by_one` | repair / `indexing` | Completed | Completed | Completed |
| 12 | `multi_fault_feedback` | repair / `multi_fault` | Completed | Completed | Completed |
| 13 | `interface_wrong_top_name` | repair / `interface` | Not completed | Not completed | Not completed |
| 14 | `functional_subtraction` | repair / `functional_arithmetic` | Completed | Completed | Completed |
| 15 | `accumulator_overwrite` | repair / `state_accumulator` | Completed | Completed | Completed |
| 16 | `loop_bound_missing_last` | repair / `loop_bound` | Completed | Completed | Completed |
| 17 | `functional_wrong_sign` | repair / `matrix_arithmetic` | Completed | Completed | Completed |
| 18 | `staged_compile_then_functional` | repair / `staged_compile_functional` | Completed | Completed | Completed |
| 19 | `vector_add` | optimisation / `ppa` | Completed | Completed | Completed |
| 20 | `stream_pipeline` | optimisation / `ppa` | Completed | Completed | Completed |

## Aggregate counts

| Group | Rows | Completed | Completion rate |
|---|---:|---:|---:|
| Generation / repair / structural | 42 | 39 | 92.9% |
| PPA run completion | 18 | 15 | 83.3% |
| **Overall** | **60** | **54** | **90.0%** |

The six non-completions are exactly:

- 3 × `atax` PPA rows: invalid starting optimisation baseline, so no valid before–after PPA comparison could be established.
- 3 × `interface_wrong_top_name` rows: wrong-top interface cases that did not reach a valid terminal state.

All other 54 model–task rows reached valid terminal states.

## Model identities

- **DeepSeek-V4-Pro:** `deepseek-ai/DeepSeek-V4-Pro`
- **Qwen3.5-122B-A10B:** `Qwen/Qwen3.5-122B-A10B`
- **Qwen3.6-27B:** `Qwen/Qwen3.6-27B`

## Machine-readable form

The corresponding 60-row machine-readable table is [`overnight_60_matrix.csv`](overnight_60_matrix.csv).
It preserves task tier, mode, role, subtype, canonical design, exact model identifier, terminal status and the status interpretation used in the dissertation.

## Reproduction

Inspect the canonical plan:

```bash
python3 -u scripts/run_experiment_matrix.py \
  --suite configs/suites/overnight_60.json \
  --list
```

A full reproduction additionally requires the public FPT Track-A harness for the external organiser tasks referenced by the suite.
