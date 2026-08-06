# Explicit 60-run overnight matrix

`configs/suites/overnight_60.json` defines a fixed experiment rather than
recursively discovering benchmark directories.

## Models

Every task is run with the three model identifiers recommended in the Track-A
guidance, in this order:

1. `deepseek-ai/DeepSeek-V4-Pro`
2. `cyankiwi/Qwen3.5-122B-A10B-AWQ-4bit`
3. `Qwen/Qwen3.6-27B-FP8`

The matrix is task-first: all three models complete one task before the runner
moves to the next task. This preserves a fair comparison if an overnight run is
interrupted part-way through.

## Tasks

The first twelve tasks form the 36-run core. The final eight tasks add 24 runs
of extended fault and optimisation coverage.

| Priority | Tier | Mode | Role | Task | Canonical design |
|---:|---|---|---|---|---|
| 1 | core | repair | generation | `vector_add_generate` | vector add |
| 2 | core | repair | synthesis repair | `synth_fix_dynamic_buffer` | vector scale |
| 3 | core | repair | functional repair | `projection_bugfix` | projection |
| 4 | core | repair | structural repair | `residual_stream_deadlock` | residual stream |
| 5 | core | repair | blind structural repair | `structural_blind_stream` | stream pipeline |
| 6 | core | optimise | PPA optimisation | `dotProduct_optimize` | dot product |
| 7 | core | optimise | PPA optimisation | `atax` | ATAX |
| 8 | core | optimise | PPA optimisation | `bicg` | BICG |
| 9 | core | optimise | PPA optimisation | `gemm` | GEMM |
| 10 | core | repair | syntax repair | `syntax_missing_semicolon` | vector add |
| 11 | core | repair | indexing repair | `indexing_off_by_one` | vector add |
| 12 | core | repair | multi-fault repair | `multi_fault_feedback` | BICG |
| 13 | extended | repair | interface repair | `interface_wrong_top_name` | vector add |
| 14 | extended | repair | functional repair | `functional_subtraction` | vector add |
| 15 | extended | repair | state repair | `accumulator_overwrite` | dot product |
| 16 | extended | repair | loop-bound repair | `loop_bound_missing_last` | dot product |
| 17 | extended | repair | matrix arithmetic repair | `functional_wrong_sign` | GEMM |
| 18 | extended | repair | staged repair | `staged_compile_then_functional` | BICG |
| 19 | extended | optimise | PPA optimisation | `vector_add` | vector add |
| 20 | extended | optimise | PPA optimisation | `stream_pipeline` | stream pipeline |

Only the six unique correct baselines enter optimisation. Fault variants stop
as soon as a repaired design passes every validation stage required by the task.

## Inspect the plan

```bash
python3 scripts/run_experiment_matrix.py --list
```

The command must print `Planned 60 run(s)`.

## Start

Source Vitis 2025.2 and export `SILICONFLOW_API_KEY`, then run:

```bash
systemd-inhibit \
  --what=sleep \
  --why="LLM4HLS 60-run overnight matrix" \
  python3 -u scripts/run_experiment_matrix.py \
    --run-id overnight_60_20260806
```

The runner refuses to start while another agent or Vitis process is active.
Runs are sequential and continue after individual model/task failures.

## Outputs

Suite-level evidence is written under:

```text
results/suites/<run-id>/
├── environment.json
├── suite_definition.json
├── matrix_manifest.json
├── suite_state.json
├── suite_summary.csv
├── manifests/
└── logs/
```

Model/task outputs are isolated under:

```text
experiments/model_comparison/<run-id>/<model-slug>/<task-id>/
```

The CSV includes task taxonomy, exact model/provider, execution mode, initial
failure stage, attempts, verification, timing/resources, tokens, tool calls,
credits and reference-harness score estimate.

## Resume

```bash
python3 -u scripts/run_experiment_matrix.py \
  --resume-suite results/suites/<run-id>
```

Completed run keys are skipped. To retry only unsuccessful rows:

```bash
python3 -u scripts/run_experiment_matrix.py \
  --resume-suite results/suites/<run-id> \
  --rerun-failed
```

## Core-only fallback

To run only the highest-priority 36 comparisons:

```bash
python3 -u scripts/run_experiment_matrix.py \
  --core-only \
  --run-id overnight_core_36_20260806
```
