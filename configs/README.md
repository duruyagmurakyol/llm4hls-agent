# `configs/`

Persistent, human-reviewed task and experiment definitions. Generated run state belongs under ignored `experiments/` or `results/`, not here.

## Layout

```text
configs/
├── tasks/    persistent task manifests and target-specific variants
└── suites/   model × task experiment definitions
```

Task manifests define the source/testbench contract, target device, budgets, model settings and output location. Suite files define reproducible collections of tasks and models for controlled comparisons.

## Submission-relevant examples

- `suites/qwen_bicg_u55c_exact_regression.json` — Qwen regression over the committed `benchmarks/bicg/u55c` package.
- `suites/optimisation_15.json` — multi-model optimisation comparison over retained benchmark families plus an external official Track-A task.
- `tasks/u55c_validation/` — persistent U55C-targeted validation manifests.
- `tasks/repair_suite/` and `tasks/combined_full_agent/` — retained repair/repair-to-optimisation definitions.

A suite may deliberately refer to `external/fpt26-harness/...` when reproducing organiser-supplied Track-A tasks. That harness is an external evaluation input and is not bundled into the submission image.

## Provenance rule

A retained configuration should point either to:

1. a committed benchmark under `benchmarks/`; or
2. an explicitly external organiser/evaluator task package.

Generated run directories are not valid persistent configuration dependencies.

## Secrets

Never place API keys in JSON. Configure providers through environment variables, for example:

```bash
export SILICONFLOW_API_KEY="your-key"
export LLM4HLS_PROVIDER=siliconflow
```

## Running a suite

```bash
python3 -u scripts/run_experiment_matrix.py \
  --suite configs/suites/qwen_bicg_u55c_exact_regression.json \
  --run-id qwen_bicg_u55c_regression
```

Generated manifests, logs and result summaries are written outside `configs/`.
