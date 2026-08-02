# Running experiments

Run commands from the repository root. The scripts resolve repository paths independently, but a consistent working directory makes config and output paths easier to read.

## Prerequisites

- Python 3.10 or newer;
- `g++` for the host-validation commands used by current configs;
- AMD/Xilinx Vitis HLS for independent HLS validation where enabled;
- Codex CLI for `autonomous` and `structured_feedback` modes;
- `SILICONFLOW_API_KEY` for direct API modes.

Never commit API keys. Export them in the shell running the experiment:

```bash
export SILICONFLOW_API_KEY="..."
```

## Choose the runner from `repair_mode`

| `repair_mode` | Runner |
| --- | --- |
| `autonomous` | `scripts/experiments/run_experiment.py` |
| `structured_feedback` | `scripts/experiments/run_structured_experiment.py` |
| `direct_api` | `scripts/experiments/run_api_experiment.py` |
| `iterative_direct_api` | `scripts/experiments/run_iterative_api_experiment.py` |

`run_suite.py` dispatches the first three modes automatically from each config. Iterative studies currently use their dedicated runner or the ablation commands.

## Single run

```bash
python3 scripts/experiments/run_api_experiment.py \
  configs/vector_add_api_qwen35/functional.json
```

Keep the mutable workspace for debugging:

```bash
python3 scripts/experiments/run_api_experiment.py \
  configs/vector_add_api_qwen35/functional.json \
  --keep-workspace
```

## Iterative repair

```bash
python3 scripts/experiments/run_iterative_api_experiment.py \
  configs/bicg_iterative_qwen35/staged_compile_then_functional.json \
  --max-iterations 3
```

Each iteration records the prompt, raw response, candidate, validation output, diff, token usage, and failure transition.

## Suite

```bash
python3 scripts/experiments/run_suite.py configs/vector_add_api_qwen35 \
  --continue-on-failure
```

A suite writes `summary.json`, `summary.csv`, and one log per configured experiment beneath `results/suites/<suite>/<timestamp>/`.

## Repeated ablations

```bash
python3 scripts/ablations/run_bicg_staged_repeated_ablation.py --repetitions 5
python3 scripts/ablations/run_atax_staged_repeated_ablation.py --repetitions 5
```

For another configured model:

```bash
python3 scripts/setup/setup_staged_model_configs.py \
  --model "<provider/model-id>" \
  --slug "<short-slug>"

python3 scripts/ablations/run_staged_model_repeated.py \
  --model-slug "<short-slug>" \
  --repetitions 5
```

Repeated runners preserve partial CSV/JSON evidence if a later run fails.

## Analysis

Compare latest benchmark-level repeated results:

```bash
python3 scripts/analysis/compare_atax_bicg_staged_results.py
```

Compare configured staged models:

```bash
python3 scripts/analysis/compare_staged_models.py
```

Diagnose generic HLS evidence:

```bash
python3 scripts/analysis/diagnose_hls_evidence.py evidence.json \
  --output results/diagnoses/example.json
```

## Result contract

A normal experiment creates a timestamped directory under `results/experiments/<experiment-id>/`. Depending on the runner, it includes:

- a copied config and prompts;
- validation logs before and after repair;
- the original and repaired source or a unified diff;
- raw provider/agent output;
- `result.json` containing the machine-readable outcome and cost metrics.

Treat that directory as immutable. Derived statistics belong in a new suite, ablation, comparison, or diagnosis directory.
