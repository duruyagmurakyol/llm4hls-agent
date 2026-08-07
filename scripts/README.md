# `scripts/`

`scripts/run_agent.py` is the only normal user-facing agent command. The remaining scripts support controlled experiments, audits and result extraction; they do not implement competing agent flows.

## Public command

```bash
python3 scripts/run_agent.py <task-package-or-manifest> [options]
```

Common options include `--mode auto|repair|optimise`, `--onboard-only`, `--resume` and `--status-only`.

The Docker wrapper calls the same entry point:

```bash
docker/run-vitis.sh <task-directory> --mode auto
```

## Retained utilities

- `run_experiment_matrix.py` — explicit task × model experiments with isolated manifests and resumable state.
- `run_task_suite.py` — sequential execution of a benchmark/task collection.
- `check_matrix_models.py` — provider/model availability preflight.
- `audit_cosim_suite.py` — post-search final/Pareto COSIM re-audit.
- `extract_final_results.py` — report-ready JSON/CSV/Markdown extraction.
- `analyse_hls_hierarchy.py` — standalone synthesis-report diagnosis.
- `audit_prompt_compaction.py` — offline prompt/token compaction analysis.

`maintenance/` contains a small number of tested migration/reproducibility helpers that are not part of the normal competition runtime.

Obsolete benchmark-specific launchers, overnight wrappers, duplicate Docker entrypoints and one-off repository-construction scripts have been removed.

## Rules

- reusable logic belongs under `agent/`;
- do not add benchmark-specific public launchers;
- do not embed API keys;
- generated outputs belong under ignored `experiments/`, `results/` or `logs/`;
- scripts must return non-zero on execution/configuration failure.

## Validation

```bash
python -m compileall -q scripts
python -m pytest -q
```
