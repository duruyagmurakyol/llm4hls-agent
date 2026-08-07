# `scripts/`

`run_agent.py` is the only normal user-facing agent command. Reusable agent behaviour lives under `agent/`; this directory now contains only execution, experiment and evidence utilities that are still used by the final workflow.

## Public command

```bash
python3 scripts/run_agent.py <task-package-or-manifest> [options]
```

The Docker wrapper invokes the same entry point:

```bash
docker/run-vitis.sh <task-directory> --mode auto
```

## Experiment runners

- `run_experiment_matrix.py` — explicit task × model matrix with isolated manifests, resumable state and CSV/JSON summaries.
- `run_task_suite.py` — sequential discovery and execution of a provenance-aware task library.

Both call `run_agent.py`; neither implements a second agent.

## Analysis and evidence utilities

- `check_matrix_models.py` — fail-fast provider/model availability check before expensive runs.
- `audit_cosim_suite.py` — durable post-search final/Pareto C/RTL co-simulation audit.
- `extract_final_results.py` — report-ready JSON/CSV/Markdown extraction.
- `analyse_hls_hierarchy.py` — standalone Vitis hierarchy/bottleneck analysis.
- `audit_prompt_compaction.py` — offline token/prompt compaction audit.

## Maintenance

`maintenance/` contains the one retained historical helper required by regression coverage. It is not a public agent entry point.

One-time repository-construction and migration scripts were removed once their generated benchmark/configuration artefacts were committed. They are not required to run or evaluate the competition agent.

## Rules

- `run_agent.py` is the only public agent launcher.
- Put reusable state machines under `agent/`, not here.
- Do not add benchmark-specific launchers.
- Do not embed API keys.
- Generated outputs belong under `experiments/`, `results/` or `runs/`.

## Validation

```bash
python -m compileall -q agent scripts tests
python -m pytest -q
```
