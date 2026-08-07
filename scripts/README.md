# `scripts/`

`scripts/run_agent.py` is the only normal user-facing agent command. Reusable behaviour belongs under `agent/`; the other scripts are research, audit or repository-maintenance utilities rather than alternative agents.

## Public command

```bash
python3 scripts/run_agent.py <task-package-or-manifest> [options]
```

Important options include `--mode auto|repair|optimise`, `--onboard-only`, `--resume`, `--status-only` and `--max-agent-steps`.

The Docker wrapper calls this same entry point:

```bash
docker/run-vitis.sh <task-directory> --mode auto
```

## Research runners

- `run_experiment_matrix.py` — explicit task × model matrix with isolated manifests, resumable state and CSV/JSON summaries.
- `run_task_suite.py` — sequential discovery/execution of a provenance-aware task library.

These scripts invoke `run_agent.py`; they do not implement a second agent.

## Analysis and audit utilities

- `check_matrix_models.py` — fail-fast provider/model availability check.
- `audit_cosim_suite.py` — durable post-search final/Pareto COSIM re-audit for existing suite results.
- `extract_final_results.py` — report-ready JSON/CSV/Markdown extraction.
- `analyse_hls_hierarchy.py` — standalone CLI for the HLS hierarchy analyser.
- `audit_prompt_compaction.py` — offline prompt/token compaction analysis.

## Reproducibility / maintenance utilities

The remaining preparation and migration scripts exist to regenerate committed benchmark/task material or recover experiment manifests. They are not part of the competition runtime path. Examples include U55C validation subset preparation, controlled repair-suite construction, HLS-Eval import, provider-ID migration and matrix refresh.

## Removed obsolete wrappers

The old standalone PPA optimiser, specialised overnight repair/agent runners and duplicate container entrypoint were removed. Their responsibilities are now covered by the unified agent, generic suite/matrix runners and `docker/entrypoint.sh`.

## Rules

- Put reusable state machines under `agent/`, not here.
- Do not add benchmark-specific public launchers.
- Do not embed API keys.
- Keep generated outputs under `experiments/`, `results/` or `runs/` as appropriate.
- Return non-zero on configuration or execution failure.

## Validation

```bash
python -m compileall -q scripts
python -m pytest -q
```
