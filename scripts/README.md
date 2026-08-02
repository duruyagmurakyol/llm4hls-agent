# `scripts/`

This directory contains command-line entry points and a small number of compatibility or analysis utilities.

## Public command

Only this script should be treated as the normal user interface:

```bash
python3 scripts/run_agent.py <task-manifest-or-benchmark-directory>
```

`run_agent.py` accepts either:

- a unified JSON task manifest; or
- a benchmark directory that can be automatically onboarded.

Options:

```text
--onboard-only       generate task/config/report files without running the agent
--status-only        inspect resumable optimisation state without new expensive work
--max-agent-steps N  execute at most N optimisation state transitions
```

Examples:

```bash
python3 scripts/run_agent.py configs/tasks/atax_track_a.json
python3 scripts/run_agent.py benchmarks/bicg/golden --onboard-only
python3 scripts/run_agent.py benchmarks/bicg/golden --max-agent-steps 1
```

## Internal and compatibility scripts

Other scripts are implementation support, analysis utilities, or transitional compatibility layers. They are not separate agents and should not be documented as alternative primary workflows.

A compatibility script may remain temporarily when code in `agent/` still invokes it, but new orchestration should be implemented as importable package code under `agent/`.

## Script design rules

- Keep argument parsing and process exit handling in scripts.
- Put reusable behaviour in `agent/` modules.
- Do not duplicate repair or optimisation state machines here.
- Do not add benchmark-specific public scripts.
- Do not embed API keys.
- Resolve repository paths predictably.
- Return non-zero on configuration or execution failure.
- Keep generated files under `experiments/`.

## Adding a command

Before adding a new script, check whether the feature belongs as:

1. an option to `run_agent.py`;
2. an importable function in `agent/`;
3. a test or offline analysis notebook rather than a permanent command.

A new script is justified only when it has a distinct operational boundary, such as a one-off data conversion or a standalone report inspection utility.

## Obsolete scripts

The project intentionally removed earlier experiment-specific wrappers for generation, validation, CSim, synthesis and evaluation. Those stages now belong to the unified optimisation implementation.

Do not recreate scripts such as separate per-model experiment runners or benchmark-specific candidate commands unless required for reproducing archived evidence.

## Validation

```bash
python -m compileall -q scripts
python -m pytest
```
