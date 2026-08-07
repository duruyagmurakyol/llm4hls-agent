# `agent/` package

The reusable implementation is grouped by responsibility. The repository keeps a stable compatibility import surface, so existing imports such as `from agent.config import TaskManifest` continue to work even though implementations now live in subpackages.

## Layout

```text
agent/
├── __init__.py          stable import compatibility surface
├── core/                orchestration, manifests, budgets and shared state
│   ├── baseline.py
│   ├── budget.py
│   ├── config.py
│   ├── controller.py
│   ├── execution_mode.py
│   ├── failures.py
│   ├── resume.py
│   └── state.py
├── competition/         FPT Track-A integration and verification policy
│   ├── final_cosim.py
│   ├── stage_aware.py
│   ├── track_a.py
│   ├── track_a_scoring.py
│   └── track_a_selection.py
├── optimise/            structured synthesis-feedback PPA search
├── repair/              correctness repair and bounded retries
├── tools/               Vitis execution, validation and report parsing
├── analysis/            HLS hierarchy and bottleneck analysis
├── providers/           model-provider adapters
└── support/             onboarding, reporting and small shared utilities
    ├── onboarding.py
    ├── onboarding_safe.py
    ├── prompt_compaction.py
    ├── reporting.py
    ├── terminal_reporting.py
    └── workspace.py
```

`agent/optimise/archive.py` owns durable selected/Pareto candidate materialisation and is kept beside the optimisation state machine that consumes it.

## Public execution path

`scripts/run_agent.py` is the single normal command-line entry point.

```text
task package / manifest
        ↓
onboarding + manifest validation
        ↓
initial correctness validation
        ↓
repair when required
        ↓
verified baseline
        ↓
structured PPA optimisation
        ↓
ranked final C/RTL co-simulation audit
        ↓
fully verified selected design
```

For Track-A packages, `competition/track_a.py` stages only public task material. Hidden tests and reference implementations are never copied into the agent-visible workspace.

## Core responsibilities

- `core/controller.py` — repair-to-optimisation orchestration and unified result writing.
- `core/config.py` — task-manifest loading, validation and path checks.
- `core/budget.py` — model/tool budgets and weighted Track-A validation credits.
- `core/baseline.py` — promotion of exact CSim/synthesis/COSIM-verified source bytes and reports.
- `core/execution_mode.py` — `auto`, `repair` and `optimise` policies.
- `core/resume.py` — safe continuation from a durable verified baseline.

## Competition responsibilities

- `competition/track_a.py` — official-style task-package ingestion, U55C target and public staging.
- `competition/stage_aware.py` — bounded `generate`, `synth_fix` and `structural` task handling.
- `competition/final_cosim.py` — selected → ranked Pareto → verified baseline final COSIM policy.
- `competition/track_a_scoring.py` — public score estimate and baseline capture without extra Vitis calls.
- `competition/track_a_selection.py` — optional organiser-facing deterministic ranking.

## Optimisation

`optimise/` owns candidate diagnosis, strategy selection, prompt construction, generation, static safety checks, CSim, synthesis, Pareto evaluation, parent selection, structured recovery and final candidate archiving.

Search-time validation is deliberately cheaper than final verification. Candidate search can use static checks, CSim and synthesis; final selected designs are audited with C/RTL co-simulation according to the task policy.

## Repair

`repair/` owns failure diagnosis, constrained source generation, response validation, retry feedback, repair artefacts and promotion of the exact validated source into the later synthesis/COSIM stages.

## Support and tools

`support/` contains non-orchestration helpers such as benchmark discovery, safe onboarding, prompt compaction and report rendering. `tools/` contains the external-tool boundary for Vitis CSim, synthesis and C/RTL co-simulation.

## Design rules

1. Keep `scripts/run_agent.py` as the single public agent command.
2. Do not branch on benchmark names in core orchestration.
3. Do not modify supplied testbenches or reference material.
4. Keep expensive work behind cheap static/correctness checks.
5. Record tool/model usage and important decisions as machine-readable evidence.
6. Promote only source bytes that match their validation hashes.
7. Keep hidden/reference Track-A files outside the agent-visible workspace.
8. Add regression tests for generic behaviour changes.

## Validation

```bash
python -m pytest -q
python -m compileall -q agent scripts tests
```
