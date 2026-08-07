# `agent/`

The reusable agent implementation is grouped by responsibility. `scripts/run_agent.py` is the only normal public CLI.

```text
agent/
├── core/          orchestration, task state, budgets, baseline and resume
├── competition/   Track-A staging, stage-aware tasks, scoring and final COSIM
├── optimise/      synthesis-feedback PPA search and candidate selection
├── repair/        correctness diagnosis, generation and bounded retries
├── tools/         Vitis execution, validation and report parsing
├── analysis/      HLS hierarchy and bottleneck analysis
├── providers/     model-provider transport
└── support/       onboarding, prompt compaction and reporting helpers
```

Compatibility imports such as `agent.config`, `agent.controller` and `agent.track_a` remain available even though implementations are physically organised in the folders above.

## Execution path

```text
task
 ↓
onboarding + manifest validation
 ↓
initial CSim / correctness checks
 ↓
repair if required
 ↓
verified baseline
 ↓
structured optimisation
 ↓
final selected → Pareto → baseline COSIM audit
 ↓
fully verified design
```

`competition/track_a.py` treats official task packages as external read-only inputs and stages only public files. Hidden tests and reference solutions never enter the agent-visible workspace.

## Main responsibilities

- `core/` owns the unified state machine, budgets, task manifests and verified-baseline promotion.
- `repair/` fixes compile/interface/functional failures without changing protected test material.
- `optimise/` diagnoses synthesis evidence, chooses structured transformations, evaluates candidates and maintains the Pareto archive.
- `competition/final_cosim.py` performs mandatory post-search RTL verification and fallback selection.
- `tools/` is the boundary to CSim, synthesis and C/RTL co-simulation.

## Design rules

- correctness before optimisation;
- no benchmark-name branches in core orchestration;
- supplied testbenches/build files are protected;
- cheap checks precede expensive Vitis calls;
- model/tool usage is budgeted and recorded;
- only validated source bytes can be promoted;
- final selection is deterministic and backed by durable evidence.

## Validation

```bash
python -m compileall -q agent
python -m pytest -q
```
