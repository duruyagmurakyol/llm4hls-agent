# Unified Agent Refactor

## Objective

Turn the repository from a collection of experiment entry points into one budgeted repair-and-optimisation agent without discarding proven repair or Vitis flows.

## Stage 1: Public integration layer — implemented

- `agent/config.py`: validates one task-manifest format.
- `agent/state.py`: defines shared budget, trajectory and result records.
- `agent/controller.py`: routes unified tasks to proven adapters.
- `scripts/run_agent.py`: provides one public command.
- `configs/tasks/vector_add_repair.json`: controlled functional-repair task.
- `configs/tasks/vector_add_track_a.json`: vector-add PPA task.

At this stage, the controller delegates to:

- `scripts/run_api_experiment.py` for direct-API repair;
- `scripts/run_track_a_agent.py` for budgeted PPA iterations.

This is deliberate. The first integration stage changes orchestration, not working repair or synthesis behaviour.

## Stage 2: Local regression validation — required before merge

Run on the Xilinx machine:

```bash
python3 -m pytest tests/test_unified_config.py tests/test_unified_state.py
python3 scripts/run_agent.py configs/tasks/vector_add_repair.json
python3 scripts/run_agent.py configs/tasks/vector_add_track_a.json --status-only
```

Then run the PPA task only after checking the remaining synthesis budget:

```bash
python3 scripts/run_agent.py configs/tasks/vector_add_track_a.json
```

Acceptance checks:

1. The repair task changes only `src/vector_add.cpp` and passes both validations.
2. The PPA task writes a Track A ledger and a unified result.
3. Existing candidate sources and experiment summaries remain unchanged unless a new iteration is intentionally requested.
4. No benchmark source is overwritten by the unified controller.

## Stage 3: True repair-to-PPA promotion — next implementation

The current repair and PPA examples use different vector-add benchmark layouts and data sizes. They must not be silently chained.

A genuine end-to-end task requires:

1. one benchmark source used by both repair and PPA;
2. a controller-owned run workspace;
3. repair output promoted into that workspace;
4. dynamically generated Vitis Tcl/config files that reference the workspace source;
5. baseline synthesis performed on the repaired source;
6. PPA candidates derived from that exact baseline.

The existing vector-add baseline Tcl contains machine-specific absolute paths. This should be replaced with a generated Tcl file before claiming portable end-to-end execution.

## Stage 4: Internal module extraction — after regression success

Move reusable logic into:

```text
agent/tools/validation.py
agent/tools/synthesis.py
agent/tools/reports.py
agent/repair/diagnose.py
agent/repair/generate.py
agent/optimise/diagnose.py
agent/optimise/generate.py
agent/optimise/evaluate.py
```

Old scripts should temporarily become compatibility wrappers. They should be archived only when the unified controller reproduces the established repair and PPA results.

## Stage 5: Repository cleanup — after module extraction

- move historical one-off runners to `archive/legacy_scripts/`;
- retain ablation runners under a clearly labelled research folder;
- remove generated Vitis binaries and workspaces from Git tracking;
- keep only curated source, prompt and compact JSON evidence;
- make `scripts/run_agent.py` the only Quick Start entry point.

## Merge rule

Merge into `main` only after Stage 2 passes locally. Continue future feature work directly on `main` after this integration branch is merged.
