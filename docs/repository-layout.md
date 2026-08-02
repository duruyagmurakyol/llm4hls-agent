# Repository layout and ownership

This repository separates reusable implementation, experiment definitions, executable entry points, and generated evidence. The separation matters because the same repair logic is reused across benchmarks, models, and ablation studies.

## Data flow

```text
benchmarks/ + configs/
        |
        v
scripts/experiments/ ----> agent/
        |
        v
results/experiments/
        |
        +----> scripts/ablations/ ----> results/ablations/
        |
        +----> scripts/analysis/  ----> results/comparisons/ or diagnosis JSON
```

## Directory responsibilities

### `agent/`

Reusable Python implementation:

- repair and validation logic;
- model-provider clients;
- HLS evidence analysis;
- code that should be unit-testable without invoking a complete study.

A second script should not copy substantial logic from a first script. Move shared behaviour into `agent/` instead.

### `benchmarks/`

Versioned experimental inputs. Each benchmark should keep its golden design, fault variants, source, testbench, and `task.cfg` close together. Benchmark-local generation helpers may stay beside the benchmark when they are meaningful only there.

### `configs/`

JSON experiment definitions and model manifests. A configuration should identify the benchmark source, repair mode, model, editable/protected files, validation commands, and model limits. New experiments should normally be expressible by adding a config rather than editing a runner.

### `scripts/`

Command-line orchestration grouped by responsibility. See `scripts/README.md`. Scripts may call into `agent/`, read configs, and write results, but they should not become a second implementation package.

### `results/`

Generated evidence organised by study type:

- `results/experiments/`: individual runs;
- `results/suites/`: config-directory aggregations;
- `results/ablations/`: controlled repeated studies;
- `results/comparisons/`: post-hoc comparisons.

Timestamped result directories are immutable research records. Re-run an experiment instead of editing an existing result.

### `docs/`, `notes/`, and `prompts/`

- `docs/`: stable instructions and architecture documentation;
- `notes/`: evolving observations, meeting notes, and research reasoning;
- `prompts/`: reusable prompt assets that are not embedded in a specific config.

## Naming conventions

- Use `run_` for commands that execute experiments.
- Use `setup_` for deterministic input/config generation.
- Use `compare_` or `diagnose_` for post-processing existing evidence.
- Use descriptive config directories such as `<benchmark>_<mode>_<model-slug>`.
- Keep model slugs filesystem-safe and store the exact provider model identifier in the config or manifest.

## Boundary rule

A useful test is: **could another benchmark or runner reuse this function?** If yes, it belongs under `agent/`; if no and it is an executable workflow, it belongs under the appropriate `scripts/` folder.
