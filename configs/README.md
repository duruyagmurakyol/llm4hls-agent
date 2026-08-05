# `configs/`

This directory contains persistent, human-authored configuration. Generated onboarding configuration belongs under `experiments/onboarding/`, not here.

## Contents

```text
configs/
├── tasks/                       unified task manifests
├── track_a_task.schema.json     task-manifest schema
└── *_ppa.json                   explicit optimisation configurations
```

## Unified task manifests

A task manifest describes what the agent is allowed to use, the public interface that must be preserved, the target device, budgets and the workflow adapter.

Typical structure:

```json
{
  "task_id": "example_001",
  "task_kind": "correct_unoptimised",
  "artifacts": {
    "source": "benchmarks/example/src/kernel.cpp",
    "testbench": ["benchmarks/example/testbench/kernel_test.cpp"],
    "headers": ["benchmarks/example/src/kernel.h"],
    "build_files": ["benchmarks/example/task.cfg"]
  },
  "interface": {
    "top_function": "kernel",
    "language": "cpp",
    "numerical_tolerance": null,
    "protected_contract": [
      "Preserve the top-level function signature.",
      "Preserve output semantics checked by the testbench."
    ]
  },
  "target": {
    "tool": "AMD Vitis HLS",
    "tool_version": "2025.2",
    "part": "xczu3eg-sfvc784-2-e",
    "clock_period_ns": 10.0,
    "resource_limits": {}
  },
  "budgets": {
    "max_iterations": 5,
    "max_csim_calls": 5,
    "max_cosim_calls": 0,
    "max_synthesis_calls": 4,
    "max_model_calls": 5,
    "max_total_tokens": null
  },
  "adapter": {
    "kind": "autonomous_ppa",
    "config": "configs/example_ppa.json"
  },
  "output_dir": "experiments/track_a/example_001"
}
```

## Important manifest fields

### `task_id`

Stable identifier for the run. It should be unique and descriptive.

### `task_kind`

Describes the starting condition, for example a correct but unoptimised design or a faulty repair task.

### `artifacts`

Declares all source-of-truth files. Paths are repository-relative unless explicitly absolute.

### `interface`

Defines the contract candidates must preserve. The top function must match the build description and testbench.

### `target`

Defines the synthesis environment. Comparisons are meaningful only when baseline and candidate use the same part, clock and Vitis version.

### `budgets`

Limits expensive operations. Model and synthesis calls are separate budgets because their cost and failure modes differ.

### `adapter`

Selects the workflow implementation:

- `autonomous_ppa`: synthesis-feedback-guided optimisation;
- `direct_api_repair`: correctness repair;
- `legacy_ppa`: compatibility alias, not recommended for new manifests.

### `output_dir`

Location for the unified result and task-level artefacts. It must not point into `benchmarks/`.

## Optimisation configuration

An optimisation config provides PPA-specific settings used by `agent.optimise.runner`.

Typical structure:

```json
{
  "experiment_name": "example_ppa",
  "benchmark": "example",
  "top_function": "kernel",
  "baseline": {
    "source": "benchmarks/example/src/kernel.cpp",
    "tcl": "benchmarks/example/task.cfg",
    "project_dir": "experiments/example/baseline_project"
  },
  "validation": {
    "constant_loop_tail_bounds": true,
    "preserve_diagnosed_loop_label": true
  },
  "prompt_constraints": [
    "Preserve the top-level function signature.",
    "Do not modify the testbench or baseline source in place."
  ],
  "output_dir": "experiments/example/autonomous_ppa",
  "model": {
    "provider": "siliconflow",
    "name": "Qwen/Qwen3.5-122B-A10B",
    "temperature": 0.0,
    "max_tokens": 4096,
    "enable_thinking": false
  },
  "budget": {
    "max_candidates": 5,
    "max_synthesis_calls": 4
  }
}
```

The key remains named `baseline.tcl` for compatibility, but it may refer to either a TCL script or a supported `task.cfg` build description.

## Hand-authored versus generated configs

Use `configs/` when:

- a task should be reproducible and retained;
- budgets or model settings need deliberate control;
- the benchmark layout cannot be safely inferred;
- the task is part of a formal experiment or submission.

Use automatic onboarding when:

- quickly checking a self-contained benchmark;
- validating generality across layouts;
- creating an initial manifest that can later be promoted into `configs/`.

Generated onboarding files should be reviewed before being copied into persistent configuration.

## Validation

Inspect status without creating a new candidate:

```bash
python3 scripts/run_agent.py configs/tasks/atax_track_a.json --status-only
```

Run all structural tests:

```bash
python -m pytest
```

## Configuration rules

- Do not embed shell command strings in task manifests.
- Do not place API keys in JSON.
- Do not use benchmark-name switches to alter core behaviour.
- Keep baseline and candidate target settings identical.
- Keep output directories outside source benchmark folders.
- Use small budgets while developing a new task.
- Treat configuration changes as part of experiment provenance.
