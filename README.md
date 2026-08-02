# LLM4HLS Agent

A task-driven autonomous agent for repairing AMD/Xilinx HLS designs and performing synthesis-feedback-guided PPA optimisation.

## Public command

```bash
python3 scripts/run_agent.py <task-manifest>
```

The controller is benchmark-independent. Vector add, ATAX and future benchmarks are task packages supplied through manifests; they are not separate agent implementations.

## Core structure

```text
agent/
├── controller.py            state-machine entry point
├── config.py                task-manifest loading and validation
├── state.py                 shared result, metric and budget records
├── workspace.py             isolated task workspace creation
│
├── tools/
│   ├── command_runner.py    external command execution
│   ├── validation.py        generic failure classification
│   ├── synthesis.py         synthesis adapter boundary
│   └── reports.py           shared JSON input/output
│
├── repair/
│   ├── diagnose.py          correctness diagnosis
│   └── generate.py          repair generation through the provider
│
├── optimise/
│   ├── diagnose.py          Vitis report and bottleneck analysis
│   ├── generate.py          PPA candidate-generation adapter
│   ├── evaluate.py          generic Pareto comparison
│   └── strategies.json      small benchmark-independent strategy library
│
├── analysis/                existing detailed Vitis analysis implementation
└── providers/
    └── siliconflow.py       existing model-provider implementation
```

Only `scripts/run_agent.py` is a public entry point. Existing scripts remain temporarily as compatibility implementations behind the clean package boundaries. They should not be treated as alternative architectures.

## Workflow

```text
task manifest
    ↓
isolated workspace
    ↓
initial validation
    ├── failure → diagnose → repair → validate
    └── pass
          ↓
baseline synthesis
    ├── failure → synthesis diagnosis/repair
    └── pass
          ↓
report diagnosis → candidate generation
          ↓
static checks → CSim → synthesis
          ↓
Pareto and budget decision
```

Correctness is established before optimisation. The controller and shared modules do not contain benchmark-name checks, fixed array sizes or vector-add-specific transformations.

## Current compatibility adapters

The refactor preserves proven behaviour while code is moved gradually:

- direct API repair currently delegates to `scripts/run_api_experiment.py`;
- PPA task execution currently delegates to `scripts/run_track_a_agent.py`;
- candidate generation and synthesis use the existing PPA scripts behind `agent/optimise/` and `agent/tools/` boundaries;
- Vitis diagnosis reuses the existing `agent/analysis/` modules.

These wrappers can be archived after the clean modules reproduce the same results.

## Quick validation

```bash
python3 -m pytest \
  tests/test_unified_config.py \
  tests/test_unified_state.py \
  tests/test_agent_structure.py
```

Inspect a task without spending synthesis calls:

```bash
python3 scripts/run_agent.py configs/tasks/vector_add_track_a.json --status-only
```

Run the existing repair adapter through the unified command:

```bash
python3 scripts/run_agent.py configs/tasks/vector_add_repair.json
```

## Repository areas

```text
agent/                    core reusable agent code
scripts/run_agent.py      public CLI
configs/tasks/            task manifests and specifications
benchmarks/               task inputs and testbenches
experiments/              generated trajectories and summaries
results/                  selected experimental evidence
scripts/                  temporary compatibility implementations
```

Historical experiments and generated Vitis workspaces are not part of the core runtime and will be archived or ignored after regression validation.

## Refactor rule

Do not rewrite working repair or PPA logic merely to move it. First expose it through the clean package, add tests, and then replace each compatibility wrapper one at a time.

## Requirements

- Python 3.10 or later
- AMD Vitis HLS 2025.2 for CSim and synthesis
- `SILICONFLOW_API_KEY` for direct model calls
- `pytest` for lightweight tests
