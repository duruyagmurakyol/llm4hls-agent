# LLM4HLS Agent

An autonomous, task-driven agent for repairing AMD/Xilinx HLS designs and improving performance, power and area through synthesis feedback.

The repository is organised around one public command:

```bash
python3 scripts/run_agent.py <task-manifest-or-benchmark-directory>
```

The controller is benchmark-independent. Vector addition, ATAX, BICG and future designs are inputs to the same agent rather than separate implementations.

## What the agent does

The agent supports two related workflows:

1. **Correctness repair**: diagnose compilation, interface or functional failures; generate a repair; and revalidate the design.
2. **PPA optimisation**: establish a correct baseline, analyse Vitis HLS reports, generate a constrained candidate, run static checks and CSim, synthesise the candidate, and compare it with the baseline.

Correctness is always established before optimisation.

## Quick start

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip pytest
```

### 2. Load AMD/Xilinx tools

The exact path depends on the installation:

```bash
source /path/to/Xilinx/Vitis/2025.2/settings64.sh
```

Confirm that Vitis is available:

```bash
vitis-run --version
```

### 3. Configure the model provider

```bash
export SILICONFLOW_API_KEY="your-key"
```

### 4. Run the tests

```bash
python -m pytest
python -m compileall -q agent scripts tests
```

### 5. Run a task

Using an existing manifest:

```bash
python3 scripts/run_agent.py configs/tasks/atax_track_a.json
```

Using a benchmark directory:

```bash
python3 scripts/run_agent.py benchmarks/vector_add
python3 scripts/run_agent.py benchmarks/hls_eval/atax
python3 scripts/run_agent.py benchmarks/bicg/golden
```

Directory onboarding accepts a supported HLS TCL flow or a repository `task.cfg`. It discovers the top function, source, testbench, part and clock, then creates generated task and optimisation configuration under `experiments/onboarding/<benchmark>/`.

## Useful command options

Generate configuration without running the agent:

```bash
python3 scripts/run_agent.py benchmarks/bicg/golden --onboard-only
```

Inspect the current optimisation state without generating or synthesising a new candidate:

```bash
python3 scripts/run_agent.py configs/tasks/atax_track_a.json --status-only
```

Limit the number of optimisation transitions performed in one invocation:

```bash
python3 scripts/run_agent.py benchmarks/bicg/golden --max-agent-steps 1
```

The optimisation workflow is resumable. Existing reports and completed candidate stages are reused where appropriate.

## End-to-end workflow

```text
input task or benchmark directory
          |
          v
manifest loading / automatic onboarding
          |
          v
isolated output and workspace preparation
          |
          v
correctness validation
    | failure
    +------> diagnosis -> model repair -> validation
    |
    v
baseline CSim and synthesis
          |
          v
hierarchical report diagnosis
          |
          v
source-level target and cause mapping
          |
          v
constrained model prompt and candidate generation
          |
          v
static validation -> duplicate check -> CSim -> synthesis
          |
          v
baseline comparison, verdict and budget decision
```

A candidate cannot reach synthesis unless it passes the cheaper safety and correctness stages first.

## Repository structure

```text
agent/                  reusable agent implementation
  analysis/             detailed Vitis hierarchy and source-cause analysis
  optimise/             autonomous PPA workflow
  providers/            model-provider adapters
  repair/               correctness-repair workflow
  tools/                validation, command, report and synthesis utilities
benchmarks/             source designs, headers, testbenches and build descriptions
configs/                task manifests, schemas and explicit optimisation configs
experiments/            generated runs, prompts, reports, candidates and results
results/                selected evidence intended for retention
scripts/                public CLI and limited compatibility/analysis utilities
tests/                  lightweight structural and behavioural regression tests
```

See the README inside each major folder for ownership rules and file-level explanations.

## Core modules

- `scripts/run_agent.py`: public command-line entry point.
- `agent/controller.py`: loads a task and dispatches to repair or optimisation.
- `agent/onboarding.py` / `agent/onboarding_safe.py`: discovers benchmark metadata and generates manifests.
- `agent/config.py`: validates unified task manifests.
- `agent/optimise/runner.py`: budgeted, resumable PPA state machine.
- `agent/optimise/diagnose.py`: prepares source-aware optimisation prompts.
- `agent/optimise/generate.py`: calls the configured model and extracts candidate C/C++.
- `agent/optimise/evaluate.py`: classifies candidates and compares synthesis metrics.
- `agent/tools/validation.py`: static HLS safety checks and failure classification.
- `agent/tools/synthesis.py`: portable CSim, synthesis and report extraction.
- `agent/repair/runner.py`: direct correctness-repair workflow.

## Benchmark build descriptions

### TCL-based benchmark

A synthesis TCL normally identifies:

```tcl
set_top kernel_name
add_files src/kernel.cpp
add_files -tb testbench/kernel_test.cpp
open_solution -reset solution1
set_part xczu3eg-sfvc784-2-e
create_clock -period 10 -name default
csim_design
csynth_design
```

### `task.cfg`-based benchmark

```ini
[hls]
flow_target=vivado
syn.file=src/kernel.cpp
syn.cflags=-Isrc
syn.top=kernel_name
tb.file=testbench/kernel_test.cpp
tb.cflags=-Isrc
part=xczu3eg-sfvc784-2-e
clock=10ns
```

The agent converts either description into controlled internal TCL scripts. Candidate runs use isolated temporary projects and do not modify the baseline source.

## Generated outputs

Automatically onboarded runs are stored under:

```text
experiments/onboarding/<benchmark>/
├── task.json
├── optimisation.json
├── onboarding_report.json
├── autonomous_ppa/
│   ├── baseline_*.json
│   ├── candidate_001.cpp
│   ├── candidate_001_prompt.txt
│   ├── candidate_001_*_validation.json
│   └── candidate_001_synthesis/
└── agent_result/
    └── unified_agent_result.json
```

These are generated artefacts, not source-of-truth configuration. Delete an incomplete generated run only when intentionally restarting that benchmark.

## Candidate decision policy

Typical terminal candidate verdicts include:

- `accept`: candidate satisfies correctness and improves the configured objective.
- `reject_static`: unsafe or contradictory HLS structure detected before Vitis.
- `reject_duplicate`: candidate is equivalent to an existing source candidate.
- `reject_csim`: candidate does not preserve testbench-observed behaviour.
- `reject_synthesis`: Vitis could not synthesise the candidate.
- `reject_synthesis_timeout`: synthesis exceeded the configured budget.
- `reject_no_performance_gain`: candidate synthesised but did not improve latency or interval.

Local loop II improvement alone is not sufficient. The final decision is based on extracted top-level synthesis metrics and configured trade-offs.

## Design principles

- One public agent interface.
- No benchmark-name conditionals in core orchestration.
- Baselines remain immutable.
- Candidate projects are isolated.
- Static and functional checks precede expensive synthesis.
- Every run emits machine-readable evidence.
- Budgets limit model and synthesis calls.
- Runs can be resumed from existing artefacts.

## Requirements

- Python 3.10 or later
- AMD Vitis HLS 2025.2
- `SILICONFLOW_API_KEY` for SiliconFlow model calls
- `pytest` for the test suite

## Current demonstrated coverage

The unified workflow has been exercised on:

- vector addition with an unlabeled source loop;
- ATAX with a TCL-described HLS flow;
- BICG with a Vitis `task.cfg` build description.

These examples validate the same generic path across different source structures and build-description formats; they are not hard-coded agent modes.
