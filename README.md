# LLM4HLS Agent

An autonomous LLM-based agent for repairing AMD/Xilinx HLS C/C++ and improving hardware performance through synthesis feedback while preserving correctness.

The repository has one public agent command:

```bash
python3 scripts/run_agent.py <task-package-or-manifest>
```

For the reproducible Vitis container path:

```bash
docker/run-vitis.sh <task-directory> --mode auto
```

## Competition path

The FPT Track-A adapter accepts an official-style task package, copies only public kernel/header/testbench/description metadata into an isolated staging area, and deliberately excludes hidden tests and reference implementations.

The demonstrated competition target is:

```text
AMD Alveo U55C
part: xcu55c-fsvh2892-2L-e
Vitis HLS: 2025.2
submission minimum: 100 MHz (10 ns)
```

The normal flow is:

```text
public task package
      ↓
manifest + constraint resolution
      ↓
initial correctness validation
      ↓
repair if required
      ↓
verified baseline
      ↓
structured synthesis-feedback PPA search
      ↓
ranked final C/RTL co-simulation audit
      ↓
fully verified selected design
```

Search-time C/RTL co-simulation can be disabled to conserve the weighted validation budget. Final/Pareto co-simulation is a separate post-search verification policy, so a search counter of `cosim_calls: 0` does not mean that the selected design skipped final RTL verification.

## Repository structure

```text
agent/
  core/          orchestration, manifests, budgets, baseline and shared state
  competition/   Track-A staging, scoring, stage-aware tasks and final COSIM
  optimise/      structured PPA search and candidate archive
  repair/        correctness repair and bounded retries
  tools/         Vitis CSim/synthesis/COSIM and validation boundaries
  analysis/      HLS report and bottleneck analysis
  providers/     model API transport
  support/       onboarding, reporting and small utilities
benchmarks/      reproducible public benchmark inputs
configs/         task and experiment-suite definitions
docker/          Vitis-aware container entrypoint and host wrapper
scripts/         one public CLI plus research/audit/reproducibility utilities
tests/           regression and policy tests
```

The package keeps stable compatibility imports such as `agent.config` and `agent.controller`, while the physical implementation is organised under the folders above.

## Setup

Python 3.10+ is supported.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Load Vitis 2025.2 on the host when not using the wrapper:

```bash
source /path/to/Xilinx/2025.2/Vitis/settings64.sh
vitis-run --version
```

Configure a model provider without committing credentials:

```bash
export SILICONFLOW_API_KEY="your-key"
export LLM4HLS_PROVIDER=siliconflow
export LLM4HLS_MODEL="Qwen/Qwen3.5-122B-A10B"
```

## Run a task

Onboard only:

```bash
python3 scripts/run_agent.py <task-directory> --onboard-only
```

Full automatic flow:

```bash
python3 scripts/run_agent.py <task-directory> --mode auto
```

Docker/Vitis flow:

```bash
docker build -t llm4hls-agent:vitis-2025.2 .
docker/run-vitis.sh <task-directory> --mode auto
```

## Validation evidence

Every run writes machine-readable evidence beneath its configured output directory, including the resolved task, budget summary, candidate state, selected source, synthesis metrics and final co-simulation audit when required.

A selected design is considered fully verified only when all required stages are true:

```text
static validation
CSim
synthesis
C/RTL co-simulation
```

## Research tooling

`run_experiment_matrix.py` executes explicit task/model pairs with isolated manifests and resumable state. `run_task_suite.py` discovers and executes task collections sequentially. Analysis/audit scripts remain separate from the public agent entrypoint and do not implement competing orchestration paths.

## Tests

```bash
python -m pytest -q
python -m compileall -q agent scripts tests
```

The Docker image intentionally uses the same package and tests as the host checkout.

## Design principles

- correctness before optimisation;
- one public agent interface;
- benchmark-independent orchestration;
- immutable supplied task/testbench material;
- isolated candidate workspaces;
- static/functional gates before expensive synthesis;
- bounded model and Vitis budgets;
- deterministic final selection with durable evidence;
- mandatory final verification according to task policy.
