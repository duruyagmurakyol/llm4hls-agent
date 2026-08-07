# LLM4HLS Agent

Autonomous LLM-based repair and optimisation of AMD/Xilinx HLS C/C++ using compiler, CSim, synthesis and C/RTL co-simulation feedback.

## Competition target

The submission path targets AMD Alveo U55C with Vitis HLS 2025.2:

```text
part: xcu55c-fsvh2892-2L-e
minimum frequency: 100 MHz
submission clock: 10 ns
```

The public entry point is:

```bash
python3 scripts/run_agent.py <task-package-or-manifest> --mode auto
```

The reproducible container path is:

```bash
docker/run-vitis.sh <task-directory> --mode auto
```

## Agent flow

```text
task package / benchmark
        ↓
public-file onboarding and constraint resolution
        ↓
initial correctness validation
        ↓
repair when required
        ↓
verified baseline
        ↓
structured synthesis-feedback optimisation
        ↓
ranked final C/RTL co-simulation audit
        ↓
fully verified selected design
```

A final design is fully verified only after static validation, CSim, synthesis and C/RTL co-simulation all pass. Search-time co-simulation may be disabled to conserve the weighted tool budget; the mandatory final audit is separate from that search counter.

## Repository layout

```text
agent/       reusable repair, optimisation and competition logic
benchmarks/  committed benchmark inputs and self-checking testbenches
configs/     persistent task and experiment-suite definitions
docker/      Vitis-aware container wrapper and entrypoint
scripts/     public CLI plus experiment/audit utilities
tests/       regression and policy tests
```

Historical development notes are intentionally not kept as a separate `docs/` tree; the READMEs are the authoritative documentation surface.

## What is version controlled

The repository contains the agent source, Docker integration, persistent experiment configurations, tests and the benchmark inputs needed for the retained local experiments. Representative committed benchmark families include:

- vector addition;
- BiCG, including a U55C-targeted regression package;
- ATAX;
- GEMM;
- dot-product and GEMM repair suites;
- hard-smoke kernels;
- capability-suite generate/synthesis/structural tasks;
- structural-diversity stream tasks.

Official FPT Track-A task packages are supplied by the organiser/evaluator and are therefore external inputs rather than bundled reference material.

Generated artefacts are deliberately not committed: `experiments/`, `results/`, `logs/`, Vitis projects, model responses and synthesis/co-simulation work directories are ignored. Each run regenerates its own machine-readable evidence.

## Reproducing the retained BiCG regression

The Qwen U55C regression is defined by:

```text
configs/suites/qwen_bicg_u55c_exact_regression.json
benchmarks/bicg/u55c/
```

Run it with:

```bash
python3 -u scripts/run_experiment_matrix.py \
  --suite configs/suites/qwen_bicg_u55c_exact_regression.json \
  --run-id qwen_bicg_u55c_regression
```

## Running an official Track-A task

Set the model provider without storing credentials:

```bash
export SILICONFLOW_API_KEY="your-key"
export LLM4HLS_PROVIDER=siliconflow
export LLM4HLS_MODEL="Qwen/Qwen3.5-122B-A10B"
```

Onboard only:

```bash
docker/run-vitis.sh /path/to/fpt26-harness/tasks/dotProduct_optimize --onboard-only
```

Full run:

```bash
docker/run-vitis.sh /path/to/fpt26-harness/tasks/dotProduct_optimize --mode auto
```

The Track-A adapter stages only public kernel/header/testbench/description metadata. Hidden tests and reference implementations are not copied into the agent-visible workspace or Docker image.

## Example verified result

A Dockerised `dotProduct_optimize` run on the U55C target completed with final C/RTL co-simulation passing, selected latency of 256 cycles, estimated frequency of 142.6 MHz and a public reference-harness score estimate of 2.5509/3.0000. The run used 6,198 model tokens and 20/40 weighted search credits. Detailed run artefacts are generated locally rather than committed.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For native execution, load Vitis 2025.2 first:

```bash
source /home/xilinx/Xilinx/2025.2/Vitis/settings64.sh
vitis-run --version
```

## Validation

```bash
python -m compileall -q agent scripts tests
python -m pytest -q
```

The same source tree is copied into the Docker image, and the host and container test collections are expected to match.
