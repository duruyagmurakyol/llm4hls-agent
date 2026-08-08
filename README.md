# LLM4HLS Agent

Autonomous LLM-based repair and optimisation of AMD/Xilinx HLS C/C++ using compiler, C simulation, synthesis and C/RTL co-simulation feedback.

This repository contains the submission-oriented agent developed for the FPT 2026 LLM4HLS Track-A workflow, together with the retained local benchmark inputs, experiment definitions, Docker integration and regression tests used to evaluate it.

The normal public entry point is always:

```bash
python3 scripts/run_agent.py <task> [options]
```

The agent can accept:

1. a local benchmark directory containing `task.cfg` or a supported HLS TCL description;
2. a persistent JSON task manifest under `configs/tasks/`; or
3. an official Track-A task package containing `task.toml`.

The same agent implementation is used for repair, optimisation, local experiments and competition tasks. There are no separate benchmark-specific agent launchers.

---

## Competition target

The repository uses a single tracked HLS target: AMD Alveo U55C with AMD Vitis HLS 2025.2.

```text
FPGA part:          xcu55c-fsvh2892-2L-e
submission clock:   10 ns
minimum frequency:  100 MHz
```

Every tracked runnable benchmark build description and task manifest is normalised to this U55C part. A regression guard scans tracked `task.cfg`, TCL, JSON and other source files and fails CI if another Xilinx part is introduced. Historical generated result artefacts retain their original provenance; standardising the current runnable configuration does not retroactively reclassify measurements from older runs as U55C results.

A final selected design is considered fully verified only after the required validation chain passes. Depending on task type this includes static checks, CSim, synthesis and final C/RTL co-simulation. Search-time co-simulation can be budgeted separately from the mandatory final verification audit.

---

## Agent flow

```text
benchmark / manifest / Track-A task package
                    │
                    ▼
      onboarding + constraint resolution
                    │
                    ▼
        initial correctness validation
                    │
          ┌─────────┴─────────┐
          │                   │
       invalid              valid
          │                   │
          ▼                   │
        repair                │
          │                   │
          └─────────┬─────────┘
                    ▼
             verified baseline
                    │
                    ▼
      synthesis-feedback optimisation
                    │
                    ▼
        candidate / Pareto selection
                    │
                    ▼
         mandatory final COSIM audit
                    │
                    ▼
          fully verified design
```

Core design rules are:

- correctness before optimisation;
- supplied testbenches and protected task material are not modified;
- cheap validation is performed before expensive Vitis stages;
- model calls and Vitis calls are explicitly budgeted and recorded;
- duplicate candidates are rejected before expensive evaluation;
- only validated source can be promoted as a final design;
- final selection and fallback decisions are persisted as machine-readable evidence.

---

## Repository tree

The tree below shows the submission-relevant structure rather than every individual benchmark file.

```text
llm4hls-agent/
├── agent/
│   ├── __init__.py
│   ├── core/                  # orchestration, task state, budgets, resume
│   ├── competition/           # Track-A staging, scoring, final COSIM
│   ├── optimise/              # synthesis-feedback PPA search / Pareto archive
│   ├── repair/                # diagnosis and correctness repair
│   ├── tools/                 # Vitis execution, parsing and validation
│   ├── analysis/              # HLS hierarchy / bottleneck analysis
│   ├── providers/             # SiliconFlow / OpenRouter transports
│   ├── support/               # onboarding, reporting, prompt compaction
│   └── README.md
│
├── benchmarks/
│   ├── vector_add/            # baseline + controlled repair faults
│   ├── bicg/
│   │   ├── golden/            # retained baseline BiCG package
│   │   ├── u55c/              # portable U55C BiCG regression package
│   │   └── faults/
│   ├── hls_eval/
│   │   ├── atax/              # ATAX source/TB + explicit task_u55c.cfg
│   │   ├── gemm/
│   │   └── vector_add/        # HLS-Eval-style vector add + task_u55c.cfg
│   ├── hls_eval_imported/     # retained Appendix-C fault cases only
│   │   ├── gemver/faults/transpose_access/
│   │   ├── gesummv/faults/accumulator_overwrite/
│   │   ├── mvt/faults/shifted_second_vector/
│   │   └── syrk/faults/missing_diagonal_scaling/
│   ├── repair_suite/          # dot-product and GEMM repair benchmarks
│   ├── capability_suite/      # generation, synth-fix, structural tasks
│   ├── hard_smoke/            # harder optimisation kernels
│   ├── structural_diversity/  # stream/dataflow-oriented tasks
│   └── README.md
│
├── configs/
│   ├── tasks/
│   │   ├── atax_u55c.json
│   │   ├── vector_add_track_a.json
│   │   ├── repair_suite/
│   │   ├── combined_full_agent/
│   │   ├── hls_eval_imported/
│   │   ├── u55c_validation/
│   │   └── specifications/
│   ├── suites/
│   │   ├── overnight_60.json                  # canonical 60-run breadth matrix
│   │   ├── optimisation_15.json               # 5 tasks × 3 models optimisation study
│   │   ├── qwen_bicg_u55c_exact_regression.json
│   │   └── ...
│   └── README.md
│
├── docker/
│   ├── entrypoint.sh
│   ├── run-vitis.sh            # convenience wrapper for official Track-A packages
│   ├── build.sh
│   └── README.md
│
├── scripts/
│   ├── run_agent.py            # public single-task CLI
│   ├── run_experiment_matrix.py
│   ├── run_task_suite.py
│   ├── check_matrix_models.py
│   ├── audit_cosim_suite.py
│   ├── extract_final_results.py
│   ├── analyse_hls_hierarchy.py
│   ├── audit_prompt_compaction.py
│   ├── maintenance/
│   └── README.md
│
├── tests/                       # regression, policy and safety tests
├── Dockerfile
├── requirements.txt
├── .dockerignore
├── .gitignore
└── README.md
```

`docs/` is intentionally absent. The READMEs are the authoritative documentation surface.

Official organiser/reference material is also intentionally absent from the tracked submission tree. A local `external/fpt26-harness/` checkout may be used to reproduce public Track-A experiments, but it is an external evaluator input and is excluded from the Docker build context.

---

# Quick start

## 1. Python environment

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

CI currently exercises the repository with Python 3.12. The code also contains the compatibility path required for Python 3.10 where `tomllib` is unavailable.

## 2. Load Vitis for native runs

The tested workstation installation is:

```bash
source /home/xilinx/Xilinx/2025.2/Vitis/settings64.sh
vitis-run --version
```

The expected tool version is Vitis HLS 2025.2.

If Vitis is installed elsewhere, source that installation's `settings64.sh` before a native run.

## 3. Configure an LLM provider

### SiliconFlow

```bash
export SILICONFLOW_API_KEY="your-key"
export LLM4HLS_PROVIDER=siliconflow
export LLM4HLS_MODEL="Qwen/Qwen3.5-122B-A10B"
```

For example, another retained evaluation model can be selected with:

```bash
export LLM4HLS_MODEL="deepseek-ai/DeepSeek-V4-Pro"
```

### OpenRouter

```bash
export OPENROUTER_API_KEY="your-key"
export LLM4HLS_PROVIDER=openrouter
export LLM4HLS_MODEL="provider/model-id"
```

Do not place API keys in JSON, source files, Dockerfiles or committed shell scripts.

Useful runtime overrides supported by the Docker wrapper include:

```text
LLM4HLS_PROVIDER
LLM4HLS_MODEL
LLM4HLS_TEMPERATURE
LLM4HLS_MAX_OUTPUT_TOKENS
LLM4HLS_MAX_TOTAL_TOKENS
LLM4HLS_API_TIMEOUT_SECONDS
LLM4HLS_COMPACT_PROMPTS
SILICONFLOW_BASE_URL
OPENROUTER_BASE_URL
OPENROUTER_HTTP_REFERER
OPENROUTER_X_TITLE
```

Task manifests can also define model and budget defaults; environment settings are useful when selecting a provider/model at runtime.

---

# Running one task

## CLI syntax

```bash
python3 scripts/run_agent.py TASK \
  [--mode auto|repair|optimise] \
  [--onboard-only] \
  [--resume] \
  [--status-only] \
  [--max-agent-steps N]
```

`TASK` may be a benchmark directory, a JSON task manifest, or an official Track-A task directory.

## Execution modes

### `--mode auto`

Default mode.

```bash
python3 scripts/run_agent.py benchmarks/vector_add --mode auto
```

The agent performs normal task routing: validate the input, repair when necessary, promote a verified baseline and continue into optimisation when the task supports it.

### `--mode repair`

Repair-only execution.

```bash
python3 scripts/run_agent.py \
  benchmarks/vector_add/faults/functional_subtraction \
  --mode repair
```

The run stops after a verified repair rather than deliberately entering a PPA search.

### `--mode optimise`

Optimisation-only execution.

```bash
python3 scripts/run_agent.py benchmarks/bicg/u55c --mode optimise
```

The supplied design must already form a valid baseline. An invalid baseline is rejected rather than silently repaired in optimisation-only mode.

## Inspect a task without running it

`--onboard-only` discovers and validates a task directory without invoking the agent:

```bash
python3 scripts/run_agent.py benchmarks/bicg/u55c --onboard-only
```

For an official Track-A package:

```bash
python3 scripts/run_agent.py \
  external/fpt26-harness/tasks/dotProduct_optimize \
  --onboard-only
```

`--onboard-only` expects a directory rather than a JSON manifest.

## Resume from a verified baseline

```bash
python3 scripts/run_agent.py configs/tasks/atax_u55c.json --resume
```

Resume mode continues an automatic task from its saved fully verified baseline, skipping the initial validation/repair phase. It cannot be combined with `--mode repair` or `--status-only`.

## Limit agent steps

Useful for a bounded smoke run:

```bash
python3 scripts/run_agent.py \
  configs/tasks/atax_u55c.json \
  --mode optimise \
  --max-agent-steps 2
```

## Status-only execution

```bash
python3 scripts/run_agent.py configs/tasks/vector_add_track_a.json --status-only
```

This is intended for compatible non-stage-aware task paths. Stage-aware task types reject `--status-only`.

---

# Targeted U55C reproductions

The whole tracked runnable repository now targets U55C. The paths below are the explicit manifests/packages associated with the reported targeted optimisation experiments.

## ATAX

The reported targeted ATAX experiment uses the committed ATAX source, header and self-checking testbench with an explicit U55C build configuration:

```bash
python3 scripts/run_agent.py \
  configs/tasks/atax_u55c.json \
  --mode optimise
```

Relevant persistent inputs:

```text
benchmarks/hls_eval/atax/src/atax.cpp
benchmarks/hls_eval/atax/src/atax.h
benchmarks/hls_eval/atax/testbench/atax_tb.cpp
benchmarks/hls_eval/atax/task_u55c.cfg
configs/tasks/atax_u55c.json
```

## BiCG

BiCG has a self-contained U55C benchmark package:

```bash
python3 scripts/run_agent.py benchmarks/bicg/u55c --mode optimise
```

Its exact Qwen regression is also available as a matrix suite:

```bash
python3 -u scripts/run_experiment_matrix.py \
  --suite configs/suites/qwen_bicg_u55c_exact_regression.json \
  --run-id qwen_bicg_u55c_regression
```

## Vector Add

Use the explicit task manifest associated with the reported targeted Vector Add experiment:

```bash
python3 scripts/run_agent.py \
  configs/tasks/vector_add_track_a.json \
  --mode optimise
```

The manifest points at the committed U55C build description under the HLS-Eval-style vector-add benchmark.

---

# Running official Track-A tasks

Official Track-A task packages are external inputs containing `task.toml` plus their public source/testbench/description material.

Example native run:

```bash
python3 scripts/run_agent.py \
  external/fpt26-harness/tasks/dotProduct_optimize \
  --mode auto
```

Other public task examples used during evaluation include:

```text
projection_bugfix
residual_stream_deadlock
dotProduct_optimize
```

The Track-A onboarding/staging layer is designed to expose only public task material to the agent. Hidden tests and reference implementations are not copied into the agent-visible workspace.

---

# Experiment matrices

`scripts/run_experiment_matrix.py` is the reproducible task × model experiment runner. It materialises an isolated manifest for every pair, executes runs sequentially, writes state after each run and can safely resume an interrupted suite.

## Matrix syntax

```bash
python3 -u scripts/run_experiment_matrix.py \
  [--suite PATH] \
  [--list] \
  [--core-only] \
  [--max-runs N] \
  [--run-id ID] \
  [--resume-suite RESULTS_DIR] \
  [--rerun-failed] \
  [--stop-on-error] \
  [--no-cleanup-vitis-cache]
```

The default suite is:

```text
configs/suites/overnight_60.json
```

## Inspect the canonical 60-run plan

```bash
python3 -u scripts/run_experiment_matrix.py \
  --suite configs/suites/overnight_60.json \
  --list
```

## Run the full 60-run breadth matrix

```bash
python3 -u scripts/run_experiment_matrix.py \
  --suite configs/suites/overnight_60.json \
  --run-id overnight_60_reproduction
```

`overnight_60.json` contains 20 tasks × 3 models = 60 explicit runs. It is the canonical breadth-study definition retained for the final evaluation.

The suite uses:

```text
deepseek-ai/DeepSeek-V4-Pro
Qwen/Qwen3.5-122B-A10B
Qwen/Qwen3.6-27B
```

The matrix includes three external organiser tasks. Therefore a full 60-run reproduction requires the public harness to be available locally at the paths referenced by the suite, normally:

```text
external/fpt26-harness/tasks/projection_bugfix
external/fpt26-harness/tasks/residual_stream_deadlock
external/fpt26-harness/tasks/dotProduct_optimize
```

The harness is not committed to this repository.

## Core-only matrix

```bash
python3 -u scripts/run_experiment_matrix.py \
  --suite configs/suites/overnight_60.json \
  --core-only \
  --run-id overnight_60_core
```

## Small smoke subset

```bash
python3 -u scripts/run_experiment_matrix.py \
  --suite configs/suites/overnight_60.json \
  --max-runs 3 \
  --run-id matrix_smoke
```

## Resume an interrupted matrix

Suite state is stored under `results/suites/<run-id>/`.

```bash
python3 -u scripts/run_experiment_matrix.py \
  --resume-suite results/suites/overnight_60_reproduction
```

To retry failed rows while resuming:

```bash
python3 -u scripts/run_experiment_matrix.py \
  --resume-suite results/suites/overnight_60_reproduction \
  --rerun-failed
```

`--resume-suite` cannot be combined with `--core-only`, `--max-runs` or `--run-id`.

Matrix-level metadata is written to:

```text
results/suites/<run-id>/
├── suite_definition.json
├── matrix_manifest.json
├── environment.json
├── suite_state.json
├── suite_summary.csv
├── manifests/
└── logs/
```

Individual task outputs are isolated under:

```text
experiments/model_comparison/<run-id>/<model-slug>/<task-id>/
```

---

# Sequential task suites

`scripts/run_task_suite.py` is a discovery/provenance-oriented sequential suite runner. It is useful when the desired input is a benchmark collection rather than an explicit task × model matrix.

## Task-suite syntax

```bash
python3 -u scripts/run_task_suite.py \
  [--suite PATH] \
  [--run-id ID] \
  [--resume-suite RESULTS_DIR] \
  [--rerun-failed] \
  [--list] \
  [--dry-run] \
  [--only PATTERN] \
  [--skip PATTERN] \
  [--max-tasks N] \
  [--fresh] \
  [--clear-vitis-cache] \
  [--resume-tasks]
```

The default task-suite definition is:

```text
configs/suites/overnight_full.json
```

List discovered tasks without executing them:

```bash
python3 -u scripts/run_task_suite.py \
  --suite configs/suites/overnight_full.json \
  --list
```

Dry-run a filtered subset:

```bash
python3 -u scripts/run_task_suite.py \
  --suite configs/suites/overnight_full.json \
  --only '*bicg*' \
  --max-tasks 3 \
  --dry-run
```

Skip a family:

```bash
python3 -u scripts/run_task_suite.py \
  --suite configs/suites/overnight_full.json \
  --skip '*track_a*'
```

Like the matrix runner, suite-level state and CSV summaries are stored under:

```text
results/suites/<run-id>/
```

---

# Docker

Docker packages the Python agent and its dependencies. Vitis itself is **not** copied into the image; the host Vitis 2025.2 installation is mounted read-only at runtime.

The Dockerfile is based on the Xilinx Alveo runtime userspace and compiles the Python source as part of the image build.

## Build the image

From the repository root:

```bash
docker build -t llm4hls-agent:vitis-2025.2 .
```

or:

```bash
./docker/build.sh
```

## Default host Vitis location

The supplied wrapper expects:

```text
/home/xilinx/Xilinx/2025.2
```

and sources:

```text
/home/xilinx/Xilinx/2025.2/Vitis/settings64.sh
```

Use another host installation root with:

```bash
export HOST_XILINX_ROOT=/path/to/Xilinx/2025.2
```

If the Xilinx settings scripts contain absolute installation paths, mount the installation at the same absolute path inside the container.

---

## Official Track-A task in Docker

`docker/run-vitis.sh` is the convenient wrapper for **official Track-A directories containing `task.toml`**.

```bash
export SILICONFLOW_API_KEY="your-key"
export LLM4HLS_PROVIDER=siliconflow
export LLM4HLS_MODEL="Qwen/Qwen3.5-122B-A10B"

./docker/run-vitis.sh \
  /path/to/fpt26-harness/tasks/dotProduct_optimize \
  --mode auto
```

Onboarding only:

```bash
./docker/run-vitis.sh \
  /path/to/fpt26-harness/tasks/dotProduct_optimize \
  --onboard-only
```

The wrapper:

- verifies that the directory contains `task.toml`;
- mounts the task read-only at `/task`;
- mounts Vitis read-only;
- passes supported model/runtime environment variables without embedding their values on the command line;
- persists generated experiment outputs back to the host `experiments/` directory;
- invokes the same `scripts/run_agent.py` entry point used natively.

Override the image name if required:

```bash
IMAGE=llm4hls-agent:my-build \
./docker/run-vitis.sh /path/to/task --mode auto
```

---

## Regular local benchmark run in Docker

For a benchmark already copied into the image, call the image directly. The Docker entrypoint accepts the same `run_agent.py` arguments.

From the repository root:

```bash
mkdir -p experiments

docker run --rm -it \
  --user "$(id -u):$(id -g)" \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -v /home/xilinx/Xilinx/2025.2:/home/xilinx/Xilinx/2025.2:ro \
  -v "$PWD/experiments:/workspace/llm4hls-agent/experiments" \
  -e SILICONFLOW_API_KEY \
  -e LLM4HLS_PROVIDER \
  -e LLM4HLS_MODEL \
  llm4hls-agent:vitis-2025.2 \
  benchmarks/bicg/u55c --mode optimise
```

Run ATAX from its JSON manifest in Docker:

```bash
docker run --rm -it \
  --user "$(id -u):$(id -g)" \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -v /home/xilinx/Xilinx/2025.2:/home/xilinx/Xilinx/2025.2:ro \
  -v "$PWD/experiments:/workspace/llm4hls-agent/experiments" \
  -e SILICONFLOW_API_KEY \
  -e LLM4HLS_PROVIDER \
  -e LLM4HLS_MODEL \
  llm4hls-agent:vitis-2025.2 \
  configs/tasks/atax_u55c.json --mode optimise
```

The same approach works with `--mode auto`, `--mode repair`, `--onboard-only`, `--resume`, `--status-only` and `--max-agent-steps` where those options are valid for the selected task.

---

## Run an experiment matrix in Docker

The normal image entrypoint runs one task, so invoke Python explicitly for a matrix.

For a fully local suite:

```bash
mkdir -p experiments results

docker run --rm -it \
  --user "$(id -u):$(id -g)" \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -v /home/xilinx/Xilinx/2025.2:/home/xilinx/Xilinx/2025.2:ro \
  -v "$PWD/experiments:/workspace/llm4hls-agent/experiments" \
  -v "$PWD/results:/workspace/llm4hls-agent/results" \
  -e SILICONFLOW_API_KEY \
  --entrypoint bash \
  llm4hls-agent:vitis-2025.2 \
  -lc 'source /home/xilinx/Xilinx/2025.2/Vitis/settings64.sh && \
       python3 -u scripts/run_experiment_matrix.py \
       --suite configs/suites/qwen_bicg_u55c_exact_regression.json \
       --run-id docker_bicg_regression'
```

To run `overnight_60.json` inside Docker, the external public Track-A harness must also be mounted because it is intentionally excluded from the image:

```bash
-v "$PWD/external/fpt26-harness:/workspace/llm4hls-agent/external/fpt26-harness:ro"
```

Add that mount to the matrix command before the image name.

---

# Output and evidence

Generated outputs are intentionally ignored by Git. A source checkout therefore stays small while each execution recreates its own evidence.

A typical single-task output directory can contain:

```text
resolved_task.json
original_scoring_baseline.json
verified_baseline.json
candidate_state.json
pareto_frontier.json
budget_summary.json
experiment_summary.json
final_cosim_audit.json
track_a_score_estimate.json
unified_agent_result.json
candidate_archive/
csim/
synthesis/
cosim/
```

The exact set depends on the task type and stages reached.

Important files include:

- `resolved_task.json` — the resolved source/testbench/build/target/model contract for the run;
- `candidate_state.json` — selected design, metrics, validation state and candidate provenance;
- `budget_summary.json` — model/token/tool usage and remaining budget;
- `experiment_summary.json` — compact run-level result summary;
- `final_cosim_audit.json` — final C/RTL verification outcome and fallback audit;
- `track_a_score_estimate.json` — public reference-harness estimate when available;
- `unified_agent_result.json` — stable top-level machine-readable result;
- `candidate_archive/` — promoted, Pareto and selected candidate source snapshots.

Generated directories such as these remain ignored:

```text
experiments/
results/
logs/
runs/
.Xil/
solution*/
Vitis work directories
```

They should not be confused with persistent benchmark/config inputs under `benchmarks/` and `configs/`.

---

# Retained evaluation definitions

## Canonical 60-run breadth study

```text
configs/suites/overnight_60.json
```

20 tasks × 3 models = 60 explicit runs spanning generation, synthesis repair, functional repair, structural repair and optimisation.

## Optimisation comparison

```text
configs/suites/optimisation_15.json
```

Five optimisation tasks × three models.

## Exact BiCG U55C regression

```text
configs/suites/qwen_bicg_u55c_exact_regression.json
benchmarks/bicg/u55c/
```

## Repeated full-agent repair-to-optimisation cases

Persistent manifests are retained under:

```text
configs/tasks/combined_full_agent/
```

The four HLS-Eval-derived fault inputs required by the retained repeated evaluation are committed under `benchmarks/hls_eval_imported/` rather than depending on ignored local copies.

The current runnable forms of these repair-to-optimisation cases use the repository-wide U55C target. The dedicated four-case competition-target validation subset is retained under:

```text
configs/tasks/u55c_validation/
```

---

# Utility scripts

The public agent remains `scripts/run_agent.py`. Other scripts support experiments and analysis rather than implementing alternative agents.

```text
scripts/run_experiment_matrix.py  explicit task × model comparison
scripts/run_task_suite.py         discovery/provenance suite execution
scripts/check_matrix_models.py     provider/model preflight
scripts/audit_cosim_suite.py       post-search final/Pareto COSIM audit
scripts/extract_final_results.py   JSON/CSV/Markdown result extraction
scripts/analyse_hls_hierarchy.py   synthesis hierarchy/bottleneck analysis
scripts/audit_prompt_compaction.py offline prompt/token analysis
```

See `scripts/README.md` for the narrower script-level description.

---

# Validation

Run the Python validation gate from the repository root:

```bash
python -m compileall -q agent scripts tests
python -m pytest -q
```

The test suite includes a single-target policy guard. It checks tracked benchmark `task.cfg` files, HLS TCL `set_part` commands, JSON `part`/`target_part` fields and tracked Xilinx part tokens, and rejects any target other than `xcu55c-fsvh2892-2L-e`.

For an image-level check:

```bash
docker build -t llm4hls-agent:vitis-2025.2 .

docker run --rm \
  --entrypoint bash \
  llm4hls-agent:vitis-2025.2 \
  -lc 'cd /workspace/llm4hls-agent && python3 -m pytest -q'
```

Verify that the host Vitis installation is visible inside the image:

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -v /home/xilinx/Xilinx/2025.2:/home/xilinx/Xilinx/2025.2:ro \
  --entrypoint bash \
  llm4hls-agent:vitis-2025.2 \
  -lc 'source /home/xilinx/Xilinx/2025.2/Vitis/settings64.sh && vitis-run --version'
```

---

# Reproducibility boundaries

The repository intentionally distinguishes persistent inputs from generated/external material.

### Version-controlled

- agent implementation;
- Docker integration;
- regression tests;
- persistent task manifests and suite definitions;
- retained local benchmark source, headers, testbenches and build descriptions;
- U55C benchmark/config variants required by the targeted experiments;
- the four HLS-Eval-derived fault cases used by the retained repeated full-agent evaluation.

### External by design

- organiser Track-A task packages/reference harness;
- hidden evaluator tests;
- hidden/reference implementations;
- the host Vitis 2025.2 installation;
- model-provider services.

### Generated and ignored

- candidate prompts and model responses;
- run summaries and experiment outputs;
- CSim/synthesis/COSIM workspaces;
- Vitis-generated project/cache files;
- archived candidate evidence from individual historical runs.

A retained persistent configuration should point either to a committed benchmark under `benchmarks/` or to an explicitly external organiser task package. It should never require an ignored historical `experiments/` directory merely to define its inputs.

---

# Example verified competition run

A Dockerised public `dotProduct_optimize` experiment targeting U55C completed with final C/RTL co-simulation passing. The selected design achieved 256-cycle reference latency, approximately 142.6 MHz estimated frequency and a public reference-harness score estimate of 2.5509 / 3.0000, using 6,198 model tokens and 20 / 40 weighted search credits.

This is a synthesis and C/RTL co-simulation result targeting the U55C device; it is not a claim of physical execution on an FPGA board.

Detailed generated artefacts are intentionally kept outside source control and can be curated separately for a submission evidence package.
