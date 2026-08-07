# Dockerised Track-A execution

This container packages the existing `structured-optimisation-v1` agent while
using the host machine's AMD/Xilinx Vitis 2025.2 installation. Vitis is not
copied into the image.

The Track-A adapter in this branch keeps the competition submission target at
10 ns (100 MHz). Docker does not override that target.

## Build

From the repository root:

```bash
chmod +x docker/build.sh docker/run-vitis.sh docker/entrypoint.sh
./docker/build.sh
```

The default image name is:

```text
llm4hls-agent:vitis-2025.2
```

Override it with `IMAGE=...` if required.

## Host Vitis layout

The runner currently defaults to the project workstation layout:

```text
/home/xilinx/Xilinx/Vitis/2025.2/settings64.sh
```

If the Xilinx installation is elsewhere, set `HOST_XILINX_ROOT` to the host
directory that contains `Vitis/`.

Example:

```bash
export HOST_XILINX_ROOT=/tools/Xilinx
```

Inside the container that directory is mounted read-only at `/tools/Xilinx`,
and the entrypoint sources:

```text
/tools/Xilinx/Vitis/2025.2/settings64.sh
```

## Run an official Track-A task

Set the same model-provider variables used by native branch runs, then pass the
official task directory to the wrapper:

```bash
export SILICONFLOW_API_KEY=...
export LLM4HLS_PROVIDER=siliconflow
export LLM4HLS_MODEL=Qwen/Qwen3.5-122B-A10B

./docker/run-vitis.sh /path/to/fpt26-harness/tasks/dotProduct_optimize
```

The task is mounted read-only at `/task`. The branch's Track-A compatibility
frontend stages only the public task files before invoking the existing unified
repair/optimisation controller.

Generated run artefacts are persisted into the repository's host-side
`experiments/` directory.

Any normal `scripts/run_agent.py` option can follow the task directory, for
example:

```bash
./docker/run-vitis.sh /path/to/task --max-agent-steps 1
```

## Smoke-test order

Use the supplied reference tasks in this order:

1. `dotProduct_optimize` - proves onboarding, CSim, synthesis and optimisation.
2. `projection_bugfix` - proves repair inside the container.
3. `residual_stream_deadlock` - proves CoSim/deadlock handling inside Docker.

## Secrets

API keys are passed through from the host environment at runtime. They are not
stored in the image or Dockerfile.
