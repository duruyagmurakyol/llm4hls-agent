# Dockerised Vitis execution

The Docker image packages the agent and Python dependencies while mounting the host AMD/Xilinx Vitis 2025.2 installation read-only. Vitis itself is not copied into the image.

## Build

From the repository root:

```bash
docker build -t llm4hls-agent:vitis-2025.2 .
```

or use the helper:

```bash
./docker/build.sh
```

## Host Vitis layout

The wrapper defaults to:

```text
/home/xilinx/Xilinx/2025.2
```

and the entrypoint sources:

```text
/home/xilinx/Xilinx/2025.2/Vitis/settings64.sh
```

Override the host installation root with `HOST_XILINX_ROOT` when necessary.

## Run an official Track-A task

```bash
export SILICONFLOW_API_KEY="your-key"
export LLM4HLS_PROVIDER=siliconflow
export LLM4HLS_MODEL="Qwen/Qwen3.5-122B-A10B"

./docker/run-vitis.sh /path/to/fpt26-harness/tasks/dotProduct_optimize --mode auto
```

Onboarding only:

```bash
./docker/run-vitis.sh /path/to/fpt26-harness/tasks/dotProduct_optimize --onboard-only
```

The task directory is mounted read-only. The Track-A adapter stages only public task files before invoking `scripts/run_agent.py`; hidden tests and reference implementations are not copied into the agent-visible workspace.

Generated run artefacts persist on the host under ignored experiment/result directories.

## Secrets

API keys are passed through the host environment at runtime. They are not stored in the Dockerfile or image.
