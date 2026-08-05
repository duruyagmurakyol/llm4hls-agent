# Containerisation and clean-room workflow

## Scope

The container packages the Python controller, tests and command-line workflow. AMD Vitis HLS is proprietary and is not redistributed in the image. For local validation, the host Vitis 2025.2 installation is mounted read-only into the container.

This scaffold is preparation for the final competition harness. The official FPT harness interface must still be checked before the submission image and entry command are frozen.

## Build

From the repository root:

```bash
docker build -f docker/Dockerfile -t llm4hls-agent:dev .
```

The build runs the Python test suite. A failed test prevents the image from being created.

## Controller-only smoke test

```bash
docker run --rm llm4hls-agent:dev \
  python3 scripts/run_agent.py --help
```

## Local Vitis-enabled smoke test

The host installation is mounted read-only. No AMD binaries or licences are copied into the image.

```bash
docker run --rm \
  -e SILICONFLOW_API_KEY \
  -e VITIS_ROOT=/opt/Xilinx/2025.2 \
  -v "$PWD:/workspace/llm4hls-agent" \
  -v /home/xilinx/Xilinx/2025.2:/opt/Xilinx/2025.2:ro \
  llm4hls-agent:dev \
  bash -lc 'command -v vitis-run && python3 scripts/run_agent.py --help'
```

Alternatively:

```bash
docker compose -f docker/compose.yaml run --rm llm4hls-agent
```

## Clean-room rehearsal

The final rehearsal must be performed from a fresh clone and must not rely on generated files from the development checkout.

```text
fresh clone
-> import or provide benchmark inputs
-> build container
-> inject API key through the environment
-> expose licensed Vitis 2025.2 installation
-> run one representative task
-> verify C simulation
-> verify synthesis
-> verify C/RTL co-simulation
-> verify final selected design and result JSON
```

## Security and packaging rules

- Never bake `SILICONFLOW_API_KEY` into the image or repository.
- Never copy AMD installation files, licence files or user caches into the image.
- Keep `runs/`, synthesis projects, logs and generated benchmark imports outside the build context.
- Pin the final Git commit and image digest in the submission manifest.
- Run a secret scan and inspect the final archive before submission.

## Remaining work

1. Verify Docker is available on the clean target machine.
2. Confirm the official FPT harness input/output and image-entry requirements.
3. Replace the development command with the competition entry command.
4. Run the complete clean-room repair-to-optimisation flow on U55C.
5. Record the image digest, command, runtime and generated evidence.
