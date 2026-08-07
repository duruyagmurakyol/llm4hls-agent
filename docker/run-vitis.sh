#!/usr/bin/env bash
set -euo pipefail

# Run one official Track-A task package through the Dockerised unified agent.
#
# The workstation's Xilinx settings scripts contain absolute paths rooted at
# /home/xilinx/Xilinx/2025.2. The installation is therefore mounted at that
# exact same path inside the container.
#
# Override with IMAGE or HOST_XILINX_ROOT when necessary. If using a different
# installation path whose settings scripts also contain absolute paths, mount
# it at the same path inside the container.

IMAGE="${IMAGE:-llm4hls-agent:vitis-2025.2}"
HOST_XILINX_ROOT="${HOST_XILINX_ROOT:-/home/xilinx/Xilinx/2025.2}"
CONTAINER_XILINX_ROOT="${CONTAINER_XILINX_ROOT:-${HOST_XILINX_ROOT}}"
CONTAINER_VITIS_ROOT="${CONTAINER_XILINX_ROOT}/Vitis"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <task-directory> [run_agent.py options...]" >&2
  exit 2
fi

TASK_DIR="$(realpath "$1")"
shift

if [[ ! -d "${TASK_DIR}" ]]; then
  echo "ERROR: task directory does not exist: ${TASK_DIR}" >&2
  exit 2
fi

if [[ ! -f "${TASK_DIR}/task.toml" ]]; then
  echo "ERROR: expected official Track-A task.toml in ${TASK_DIR}" >&2
  exit 2
fi

if [[ ! -f "${HOST_XILINX_ROOT}/Vitis/settings64.sh" ]]; then
  echo "ERROR: Vitis 2025.2 not found under ${HOST_XILINX_ROOT}" >&2
  echo "Set HOST_XILINX_ROOT to the 2025.2 directory containing Vitis/." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
mkdir -p "${REPO_ROOT}/experiments"

TTY=()
if [[ -t 0 && -t 1 ]]; then
  TTY=(-it)
fi

ENV_ARGS=(
  -e "LLM4HLS_VITIS_ROOT=${CONTAINER_VITIS_ROOT}"
)

# Pass through only the model/runtime settings the agent understands. Docker's
# '-e NAME' form reads the value from the host without exposing it on the CLI.
for name in \
  SILICONFLOW_API_KEY SILICONFLOW_KEY SILICONFLOW_BASE_URL \
  OPENROUTER_API_KEY OPENROUTER_BASE_URL OPENROUTER_HTTP_REFERER OPENROUTER_X_TITLE \
  LLM4HLS_PROVIDER LLM4HLS_MODEL LLM4HLS_TEMPERATURE \
  LLM4HLS_MAX_OUTPUT_TOKENS LLM4HLS_MAX_TOTAL_TOKENS \
  LLM4HLS_API_TIMEOUT_SECONDS LLM4HLS_COMPACT_PROMPTS; do
  if [[ -n "${!name:-}" ]]; then
    ENV_ARGS+=(-e "${name}")
  fi
done

echo "run-vitis: image=${IMAGE} task=${TASK_DIR} vitis=${HOST_XILINX_ROOT}/Vitis" >&2

exec docker run --rm "${TTY[@]}" \
  --user "$(id -u):$(id -g)" \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -v "${HOST_XILINX_ROOT}:${CONTAINER_XILINX_ROOT}:ro" \
  -v "${TASK_DIR}:/task:ro" \
  -v "${REPO_ROOT}/experiments:/workspace/llm4hls-agent/experiments" \
  "${ENV_ARGS[@]}" \
  "${IMAGE}" /task "$@"
