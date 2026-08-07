#!/usr/bin/env bash
set -euo pipefail

VITIS_ROOT="${LLM4HLS_VITIS_ROOT:-/tools/Xilinx/Vitis/2025.2}"
SETTINGS="${VITIS_ROOT}/settings64.sh"

if [[ ! -f "${SETTINGS}" ]]; then
  echo "ERROR: Vitis 2025.2 settings not found at ${SETTINGS}" >&2
  echo "Mount the host Xilinx installation or set LLM4HLS_VITIS_ROOT." >&2
  exit 2
fi

# shellcheck disable=SC1090
source "${SETTINGS}"

if ! command -v vitis-run >/dev/null 2>&1; then
  echo "ERROR: vitis-run is unavailable after sourcing ${SETTINGS}" >&2
  exit 2
fi

cd /workspace/llm4hls-agent

# The image has one public purpose: run the existing unified agent on a task
# package or benchmark directory. Extra CLI arguments are passed through.
exec python3 scripts/run_agent.py "$@"
