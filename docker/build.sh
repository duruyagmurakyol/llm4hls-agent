#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-llm4hls-agent:vitis-2025.2}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

exec docker build --tag "${IMAGE}" "${REPO_ROOT}"
