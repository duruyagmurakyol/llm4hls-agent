#!/usr/bin/env bash
set -euo pipefail

# AMD setup scripts embed the absolute installation path. The default therefore
# matches the host installation path used by docker/compose.yaml.
VITIS_ROOT="${VITIS_ROOT:-/home/xilinx/Xilinx/2025.2}"

for settings in \
  "$VITIS_ROOT/Vitis/settings64.sh" \
  "$VITIS_ROOT/settings64.sh"; do
  if [[ -f "$settings" ]]; then
    # AMD setup scripts are not guaranteed to be nounset-safe.
    set +u
    # shellcheck disable=SC1090
    source "$settings"
    set -u
    break
  fi
done

if command -v vitis-run >/dev/null 2>&1; then
  echo "Vitis available: $(command -v vitis-run)"
else
  echo "Warning: vitis-run is unavailable." >&2
  echo "Mount a licensed Vitis 2025.2 installation at $VITIS_ROOT or run controller-only tests." >&2
fi

exec "$@"
