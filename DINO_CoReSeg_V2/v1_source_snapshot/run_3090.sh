#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
ACTION="${1:-start}"
if [[ $# -gt 0 ]]; then shift; fi
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
# Preserve scheduler/user masks. Otherwise select the six matching RTX 3090s
# and persist UUID/order so reconnecting does not change checkpoint identity.
case "$ACTION" in
  status|stop) exec ./run_server.sh "$ACTION" "$@" ;;
esac
SELECT_PYTHON="${CORESEG_PYTHON:-${PYTHON_BIN:-python3}}"
CUDA_VISIBLE_DEVICES="$("$SELECT_PYTHON" scripts/select_gpus.py)"
export CUDA_VISIBLE_DEVICES
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
if [[ "$ACTION" == install ]]; then
  exec ./install_server.sh "$@"
fi
exec ./run_server.sh "$ACTION" "$@"
