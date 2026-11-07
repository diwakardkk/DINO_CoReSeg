#!/usr/bin/env bash
set -euo pipefail
cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")"
ACTION="${1:-status}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ "$ACTION" == install ]]; then exec ./install_v2.sh "$@"; fi
PYTHON="${CORESEG_V2_PYTHON:-.venv/bin/python}"
CONFIG="configs/v2_resolution_448.yaml"
NEED_CONFIG=0
for argument in "$@"; do
  if [[ "$NEED_CONFIG" == 1 ]]; then CONFIG="$argument"; NEED_CONFIG=0
  elif [[ "$argument" == --config ]]; then NEED_CONFIG=1
  elif [[ "$argument" == --config=* ]]; then CONFIG="${argument#--config=}"
  fi
done
if [[ "$NEED_CONFIG" == 1 || -z "$CONFIG" ]]; then echo '--config needs a path'; exit 2; fi
CONFIG="$("$PYTHON" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$CONFIG")"
EXPERIMENT="$("$PYTHON" -c 'import yaml,sys,re; x=yaml.safe_load(open(sys.argv[1]))["experiment"]; assert re.fullmatch(r"dino_coreseg_v2_[A-Za-z0-9_]+",x); print(x)' "$CONFIG")"
LOG_DIR="outputs/$EXPERIMENT/logs"
mkdir -p "$LOG_DIR"
PID_FILE="$LOG_DIR/pipeline.pid"
is_running() {
  [[ -f "$PID_FILE" ]] || return 1
  local job_pid command_line
  job_pid="$(cat "$PID_FILE")"
  [[ "$job_pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$job_pid" 2>/dev/null || return 1
  command_line="$(ps -ww -p "$job_pid" -o args= 2>/dev/null || true)"
  [[ "$command_line" == *src.v2_pipeline* && "$command_line" == *"$CONFIG"* ]]
}
case "$ACTION" in
  test) CUDA_VISIBLE_DEVICES='' exec "$PYTHON" -m pytest -q tests ;;
  status)
    if is_running; then echo "Process alive: $(cat "$PID_FILE")"; else echo 'No live V2 launcher process found.'; fi
    "$PYTHON" -c 'from pathlib import Path; import sys; p=Path(sys.argv[1]); print(p.read_text() if p.exists() else "No V2 pipeline status yet")' "$LOG_DIR/pipeline_status.json"
    exit ;;
  stop)
    if is_running; then kill -TERM "$(cat "$PID_FILE")"; echo 'Graceful stop requested; wait for the committed optimizer boundary.'; else echo 'No live V2 process found.'; fi
    exit ;;
  plots)
    exec "$PYTHON" -m src.plot_paper --experiment "$EXPERIMENT" ;;
esac
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to explicitly allocated GPU(s); V2 does not auto-select shared GPUs}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export PYTHONFAULTHANDLER=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
case "$ACTION" in
  start|resume|start-cv|resume-cv)
    if is_running; then echo "V2 already running: $(cat "$PID_FILE")"; exit 1; fi
    STAGE=all
    if [[ "$ACTION" == *-cv ]]; then STAGE=cv; fi
    nohup "$PYTHON" -u -m src.v2_pipeline --config "$CONFIG" "$@" --stage "$STAGE" >> "$LOG_DIR/console.log" 2>&1 < /dev/null &
    echo "$!" > "$PID_FILE"
    echo "Started V2 PID $(cat "$PID_FILE"); automatically resumes only matching V2 state."
    echo "Log: $PWD/$LOG_DIR/console.log" ;;
  foreground) exec "$PYTHON" -u -m src.v2_pipeline --config "$CONFIG" "$@" --stage all ;;
  check|prepare|smoke|extract|train|evaluate|cv|report-v1|cleanup-dry-run)
    exec "$PYTHON" -u -m src.v2_pipeline --config "$CONFIG" "$@" --stage "$ACTION" ;;
  *) echo 'Usage: run_v2.sh {install|test|check|prepare|smoke|extract|start|resume|foreground|train|evaluate|cv|start-cv|resume-cv|plots|status|stop|report-v1|cleanup-dry-run} [--config PATH] [--data-root PATH] [--v1-root PATH]'; exit 2 ;;
esac
