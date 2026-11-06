#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
ACTION="${1:-start}"
if [[ $# -gt 0 ]]; then shift; fi
PYTHON="${CORESEG_PYTHON:-.venv/bin/python}"
LAUNCH_DIR="outputs/launcher"
mkdir -p "$LAUNCH_DIR"
PID_FILE="$LAUNCH_DIR/pipeline.pid"
CONFIG_FILE="$PWD/configs/gpu_100.yaml"
is_running() {
  [[ -f "$PID_FILE" ]] || return 1
  local saved_pid command_line
  saved_pid="$(cat "$PID_FILE")"
  [[ "$saved_pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$saved_pid" 2>/dev/null || return 1
  command_line="$(ps -ww -p "$saved_pid" -o args= 2>/dev/null || true)"
  [[ "$command_line" == *"src.server_pipeline"* && "$command_line" == *"$CONFIG_FILE"* ]]
}
case "$ACTION" in
  check)
    exec "$PYTHON" -u -m src.server_pipeline --config "$CONFIG_FILE" --check-only "$@"
    ;;
  foreground)
    exec "$PYTHON" -u -m src.server_pipeline --config "$CONFIG_FILE" "$@"
    ;;
  start|resume)
    if is_running; then
      echo "A saved process is still alive: $(cat "$PID_FILE"). Check ./run_server.sh status."
      exit 1
    fi
    nohup "$PYTHON" -u -m src.server_pipeline --config "$CONFIG_FILE" "$@" >> "$LAUNCH_DIR/console.log" 2>&1 < /dev/null &
    echo "$!" > "$PID_FILE"
    echo "Started PID $(cat "$PID_FILE"). It automatically resumes existing progress."
    echo "Log: $PWD/$LAUNCH_DIR/console.log"
    ;;
  status)
    if is_running; then
      echo "Process alive: $(cat "$PID_FILE")"
    else echo 'No live launcher process found.'; fi
    "$PYTHON" scripts/status.py
    ;;
  stop)
    if is_running; then
      kill -TERM "$(cat "$PID_FILE")"
      echo 'SIGTERM sent. The program saves at a safe batch boundary; inspect status before restarting.'
    else echo 'No live launcher process found.'; fi
    ;;
  *) echo 'Usage: ./run_server.sh {check|start|resume|foreground|status|stop} [pipeline options]'; exit 2 ;;
esac
