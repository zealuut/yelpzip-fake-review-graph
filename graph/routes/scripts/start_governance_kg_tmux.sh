#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SESSION_NAME="${1:-governance_kg}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed; cannot start durable background queue" >&2
  exit 1
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME" >&2
  echo "attach: tmux attach -t $SESSION_NAME" >&2
  echo "status: bash $ROOT_DIR/graph/routes/scripts/status_governance_kg.sh" >&2
  exit 1
fi

mkdir -p "$ROOT_DIR/graph/logs"
MONITOR_LOG="$ROOT_DIR/graph/logs/governance_kg_monitor_$(date +%Y%m%d_%H%M%S).log"

tmux new-session -d -s "$SESSION_NAME" -c "$ROOT_DIR" \
  "bash graph/routes/scripts/run_governance_kg_queue.sh; status=\$?; echo; echo \"tmux runner finished status=\$status at \$(date -Is)\"; exec bash"

tmux new-window -d -t "$SESSION_NAME:" -n monitor -c "$ROOT_DIR" \
  "while true; do bash graph/routes/scripts/status_governance_kg.sh --brief; sleep 60; done | tee -a '$MONITOR_LOG'"

echo "started tmux session: $SESSION_NAME"
echo "attach: tmux attach -t $SESSION_NAME"
echo "watch pane: tmux capture-pane -pt $SESSION_NAME:0 -S -80"
echo "logs: $ROOT_DIR/graph/logs"
echo "monitor_log: $MONITOR_LOG"
echo "status: bash $ROOT_DIR/graph/routes/scripts/status_governance_kg.sh"
