#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LOG_DIR="$ROOT_DIR/graph/logs"
STATUS_DIR="$LOG_DIR/status"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="$LOG_DIR/governance_kg_queue_${STAMP}.log"
STATUS_FILE="$STATUS_DIR/governance_kg.status"

mkdir -p "$LOG_DIR" "$STATUS_DIR"
ln -sfn "$RUN_LOG" "$LOG_DIR/governance_kg_queue_latest.log"

stage="init"

write_status() {
  local state="$1"
  shift || true
  {
    echo "state=$state"
    echo "updated_at=$(date -Is)"
    echo "root=$ROOT_DIR"
    echo "branch=$(git -C "$ROOT_DIR" branch --show-current 2>/dev/null || true)"
    echo "commit=$(git -C "$ROOT_DIR" rev-parse --short HEAD 2>/dev/null || true)"
    echo "run_log=$RUN_LOG"
    echo "stage=$stage"
    for item in "$@"; do
      echo "$item"
    done
  } > "$STATUS_FILE.tmp"
  mv "$STATUS_FILE.tmp" "$STATUS_FILE"
}

on_interrupt() {
  echo "$(date -Is) governance KG runner interrupted stage=$stage"
  write_status "INTERRUPTED"
  exit 130
}

trap on_interrupt INT TERM

exec > >(tee -a "$RUN_LOG") 2>&1

echo "governance KG runner started at=$(date -Is)"
echo "root=$ROOT_DIR"
echo "branch=$(git -C "$ROOT_DIR" branch --show-current 2>/dev/null || true)"
echo "commit=$(git -C "$ROOT_DIR" rev-parse --short HEAD 2>/dev/null || true)"
echo "run_log=$RUN_LOG"
echo "status_file=$STATUS_FILE"
echo "status_command=bash $ROOT_DIR/graph/routes/scripts/status_governance_kg.sh"

cd "$ROOT_DIR"
write_status "RUNNING"

stage="K"
echo "$(date -Is) launching K candidate-pool queue"
write_status "RUNNING" "active_queue=K"
bash graph/routes/routeK_topk/scripts/run_routeD1_kattach_candidate_pool_queue.sh
k_status=$?
echo "$(date -Is) K candidate-pool queue exited status=$k_status"

if [ "$k_status" -ne 0 ]; then
  echo "$(date -Is) stopping before G because K failed"
  write_status "FAILED" "failed_queue=K" "exit_status=$k_status"
  bash graph/routes/scripts/status_governance_kg.sh || true
  exit "$k_status"
fi

stage="G"
echo "$(date -Is) launching G EGAT++ queue"
write_status "RUNNING" "active_queue=G"
bash graph/routes/routeG_egatpp/scripts/run_routeG_egatpp_queue.sh
g_status=$?
echo "$(date -Is) G EGAT++ queue exited status=$g_status"

if [ "$g_status" -ne 0 ]; then
  echo "$(date -Is) governance KG runner failed status=$g_status"
  write_status "FAILED" "failed_queue=G" "exit_status=$g_status"
  bash graph/routes/scripts/status_governance_kg.sh || true
  exit "$g_status"
fi

stage="done"
echo "$(date -Is) governance KG runner completed successfully"
write_status "COMPLETED" "exit_status=0"
