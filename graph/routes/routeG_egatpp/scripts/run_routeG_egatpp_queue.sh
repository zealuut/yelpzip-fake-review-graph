#!/usr/bin/env bash
set -uo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../" && pwd)"
LOG_DIR="$ROOT_DIR/graph/logs"
STATUS_DIR="$LOG_DIR/status"
OUTPUTS_DIR="$ROOT_DIR/graph/outputs"
mkdir -p "$LOG_DIR" "$STATUS_DIR" "$OUTPUTS_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/routeG_egatpp_queue_${STAMP}.log"
OUT_DIR="$ROOT_DIR/graph/outputs/routeG_egatpp_${STAMP}"
STATUS_FILE="$STATUS_DIR/routeG_egatpp.status"
LATEST_LOG="$LOG_DIR/routeG_egatpp_queue_latest.log"
LATEST_OUT="$OUTPUTS_DIR/routeG_egatpp_latest"

ln -sfn "$LOG_FILE" "$LATEST_LOG"
ln -sfn "$OUT_DIR" "$LATEST_OUT"

run_pid=""

write_status() {
  local state="$1"
  shift || true
  {
    echo "state=$state"
    echo "updated_at=$(date -Is)"
    echo "root=$ROOT_DIR"
    echo "branch=$(git -C "$ROOT_DIR" branch --show-current 2>/dev/null || true)"
    echo "commit=$(git -C "$ROOT_DIR" rev-parse --short HEAD 2>/dev/null || true)"
    echo "log=$LOG_FILE"
    echo "out=$OUT_DIR"
    for item in "$@"; do
      echo "$item"
    done
  } > "$STATUS_FILE.tmp"
  mv "$STATUS_FILE.tmp" "$STATUS_FILE"
}

latest_file() {
  if [ -d "$OUT_DIR" ]; then
    find "$OUT_DIR" -type f -printf '%TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' 2>/dev/null | sort | tail -1
  fi
}

process_line() {
  local pid="$1"
  ps -p "$pid" -o pid=,stat=,etime=,pcpu=,pmem=,rss=,cmd= 2>/dev/null | sed 's/^ *//' || true
}

summarize_failure() {
  echo "$(date -Is) recent error lines from $LOG_FILE"
  grep -nE 'Traceback|RuntimeError|NameError|TypeError|ValueError|ImportError|ModuleNotFoundError|Killed|FAILED|failed|No such file|Permission denied' "$LOG_FILE" | tail -60 || true
  echo "$(date -Is) last log lines from $LOG_FILE"
  tail -120 "$LOG_FILE" || true
}

on_interrupt() {
  echo "$(date -Is) route G queue interrupted pid=${run_pid:-none}"
  if [ -n "${run_pid:-}" ] && kill -0 "$run_pid" 2>/dev/null; then
    kill "$run_pid" 2>/dev/null || true
  fi
  write_status "INTERRUPTED" "pid=${run_pid:-}"
  exit 130
}

trap on_interrupt INT TERM
exec > >(tee -a "$LOG_FILE") 2>&1

echo "queue started at=$(date -Is)"
echo "root=$ROOT_DIR"
echo "branch=$(git -C "$ROOT_DIR" branch --show-current 2>/dev/null || true)"
echo "commit=$(git -C "$ROOT_DIR" rev-parse --short HEAD 2>/dev/null || true)"
echo "log=$LOG_FILE"
echo "out=$OUT_DIR"
echo "status_file=$STATUS_FILE"
write_status "WAITING"

while pgrep -f "python3 .*run_routeD1_kattach_k2k4_ce6a9d6.py" >/dev/null || pgrep -f "python3 .*graph/routes/routeK_topk/scripts/run_routeD1_kattach_candidate_pool.py" >/dev/null; do
  echo "$(date -Is) waiting for K lines..."
  write_status "WAITING" "reason=K_process_active"
  sleep 30
done

echo "$(date -Is) starting route G"
cd "$ROOT_DIR"
export PYTHONUNBUFFERED=1
write_status "RUNNING" "pid=pending"

python3 -u graph/routes/routeG_egatpp/scripts/run_routeG_egatpp.py --output_root "$OUT_DIR" >>"$LOG_FILE" 2>&1 &
run_pid=$!
echo "$(date -Is) route G child pid=$run_pid"
write_status "RUNNING" "pid=$run_pid"

while kill -0 "$run_pid" 2>/dev/null; do
  sleep 60
  if kill -0 "$run_pid" 2>/dev/null; then
    proc="$(process_line "$run_pid")"
    latest="$(latest_file)"
    echo "$(date -Is) heartbeat G pid=$run_pid ${proc:-process_not_listed}"
    if [ -n "$latest" ]; then
      echo "$(date -Is) latest_output=$latest"
    fi
    write_status "RUNNING" "pid=$run_pid" "process=$proc" "latest_output=$latest"
  fi
done

wait "$run_pid"
run_status=$?

if [ "$run_status" -eq 0 ]; then
  echo "$(date -Is) route G finished status=0 out=$OUT_DIR"
  write_status "COMPLETED" "exit_status=0"
else
  echo "$(date -Is) route G finished status=$run_status out=$OUT_DIR"
  write_status "FAILED" "exit_status=$run_status"
  summarize_failure
fi

exit "$run_status"
