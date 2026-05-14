#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../" && pwd)"
LOG_DIR="$ROOT_DIR/graph/logs"
STATUS_DIR="$LOG_DIR/status"
OUTPUTS_DIR="$ROOT_DIR/graph/outputs"
mkdir -p "$LOG_DIR" "$STATUS_DIR" "$OUTPUTS_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/routeD1_fresh_repro_queue_${STAMP}.log"
STATUS_FILE="$STATUS_DIR/routeD1_fresh_repro.status"
LATEST_LOG="$LOG_DIR/routeD1_fresh_repro_queue_latest.log"
LATEST_OUT="$OUTPUTS_DIR/routeD1_fresh_repro_latest"

if [ -n "${ROUTED1_FRESH_OUTPUT_ROOT:-}" ]; then
  OUT_DIR="$ROUTED1_FRESH_OUTPUT_ROOT"
else
  OUT_DIR="$OUTPUTS_DIR/routeD1_fresh_repro_${STAMP}"
fi

SMOKE_ARGS=()
if [ "${ROUTED1_FRESH_SMOKE:-0}" = "1" ]; then
  SMOKE_ARGS=(--smoke_test)
fi

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
    echo "smoke_args=${SMOKE_ARGS[*]-}"
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
  if [ -n "$1" ]; then
    ps -p "$1" -o pid=,stat=,etime=,pcpu=,pmem=,rss=,cmd= 2>/dev/null | sed 's/^ *//' || true
  fi
}

summarize_failure() {
  echo "$(date -Is) recent error lines from $LOG_FILE"
  grep -nE 'Traceback|RuntimeError|NameError|TypeError|ValueError|ImportError|ModuleNotFoundError|Killed|FAILED|failed|No such file|Permission denied|CUDA out of memory|out of memory' "$LOG_FILE" | tail -160 || true
  echo "$(date -Is) last log lines from $LOG_FILE"
  tail -260 "$LOG_FILE" || true
}

on_interrupt() {
  echo "$(date -Is) routeD1 fresh repro queue interrupted pid=${run_pid:-none}"
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
echo "semantic_contract=fresh_d1_train; no historical D1 outputs used as training inputs"
write_status "WAITING"

while pgrep -f "python3 .*run_routeD1_fresh_repro.py" >/dev/null; do
  echo "$(date -Is) waiting for existing routeD1 fresh repro process..."
  write_status "WAITING" "reason=existing_routeD1_fresh_repro_process"
  sleep 60
done

echo "$(date -Is) starting routeD1 fresh repro"
cd "$ROOT_DIR" || exit 1
export PYTHONUNBUFFERED=1
write_status "RUNNING" "pid=pending"

python3 -u graph/routes/routeD1_fresh_repro/scripts/run_routeD1_fresh_repro.py --output_root "$OUT_DIR" "${SMOKE_ARGS[@]}" >>"$LOG_FILE" 2>&1 &
run_pid=$!
echo "$(date -Is) routeD1 fresh repro child pid=$run_pid"
write_status "RUNNING" "pid=$run_pid"

while kill -0 "$run_pid" 2>/dev/null; do
  sleep 120
  if kill -0 "$run_pid" 2>/dev/null; then
    proc="$(process_line "$run_pid")"
    latest="$(latest_file)"
    echo "$(date -Is) heartbeat routeD1-fresh pid=$run_pid ${proc:-process_not_listed}"
    if [ -n "$latest" ]; then
      echo "$(date -Is) latest_output=$latest"
    fi
    write_status "RUNNING" "pid=$run_pid" "process=$proc" "latest_output=$latest"
  fi
done

wait "$run_pid"
run_status=$?

if [ "$run_status" -eq 0 ]; then
  echo "$(date -Is) routeD1 fresh repro finished status=0 out=$OUT_DIR"
  write_status "COMPLETED" "exit_status=0"
else
  echo "$(date -Is) routeD1 fresh repro finished status=$run_status out=$OUT_DIR"
  write_status "FAILED" "exit_status=$run_status"
  summarize_failure
fi

exit "$run_status"
