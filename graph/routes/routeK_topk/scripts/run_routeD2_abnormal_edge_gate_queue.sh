#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../" && pwd)"
LOG_DIR="$ROOT_DIR/graph/logs"
STATUS_DIR="$LOG_DIR/status"
OUTPUTS_DIR="$ROOT_DIR/graph/outputs"
mkdir -p "$LOG_DIR" "$STATUS_DIR" "$OUTPUTS_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/routeD2_abnormal_edge_gate_queue_${STAMP}.log"
STATUS_FILE="$STATUS_DIR/routeD2_abnormal_edge_gate.status"
LATEST_LOG="$LOG_DIR/routeD2_abnormal_edge_gate_queue_latest.log"
LATEST_OUT="$OUTPUTS_DIR/routeD2_abnormal_edge_gate_latest"
RESUME_ARGS=()

if [ -n "${D2_EDGE_GATE_OUTPUT_ROOT:-}" ]; then
  OUT_DIR="$D2_EDGE_GATE_OUTPUT_ROOT"
  RESUME_ARGS=(--resume)
elif [ "${D2_EDGE_GATE_RESUME_LATEST:-1}" = "1" ]; then
  CANDIDATE_OUT="$(find "$OUTPUTS_DIR" -maxdepth 1 -type d -name 'routeD2_abnormal_edge_gate_20*' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2- || true)"
  if [ -n "$CANDIDATE_OUT" ] && [ ! -f "$CANDIDATE_OUT/routeD2_abnormal_edge_gate_summary.csv" ]; then
    OUT_DIR="$CANDIDATE_OUT"
    RESUME_ARGS=(--resume)
  else
    OUT_DIR="$OUTPUTS_DIR/routeD2_abnormal_edge_gate_${STAMP}"
  fi
else
  OUT_DIR="$OUTPUTS_DIR/routeD2_abnormal_edge_gate_${STAMP}"
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
    echo "resume_args=${RESUME_ARGS[*]-}"
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
  grep -nE 'Traceback|RuntimeError|NameError|TypeError|ValueError|ImportError|ModuleNotFoundError|Killed|FAILED|failed|No such file|Permission denied|CUDA out of memory|out of memory' "$LOG_FILE" | tail -100 || true
  echo "$(date -Is) last log lines from $LOG_FILE"
  tail -180 "$LOG_FILE" || true
}

on_interrupt() {
  echo "$(date -Is) d2 abnormal-edge-gate queue interrupted pid=${run_pid:-none}"
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
echo "resume_args=${RESUME_ARGS[*]-}"
echo "status_file=$STATUS_FILE"
write_status "WAITING"

while pgrep -f "python3 .*run_routeD2_abnormal_edge_gate.py" >/dev/null; do
  echo "$(date -Is) waiting for existing d2 abnormal-edge-gate process..."
  write_status "WAITING" "reason=existing_d2_abnormal_edge_gate_process"
  sleep 30
done

echo "$(date -Is) starting d2 abnormal-edge-gate line"
cd "$ROOT_DIR"
export PYTHONUNBUFFERED=1
write_status "RUNNING" "pid=pending"

python3 -u graph/routes/routeK_topk/scripts/run_routeD2_abnormal_edge_gate.py --output_root "$OUT_DIR" "${RESUME_ARGS[@]}" >>"$LOG_FILE" 2>&1 &
run_pid=$!
echo "$(date -Is) d2 abnormal-edge-gate child pid=$run_pid"
write_status "RUNNING" "pid=$run_pid"

while kill -0 "$run_pid" 2>/dev/null; do
  sleep 60
  if kill -0 "$run_pid" 2>/dev/null; then
    proc="$(process_line "$run_pid")"
    latest="$(latest_file)"
    echo "$(date -Is) heartbeat D2-edge-gate pid=$run_pid ${proc:-process_not_listed}"
    if [ -n "$latest" ]; then
      echo "$(date -Is) latest_output=$latest"
    fi
    write_status "RUNNING" "pid=$run_pid" "process=$proc" "latest_output=$latest"
  fi
done

wait "$run_pid"
run_status=$?

if [ "$run_status" -eq 0 ]; then
  echo "$(date -Is) d2 abnormal-edge-gate line finished status=0 out=$OUT_DIR"
  write_status "COMPLETED" "exit_status=0"
else
  echo "$(date -Is) d2 abnormal-edge-gate line finished status=$run_status out=$OUT_DIR"
  write_status "FAILED" "exit_status=$run_status"
  summarize_failure
fi

exit "$run_status"
