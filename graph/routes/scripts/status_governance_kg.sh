#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LOG_DIR="$ROOT_DIR/graph/logs"
STATUS_DIR="$LOG_DIR/status"
OUTPUTS_DIR="$ROOT_DIR/graph/outputs"
BRIEF=0
if [ "${1:-}" = "--brief" ]; then
  BRIEF=1
fi

latest_match() {
  local dir="$1"
  local name="$2"
  if [ -d "$dir" ]; then
    find "$dir" -maxdepth 1 -name "$name" -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-
  fi
}

print_status_file() {
  local label="$1"
  local file="$2"
  echo "[$label]"
  if [ -f "$file" ]; then
    sed -n '1,80p' "$file"
  else
    echo "status_file_missing=$file"
  fi
}

print_recent_errors() {
  local label="$1"
  local file="$2"
  echo "[$label errors]"
  if [ -n "$file" ] && [ -f "$file" ]; then
    grep -nE 'Traceback|RuntimeError|NameError|TypeError|ValueError|ImportError|ModuleNotFoundError|Killed|FAILED|failed|finished status=[1-9]|No such file|Permission denied' "$file" | tail -40 || true
  else
    echo "log_missing"
  fi
}

print_tail() {
  local label="$1"
  local file="$2"
  local lines="${3:-30}"
  echo "[$label tail]"
  if [ -n "$file" ] && [ -f "$file" ]; then
    tail -c 20000 "$file" 2>/dev/null | tr '\r' '\n' | tail -n "$lines" || true
  else
    echo "log_missing"
  fi
}

print_latest_outputs() {
  local label="$1"
  local dir="$2"
  local lines="${3:-20}"
  echo "[$label latest outputs]"
  if [ -d "$dir" ]; then
    find "$dir" -type f -printf '%TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' 2>/dev/null | sort | tail -n "$lines"
  else
    echo "output_dir_missing=${dir:-none}"
  fi
}

print_metric_hint() {
  local label="$1"
  local dir="$2"
  echo "[$label summaries]"
  if [ ! -d "$dir" ]; then
    echo "output_dir_missing=${dir:-none}"
    return 0
  fi
  find "$dir" -maxdepth 4 -type f \( -name 'run_summary.json' -o -name 'model_results.csv' \) -printf '%TY-%Tm-%TdT%TH:%TM:%TS %p\n' 2>/dev/null | sort | tail -20
}

runner_log="$(latest_match "$LOG_DIR" 'governance_kg_queue_20*.log')"
k_log="$(latest_match "$LOG_DIR" 'routeD1_kattach_candidate_pool_queue_20*.log')"
g_log="$(latest_match "$LOG_DIR" 'routeG_egatpp_queue_20*.log')"
k_out="$(latest_match "$OUTPUTS_DIR" 'routeD1_kattach_candidate_pool_20*')"
g_out="$(latest_match "$OUTPUTS_DIR" 'routeG_egatpp_20*')"

echo "===== governance KG status $(date -Is) ====="
echo "root=$ROOT_DIR"
echo "branch=$(git -C "$ROOT_DIR" branch --show-current 2>/dev/null || true)"
echo "commit=$(git -C "$ROOT_DIR" rev-parse --short HEAD 2>/dev/null || true)"
echo "runner_log=$runner_log"
echo "k_log=$k_log"
echo "g_log=$g_log"
echo "k_out=$k_out"
echo "g_out=$g_out"
echo

echo "[tmux]"
tmux ls 2>/dev/null | grep -E '^governance_kg:' || echo "governance_kg_session=missing"
tmux list-windows -t governance_kg 2>/dev/null || true
echo

echo "[processes]"
ps -eo pid,ppid,stat,etime,pcpu,pmem,rss,args | grep -E 'run_governance_kg_queue|run_routeD1_kattach_candidate_pool|run_routeG_egatpp|python3 -u' | grep -v grep || echo "no_matching_processes"
echo

echo "[gpu]"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true
echo

print_status_file "runner status" "$STATUS_DIR/governance_kg.status"
echo
print_status_file "K status" "$STATUS_DIR/routeK_candidate_pool.status"
echo
print_status_file "G status" "$STATUS_DIR/routeG_egatpp.status"
echo

print_metric_hint "K" "$k_out"
echo
if [ "$BRIEF" -eq 1 ]; then
  print_latest_outputs "K" "$k_out" 8
else
  print_latest_outputs "K" "$k_out" 20
fi
echo
print_metric_hint "G" "$g_out"
echo
if [ "$BRIEF" -eq 1 ]; then
  print_latest_outputs "G" "$g_out" 8
else
  print_latest_outputs "G" "$g_out" 20
fi
echo

print_recent_errors "runner" "$runner_log"
echo
print_recent_errors "K" "$k_log"
echo
print_recent_errors "G" "$g_log"
echo

if [ "$BRIEF" -eq 0 ]; then
  print_tail "runner" "$runner_log" 30
  echo
  print_tail "K" "$k_log" 40
  echo
  print_tail "G" "$g_log" 40
fi

echo "===== end status ====="
