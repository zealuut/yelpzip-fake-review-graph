#!/usr/bin/env bash
set -uo pipefail

SCRIPT_PATH="$(readlink -f "$0")"
ROOT_DIR="$(cd "$(dirname "$SCRIPT_PATH")/../../../.." && pwd)"

if [[ "${ROUTEV_TMUX_WORKER:-0}" != "1" ]]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  RUN_ID="${1:-routeV_vector_quality_strict_${TS}}"
  SESSION="${ROUTEV_TMUX_SESSION:-routeV_strict_${TS}}"
  WORKER_CMD="ROUTEV_TMUX_WORKER=1 bash $(printf '%q' "$SCRIPT_PATH") $(printf '%q' "$RUN_ID")"
  tmux new-session -d -s "$SESSION" "$WORKER_CMD"
  echo "SESSION=$SESSION"
  echo "RUN_ID=$RUN_ID"
  echo "OUTPUT=$ROOT_DIR/graph/outputs/$RUN_ID"
  echo "LOG=$ROOT_DIR/graph/logs/$RUN_ID.log"
  echo "STATUS=$ROOT_DIR/graph/logs/status/$RUN_ID.json"
  exit 0
fi

RUN_ID="${1:?RUN_ID is required in worker mode}"
OUTPUT_DIR="$ROOT_DIR/graph/outputs/$RUN_ID"
LOG_FILE="$ROOT_DIR/graph/logs/$RUN_ID.log"
STATUS_FILE="$ROOT_DIR/graph/logs/status/$RUN_ID.json"

mkdir -p "$OUTPUT_DIR" "$(dirname "$LOG_FILE")" "$(dirname "$STATUS_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1

write_status() {
  local status="$1"
  local exit_code="${2:-}"
  python3 - "$STATUS_FILE" "$status" "$exit_code" "$RUN_ID" "$ROOT_DIR" "$OUTPUT_DIR" "$LOG_FILE" <<'PY'
import datetime
import json
import subprocess
import sys
from pathlib import Path

status_file, status, exit_code, run_id, root_dir, output_dir, log_file = sys.argv[1:]
root = Path(root_dir)
output = Path(output_dir)

try:
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=root, text=True).strip()
except Exception:
    branch = ""
try:
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=root, text=True).strip()
except Exception:
    commit = ""

payload = {
    "status": status,
    "run_id": run_id,
    "root": root_dir,
    "branch": branch,
    "commit": commit,
    "output_dir": output_dir,
    "log_file": log_file,
    "status_file": status_file,
    "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    "queue": ["V_control", "V0_proxy_checkpoint", "V1a_supcon_reg", "V1b_triplet_reg", "V2_dual_head"],
    "control_gate": "enabled",
    "latest_output_files": sorted(str(path) for path in output.glob("**/run_summary.json"))[-12:],
}
if exit_code:
    payload["exit_code"] = int(exit_code)
Path(status_file).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
}

cd "$ROOT_DIR" || exit 1
echo "========================================================================"
echo "RouteV strict full queue"
echo "Started: $(date -Is)"
echo "Root: $ROOT_DIR"
echo "Branch: $(git branch --show-current)"
echo "Commit: $(git rev-parse --short HEAD)"
echo "Output: $OUTPUT_DIR"
echo "Log: $LOG_FILE"
echo "Status: $STATUS_FILE"
echo "========================================================================"

write_status "starting"

python3 -u -m graph.routes.routeV_vector_quality.scripts.run_routeV_queue \
  --output_root "$OUTPUT_DIR" &
RUNNER_PID=$!
echo "RouteV runner PID: $RUNNER_PID"

while kill -0 "$RUNNER_PID" 2>/dev/null; do
  write_status "running"
  sleep 300
done

wait "$RUNNER_PID"
RC=$?
if [[ "$RC" -eq 0 ]]; then
  write_status "complete" "$RC"
else
  write_status "failed" "$RC"
  echo "Recent error lines:"
  grep -Ei "traceback|error|exception|out of memory|failed|SystemExit" "$LOG_FILE" | tail -n 80 || true
fi
echo "Finished with exit code $RC at $(date -Is)"
exit "$RC"
