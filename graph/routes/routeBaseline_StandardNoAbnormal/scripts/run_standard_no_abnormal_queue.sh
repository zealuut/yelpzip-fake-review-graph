#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
ROUTE_DIR="$ROOT_DIR/graph/routes/routeBaseline_StandardNoAbnormal"
LOG_DIR="$ROOT_DIR/graph/logs"
STATUS_DIR="$LOG_DIR/status"
OUTPUTS_DIR="$ROOT_DIR/graph/outputs"
mkdir -p "$LOG_DIR" "$STATUS_DIR" "$OUTPUTS_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${STANDARD_NO_ABNORMAL_OUT:-$OUTPUTS_DIR/routeBaseline_StandardNoAbnormal_${STAMP}}"
LOG_FILE="$LOG_DIR/routeBaseline_StandardNoAbnormal_${STAMP}.log"
STATUS_FILE="$STATUS_DIR/routeBaseline_StandardNoAbnormal.status"
LATEST_LOG="$LOG_DIR/routeBaseline_StandardNoAbnormal_latest.log"
LATEST_OUT="$OUTPUTS_DIR/routeBaseline_StandardNoAbnormal_latest"
MODELS=(gat graphsage rgcn)

mkdir -p "$OUT_DIR"
ln -sfn "$LOG_FILE" "$LATEST_LOG"
ln -sfn "$OUT_DIR" "$LATEST_OUT"

write_status() {
  local state="$1"
  shift || true
  {
    echo "state=$state"
    echo "updated_at=$(date -Is)"
    echo "root=$ROOT_DIR"
    echo "route=$ROUTE_DIR"
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

summarize_results() {
  python3 - "$OUT_DIR" <<'PY'
from pathlib import Path
import sys
import pandas as pd

out = Path(sys.argv[1])
rows = []
for path in sorted(out.glob("*/metrics/model_results.csv")):
    rows.append(pd.read_csv(path).iloc[0].to_dict())
if rows:
    df = pd.DataFrame(rows)
    order = [
        "experiment_name",
        "model",
        "feature_dim",
        "num_edges",
        "val_auc",
        "val_ap",
        "auc",
        "ap",
        "f1",
        "precision",
        "recall",
        "best_epoch",
        "threshold",
    ]
    cols = [col for col in order if col in df.columns]
    df[cols].to_csv(out / "standard_no_abnormal_summary.csv", index=False)
    (out / "standard_no_abnormal_summary.md").write_text(df[cols].to_csv(index=False), encoding="utf-8")
PY
}

exec > >(tee -a "$LOG_FILE") 2>&1
echo "queue started at=$(date -Is)"
echo "root=$ROOT_DIR"
echo "route=$ROUTE_DIR"
echo "out=$OUT_DIR"
echo "log=$LOG_FILE"
write_status "RUNNING" "current_model=init"

exit_status=0
for model in "${MODELS[@]}"; do
  echo "$(date -Is) starting model=$model"
  write_status "RUNNING" "current_model=$model"
  if bash "$ROUTE_DIR/models/$model/run.sh" "$OUT_DIR"; then
    echo "$(date -Is) completed model=$model"
  else
    exit_status=$?
    echo "$(date -Is) failed model=$model status=$exit_status"
    write_status "FAILED" "current_model=$model" "exit_status=$exit_status"
    break
  fi
  summarize_results || true
done

summarize_results || true
if [ "$exit_status" -eq 0 ]; then
  write_status "COMPLETED" "exit_status=0"
  echo "$(date -Is) queue completed out=$OUT_DIR"
else
  echo "$(date -Is) queue failed status=$exit_status out=$OUT_DIR"
fi

exit "$exit_status"
