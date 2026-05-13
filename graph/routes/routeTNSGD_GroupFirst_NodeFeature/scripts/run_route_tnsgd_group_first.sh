#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../../.. && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$ROOT_DIR/graph/outputs/routeTNSGD_GroupFirst_NodeFeature_${TS}"
LOG_DIR="$ROOT_DIR/graph/logs"
LOG_FILE="$LOG_DIR/routeTNSGD_GroupFirst_NodeFeature_${TS}.log"

mkdir -p "$OUT_DIR" "$LOG_DIR"

PYTHONUNBUFFERED=1 python3 -u "$ROOT_DIR/graph/routes/routeTNSGD_GroupFirst_NodeFeature/scripts/run_route_tnsgd_group_first.py" \
  --config "$ROOT_DIR/graph/routes/routeTNSGD_GroupFirst_NodeFeature/configs/tnsgd_group_first_phi5.yaml" \
  --output_root "$OUT_DIR" 2>&1 | tee "$LOG_FILE"

echo "output_root=$OUT_DIR"
echo "log=$LOG_FILE"
