#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/graph/outputs/routeC_cb_only_abnormal_weight_v2_${TIMESTAMP}}"
LOG_DIR="${ROOT_DIR}/graph/logs"
LOG_FILE="${LOG_DIR}/routeC_cb_only_abnormal_weight_v2_${TIMESTAMP}.log"
DETACH="${DETACH:-1}"

mkdir -p "${LOG_DIR}"
source "${ROOT_DIR}/graph/run_all.env.sh"

if [[ "${DETACH}" == "1" ]]; then
  nohup setsid bash -lc "
    cd '${ROOT_DIR}' &&
    export PYTHONUNBUFFERED=1 &&
    '${PYTHON_BIN:-python3}' -m graph.scripts.route_runner \
      --route C \
      --output_root '${OUTPUT_DIR}' \
      --abnormal_edge_eta '0.5' \
      --abnormal_gate_eta '0.5' \
      --abnormal_pair_mode 'both_high' \
      --abnormal_score_source 'auto'
  " </dev/null >"${LOG_FILE}" 2>&1 &
  echo "PID=$!"
  echo "OUTPUT_DIR=${OUTPUT_DIR}"
  echo "LOG_FILE=${LOG_FILE}"
else
  cd "${ROOT_DIR}"
  export PYTHONUNBUFFERED=1
  exec "${PYTHON_BIN:-python3}" -m graph.scripts.route_runner \
    --route C \
    --output_root "${OUTPUT_DIR}" \
    --abnormal_edge_eta "0.5" \
    --abnormal_gate_eta "0.5" \
    --abnormal_pair_mode "both_high" \
    --abnormal_score_source "auto" 2>&1 | tee "${LOG_FILE}"
fi
