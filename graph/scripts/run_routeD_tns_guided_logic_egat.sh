#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/graph/outputs/routeD_tns_guided_logic_egat_${TIMESTAMP}}"
LOG_DIR="${ROOT_DIR}/graph/logs"
LOG_FILE="${LOG_DIR}/routeD_tns_guided_logic_egat_${TIMESTAMP}.log"
DETACH="${DETACH:-1}"

mkdir -p "${LOG_DIR}"
source "${ROOT_DIR}/graph/run_all.env.sh"

if [[ "${DETACH}" == "1" ]]; then
  nohup setsid bash -lc "
    cd '${ROOT_DIR}' &&
    export PYTHONUNBUFFERED=1 &&
    '${PYTHON_BIN:-python3}' -m graph.scripts.route_runner \
      --route D \
      --output_root '${OUTPUT_DIR}' \
      --tns_phi_days '${TNS_PHI_DAYS:-5}' \
      --tns_logic_mode '${TNS_LOGIC_MODE:-boost}' \
      --tns_logic_lambda '${TNS_LOGIC_LAMBDA:-1.0}' \
      --logic_tns_topk '${LOGIC_TNS_TOPK:-20}' \
      --abnormal_score_source auto
  " </dev/null >"${LOG_FILE}" 2>&1 &
  echo "PID=$!"
  echo "OUTPUT_DIR=${OUTPUT_DIR}"
  echo "LOG_FILE=${LOG_FILE}"
else
  cd "${ROOT_DIR}"
  export PYTHONUNBUFFERED=1
  exec "${PYTHON_BIN:-python3}" -m graph.scripts.route_runner \
    --route D \
    --output_root "${OUTPUT_DIR}" \
    --tns_phi_days "${TNS_PHI_DAYS:-5}" \
    --tns_logic_mode "${TNS_LOGIC_MODE:-boost}" \
    --tns_logic_lambda "${TNS_LOGIC_LAMBDA:-1.0}" \
    --logic_tns_topk "${LOGIC_TNS_TOPK:-20}" \
    --abnormal_score_source auto 2>&1 | tee "${LOG_FILE}"
fi
