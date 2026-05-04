#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/graph/outputs/routeA_current_topk_egat_${TIMESTAMP}}"
LOG_DIR="${ROOT_DIR}/graph/logs"
LOG_FILE="${LOG_DIR}/routeA_current_topk_egat_${TIMESTAMP}.log"
DETACH="${DETACH:-1}"

mkdir -p "${LOG_DIR}"
source "${ROOT_DIR}/graph/run_all.env.sh"

if [[ "${DETACH}" == "1" ]]; then
  nohup setsid bash -lc "
    cd '${ROOT_DIR}' &&
    export PYTHONUNBUFFERED=1 &&
    '${PYTHON_BIN:-python3}' -m graph.scripts.route_runner \
      --route A \
      --output_root '${OUTPUT_DIR}'
  " </dev/null >"${LOG_FILE}" 2>&1 &
  echo "PID=$!"
  echo "OUTPUT_DIR=${OUTPUT_DIR}"
  echo "LOG_FILE=${LOG_FILE}"
else
  cd "${ROOT_DIR}"
  export PYTHONUNBUFFERED=1
  exec "${PYTHON_BIN:-python3}" -m graph.scripts.route_runner \
    --route A \
    --output_root "${OUTPUT_DIR}" 2>&1 | tee "${LOG_FILE}"
fi
