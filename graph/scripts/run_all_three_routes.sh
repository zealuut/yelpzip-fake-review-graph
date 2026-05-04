#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "${ROOT_DIR}/graph/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="${ROOT_DIR}/graph/logs/run_all_three_routes_${TIMESTAMP}.log"

run_serial() {
  local script_path="$1"
  DETACH=0 bash "${script_path}"
}

if [[ "${DETACH:-1}" == "1" ]]; then
  nohup setsid bash -lc "
    cd '${ROOT_DIR}' &&
    export DETACH=0 &&
    bash '${ROOT_DIR}/graph/scripts/run_all_three_routes.sh'
  " </dev/null >"${MASTER_LOG}" 2>&1 &
  echo "PID=$!"
  echo "MASTER_LOG=${MASTER_LOG}"
else
  run_serial "${ROOT_DIR}/graph/scripts/run_routeA_current_topk_egat.sh"
  run_serial "${ROOT_DIR}/graph/scripts/run_routeB_senior_exact_plus_logic_edges.sh"
  run_serial "${ROOT_DIR}/graph/scripts/run_routeC_abnormal_weight_gate.sh"
  echo "All three routes completed in serial."
fi
