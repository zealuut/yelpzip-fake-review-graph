#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/graph/outputs/routeK_d1main_rns_topk_${TIMESTAMP}}"
LOG_DIR="${ROOT_DIR}/graph/logs"
LOG_FILE="${LOG_DIR}/routeK_d1main_rns_topk_${TIMESTAMP}.log"
PUSH_LOG="${LOG_DIR}/auto_push_routeK_d1main_rns_topk_${TIMESTAMP}.log"
DETACH="${DETACH:-1}"

mkdir -p "${LOG_DIR}"
source "${ROOT_DIR}/graph/run_all.env.sh"

RUNNER_CMD="${PYTHON_BIN:-python3} -m graph.scripts.routek_d1main_rns_runner \
  --output_root '${OUTPUT_DIR}' \
  --seed 42 \
  --abnormal_score_source auto \
  --k4_warmup_epochs 15 \
  --k5_warmup_epochs 20 \
  --bandit_lambda_density 0.02"

WAIT_GPU_CMD="while nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | rg -q '^[0-9]'; do sleep 20; done"

if [[ "${DETACH}" == "1" ]]; then
  nohup setsid bash -lc "
    set -euo pipefail
    cd '${ROOT_DIR}'
    source '${ROOT_DIR}/graph/run_all.env.sh'
    export PYTHONUNBUFFERED=1
    echo \"[\$(date '+%F %T')] waiting for GPU compute idle\"
    ${WAIT_GPU_CMD}
    echo \"[\$(date '+%F %T')] GPU idle, starting Route K D1Main/RNS\"
    ${RUNNER_CMD}
    echo \"[\$(date '+%F %T')] Route K D1Main/RNS finished\"
  " </dev/null >"${LOG_FILE}" 2>&1 &
  RUN_PID=$!

  nohup setsid bash -lc "
    set -euo pipefail
    cd '${ROOT_DIR}'
    echo \"[\$(date '+%F %T')] watching Route K D1Main/RNS pid ${RUN_PID}\"
    while kill -0 ${RUN_PID} 2>/dev/null; do sleep 60; done
    if [[ -f '${OUTPUT_DIR}/routeK_d1main_rns_summary.csv' ]]; then
      echo \"[\$(date '+%F %T')] outputs ready, preparing git commit\"
      git add '${OUTPUT_DIR}' \
        '${ROOT_DIR}/graph/scripts/run_routeK_d1main_rns_topk.sh' \
        '${ROOT_DIR}/graph/scripts/routek_d1main_rns_runner.py' \
        '${ROOT_DIR}/graph/graph_pipeline.py' \
        '${ROOT_DIR}/graph/relation_model.py'
      if ! git diff --cached --quiet; then
        git commit -m 'Add Route K D1Main RNS adaptive top-k results' || true
      fi
      git push origin main || true
      echo \"[\$(date '+%F %T')] push finished\"
    else
      echo \"[\$(date '+%F %T')] Route K D1Main/RNS outputs missing; skipping push\"
    fi
  " </dev/null >"${PUSH_LOG}" 2>&1 &
  WATCH_PID=$!

  echo "RUN_PID=${RUN_PID}"
  echo "WATCH_PID=${WATCH_PID}"
  echo "OUTPUT_DIR=${OUTPUT_DIR}"
  echo "LOG_FILE=${LOG_FILE}"
  echo "PUSH_LOG=${PUSH_LOG}"
else
  cd "${ROOT_DIR}"
  export PYTHONUNBUFFERED=1
  bash -lc "
    set -euo pipefail
    echo \"[\$(date '+%F %T')] waiting for GPU compute idle\"
    ${WAIT_GPU_CMD}
    echo \"[\$(date '+%F %T')] GPU idle, starting Route K D1Main/RNS\"
    ${RUNNER_CMD}
  " 2>&1 | tee "${LOG_FILE}"
fi
