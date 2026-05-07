#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/graph/outputs/routeK_adaptive_topk_${TIMESTAMP}}"
LOG_DIR="${ROOT_DIR}/graph/logs"
LOG_FILE="${LOG_DIR}/routeK_adaptive_topk_${TIMESTAMP}.log"
DETACH="${DETACH:-1}"

mkdir -p "${LOG_DIR}"
source "${ROOT_DIR}/graph/run_all.env.sh"

RUNNER_CMD="${PYTHON_BIN:-python3} -m graph.scripts.routek_runner \
  --output_root '${OUTPUT_DIR}' \
  --seed 42 \
  --abnormal_score_source auto \
  --bandit_warmup_epochs 15 \
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
    echo \"[\$(date '+%F %T')] GPU idle, starting Route K\"
    ${RUNNER_CMD}
    echo \"[\$(date '+%F %T')] Route K finished\"
  " </dev/null >"${LOG_FILE}" 2>&1 &
  echo "PID=$!"
  echo "OUTPUT_DIR=${OUTPUT_DIR}"
  echo "LOG_FILE=${LOG_FILE}"
else
  cd "${ROOT_DIR}"
  export PYTHONUNBUFFERED=1
  bash -lc "
    set -euo pipefail
    echo \"[\$(date '+%F %T')] waiting for GPU compute idle\"
    ${WAIT_GPU_CMD}
    echo \"[\$(date '+%F %T')] GPU idle, starting Route K\"
    ${RUNNER_CMD}
  " 2>&1 | tee "${LOG_FILE}"
fi
