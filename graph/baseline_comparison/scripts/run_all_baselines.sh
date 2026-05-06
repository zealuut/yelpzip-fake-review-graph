#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/graph/baseline_comparison/outputs/${TIMESTAMP}}"
LOG_DIR="${ROOT_DIR}/graph/baseline_comparison/logs"
LOG_FILE="${LOG_DIR}/run_all_baselines_${TIMESTAMP}.log"
DETACH="${DETACH:-1}"

mkdir -p "${LOG_DIR}"

RUN_CMD="
  set -euo pipefail
  cd '${ROOT_DIR}'
  echo \"[\$(date '+%F %T')] waiting for GPU compute idle\"
  while nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | rg -q '^[0-9]'; do sleep 20; done
  echo \"[\$(date '+%F %T')] GPU idle, starting baseline comparison\"
  export OUTPUT_ROOT='${OUTPUT_ROOT}'
  bash '${ROOT_DIR}/graph/baseline_comparison/scripts/run_gat_current_topk.sh'
  bash '${ROOT_DIR}/graph/baseline_comparison/scripts/run_graphsage_current_topk.sh'
  bash '${ROOT_DIR}/graph/baseline_comparison/scripts/run_rgcn_current_topk.sh'
  python3 -m graph.baseline_comparison.scripts.summarize_baselines --output-root '${OUTPUT_ROOT}'
  echo \"[\$(date '+%F %T')] baseline comparison finished\"
"

if [[ "${DETACH}" == "1" ]]; then
  nohup setsid bash -lc "${RUN_CMD}" </dev/null >"${LOG_FILE}" 2>&1 &
  echo "PID=$!"
  echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
  echo "LOG_FILE=${LOG_FILE}"
else
  bash -lc "${RUN_CMD}" 2>&1 | tee "${LOG_FILE}"
fi
