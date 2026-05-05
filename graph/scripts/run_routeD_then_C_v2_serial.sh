#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${ROOT_DIR}/graph/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
QUEUE_LOG="${LOG_DIR}/routeD_then_C_v2_serial_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}"

nohup setsid bash -lc "
  cd '${ROOT_DIR}' || exit 1
  echo \"[$(date '+%F %T')] launching Route D v2\"
  D_OUT=\"${ROOT_DIR}/graph/outputs/routeD_tns_confirmed_logic_egat_v2_${TIMESTAMP}\"
  D_LOG=\"${LOG_DIR}/routeD_tns_confirmed_logic_egat_v2_${TIMESTAMP}.log\"
  DETACH=1 OUTPUT_DIR=\"\$D_OUT\" bash graph/scripts/run_routeD_tns_confirmed_logic_egat_v2.sh > /tmp/route_d_v2_launch_${TIMESTAMP}.txt
  cat /tmp/route_d_v2_launch_${TIMESTAMP}.txt
  D_PID=\$(awk -F= '/^PID=/{print \$2}' /tmp/route_d_v2_launch_${TIMESTAMP}.txt | tail -n 1)
  echo \"[$(date '+%F %T')] waiting for Route D v2 pid \$D_PID\"
  while kill -0 \"\$D_PID\" 2>/dev/null; do sleep 30; done
  echo \"[$(date '+%F %T')] Route D v2 finished; launching Route C v2\"
  C_OUT=\"${ROOT_DIR}/graph/outputs/routeC_cb_only_abnormal_weight_v2_${TIMESTAMP}\"
  DETACH=1 OUTPUT_DIR=\"\$C_OUT\" bash graph/scripts/run_routeC_cb_only_abnormal_weight_v2.sh
  echo \"[$(date '+%F %T')] Route C v2 launched\"
" </dev/null >"${QUEUE_LOG}" 2>&1 &

echo "QUEUE_PID=$!"
echo "QUEUE_LOG=${QUEUE_LOG}"
