#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

timestamp="$(date +%Y%m%d_%H%M%S)"
output_dir="$ROOT_DIR/graph/outputs/routeD_tns_heavy_logic_${timestamp}"
log_dir="$ROOT_DIR/graph/logs"
mkdir -p "$log_dir"
log_file="$log_dir/routeD_tns_heavy_logic_${timestamp}.log"

wait_for_gpu_idle() {
  while true; do
    local active
    active="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | rg -v '^[[:space:]]*$' || true)"
    if [[ -z "${active}" ]]; then
      return 0
    fi
    sleep 20
  done
}

wait_for_gpu_idle

cmd=(
  python3 -m graph.scripts.route_runner
  --route D_HEAVY
  --output_root "$output_dir"
  --seed 42
  --tns_phi_days "${TNS_PHI_DAYS:-5}"
  --logic_tns_topk "${LOGIC_TNS_TOPK:-20}"
  --tns_heavy_lambda "${TNS_HEAVY_LAMBDA:-0.3}"
  --use_tns_heavy
)

nohup setsid "${cmd[@]}" >"$log_file" 2>&1 </dev/null &
pid=$!

echo "$pid" > "$log_dir/routeD_tns_heavy_logic_${timestamp}.pid"
echo "$output_dir"
