#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../../../ && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$ROOT_DIR/graph/outputs/routeTNS_A_node_profile_${TS}"

python3 "$ROOT_DIR/graph/routes/routeTNS_A_node_profile/scripts/run_route_tns_a.py" \
  --output_root "$OUT_DIR" \
  --config_paths \
    "$ROOT_DIR/graph/routes/routeTNS_A_node_profile/configs/TNSA0_baseline.yaml" \
    "$ROOT_DIR/graph/routes/routeTNS_A_node_profile/configs/TNSA1_basic.yaml"

echo "$OUT_DIR"
