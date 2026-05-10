#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

STAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="$ROOT_DIR/graph/outputs/routeL_textheads_phase2_${STAMP}"

python3 -m graph.routeL_llmmask_anomaly_aux.scripts.run_routeL_textheads_phase2 \
  --output_root "$OUTPUT_ROOT"
