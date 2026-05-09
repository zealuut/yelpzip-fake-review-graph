#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

TS="$(date +%Y%m%d_%H%M%S)"
OUTDIR="graph/outputs/routeL_llmmask_anomaly_aux_${TS}"

python3 -m graph.routeL_llmmask_anomaly_aux.scripts.run_routeL_phase2 \
  --output_root "$OUTDIR" \
  --seed 42
