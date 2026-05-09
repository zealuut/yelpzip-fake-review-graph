#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

TS="$(date +%Y%m%d_%H%M%S)"
OUTDIR="graph/outputs/routeL_phase1_dryrun_${TS}"

python3 -m graph.routeL_llmmask_anomaly_aux.src.dryrun \
  --configs_dir "graph/routeL_llmmask_anomaly_aux/configs" \
  --output_dir "$OUTDIR" \
  --batch_size 4 \
  --max_seq_length 256

