#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
python3 -m graph.baseline_comparison.src.train_baseline \
  --config "${ROOT_DIR}/graph/baseline_comparison/configs/graphsage_current_topk.yaml" \
  --output-root "${OUTPUT_ROOT}"
