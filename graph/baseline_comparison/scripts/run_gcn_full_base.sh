#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT is required}"

cd "${ROOT_DIR}"
python3 -m graph.baseline_comparison.src.train_baseline \
  --config "${ROOT_DIR}/graph/baseline_comparison/configs/gcn_full_base.yaml" \
  --output-root "${OUTPUT_ROOT}"
