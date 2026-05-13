#!/usr/bin/env bash
set -euo pipefail
MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$MODEL_DIR/../../../../.." && pwd)"
OUTPUT_ROOT="${1:-$ROOT_DIR/graph/outputs/routeBaseline_StandardNoAbnormal_manual_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTPUT_ROOT"
cd "$ROOT_DIR"
python3 -u "$MODEL_DIR/train.py" --config "$MODEL_DIR/config.json" --output-root "$OUTPUT_ROOT"
