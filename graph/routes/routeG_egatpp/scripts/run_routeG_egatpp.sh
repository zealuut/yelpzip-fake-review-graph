#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../" && pwd)"
cd "$ROOT_DIR"
OUTPUT_ROOT="${1:-$ROOT_DIR/graph/outputs/routeG_egatpp_$(date +%Y%m%d_%H%M%S)}"
python3 graph/routes/routeG_egatpp/scripts/run_routeG_egatpp.py --output_root "$OUTPUT_ROOT"
