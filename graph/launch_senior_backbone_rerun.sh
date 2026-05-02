#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/graph/outputs/yelpzip_senior_backbone_clean_formal_${TIMESTAMP}}"
LOG_DIR="${ROOT_DIR}/graph/logs"
LOG_FILE="${LOG_DIR}/senior_backbone_clean_formal_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}"

source "${ROOT_DIR}/graph/run_all.env.sh"

export SENIOR_PROTOCOL=1
export FAKE_EXTRACTOR_ONLY=1
export RUN_LLM_CACHE=0
export DISABLE_GRAPH_REWEIGHTING=1
export RUN_LEGACY_BASELINES=0
export GRAPH_MODE=senior
export OUTPUT_DIR

# Clean senior backbone ablation: no SKEP branch.
export SECONDARY_MODEL_NAME_OR_PATH=""

echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "LOG_FILE=${LOG_FILE}"
echo "GRAPH_MODE=${GRAPH_MODE}"
echo "DISABLE_GRAPH_REWEIGHTING=${DISABLE_GRAPH_REWEIGHTING}"
echo "SECONDARY_MODEL_NAME_OR_PATH='${SECONDARY_MODEL_NAME_OR_PATH}'"

nohup setsid bash -lc "
  cd '${ROOT_DIR}' &&
  export PYTHON_BIN='${PYTHON_BIN:-python3}' &&
  export PYTHONUNBUFFERED=1 &&
  export TOKENIZERS_PARALLELISM='${TOKENIZERS_PARALLELISM:-false}' &&
  export HF_ENDPOINT='${HF_ENDPOINT:-}' &&
  export HF_HOME='${HF_HOME:-}' &&
  export TRANSFORMERS_CACHE='${TRANSFORMERS_CACHE:-}' &&
  export PRIMARY_MODEL_NAME_OR_PATH='${PRIMARY_MODEL_NAME_OR_PATH:-}' &&
  export LEGACY_ROBERTA_MODEL_DIR='${LEGACY_ROBERTA_MODEL_DIR:-}' &&
  export SENIOR_PROTOCOL='1' &&
  export FAKE_EXTRACTOR_ONLY='1' &&
  export RUN_LLM_CACHE='0' &&
  export DISABLE_GRAPH_REWEIGHTING='1' &&
  export RUN_LEGACY_BASELINES='0' &&
  export GRAPH_MODE='senior' &&
  export OUTPUT_DIR='${OUTPUT_DIR}' &&
  export SECONDARY_MODEL_NAME_OR_PATH='' &&
  bash graph/run_all.sh
" </dev/null >"${LOG_FILE}" 2>&1 &

PID=$!
echo "PID=${PID}"
