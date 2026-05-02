#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/graph/outputs/yelpzip_balanced_current_graph_no_reweight_${TIMESTAMP}}"
LOG_DIR="${ROOT_DIR}/graph/logs"
LOG_FILE="${LOG_DIR}/balanced_current_graph_no_reweight_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}"

source "${ROOT_DIR}/graph/run_all.env.sh"

export FAKE_EXTRACTOR_ONLY=1
export RUN_LLM_CACHE=0
export BALANCE_USER_LABELS=1
export BALANCED_USER_COUNT=6742
export GRAPH_MODE=current
export DISABLE_GRAPH_REWEIGHTING=1
export RUN_LEGACY_BASELINES=0
export OUTPUT_DIR

# Keep the same clean comparison setting: no SKEP branch.
export SECONDARY_MODEL_NAME_OR_PATH=""

echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "LOG_FILE=${LOG_FILE}"
echo "GRAPH_MODE=${GRAPH_MODE}"
echo "BALANCE_USER_LABELS=${BALANCE_USER_LABELS}"
echo "BALANCED_USER_COUNT=${BALANCED_USER_COUNT}"
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
  export FAKE_EXTRACTOR_ONLY='1' &&
  export RUN_LLM_CACHE='0' &&
  export BALANCE_USER_LABELS='1' &&
  export BALANCED_USER_COUNT='6742' &&
  export GRAPH_MODE='current' &&
  export DISABLE_GRAPH_REWEIGHTING='1' &&
  export RUN_LEGACY_BASELINES='0' &&
  export OUTPUT_DIR='${OUTPUT_DIR}' &&
  export SECONDARY_MODEL_NAME_OR_PATH='' &&
  bash graph/run_all.sh --train_ratio 0.64 --val_ratio 0.16 --test_ratio 0.20
" </dev/null >"${LOG_FILE}" 2>&1 &

PID=$!
echo "PID=${PID}"
