#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "No usable Python interpreter found. Set PYTHON_BIN or install python3." >&2
    exit 1
  fi
fi
MASK_SOURCE="${MASK_SOURCE:-llm}"
FAKE_EXTRACTOR_ONLY="${FAKE_EXTRACTOR_ONLY:-0}"
SENIOR_PROTOCOL="${SENIOR_PROTOCOL:-0}"
if [[ "${SENIOR_PROTOCOL}" == "1" ]]; then
  DEFAULT_LLM_JSONL="${ROOT_DIR}/graph/outputs/llm_cache/yelpzip_senior_llm_abnormal_patterns.jsonl"
else
  DEFAULT_LLM_JSONL="${ROOT_DIR}/graph/outputs/llm_cache/yelpzip_llm_abnormal_patterns.jsonl"
fi
if [[ "${FAKE_EXTRACTOR_ONLY}" == "1" ]]; then
  MASK_SOURCE="full_text"
fi
if [[ -z "${OUTPUT_DIR:-}" ]]; then
  if [[ "${SENIOR_PROTOCOL}" == "1" && "${FAKE_EXTRACTOR_ONLY}" == "1" ]]; then
    OUTPUT_DIR="${ROOT_DIR}/graph/outputs/yelpzip_senior_fake_extractor_only"
  elif [[ "${SENIOR_PROTOCOL}" == "1" ]]; then
    OUTPUT_DIR="${ROOT_DIR}/graph/outputs/yelpzip_senior_protocol"
  elif [[ "${FAKE_EXTRACTOR_ONLY}" == "1" ]]; then
    OUTPUT_DIR="${ROOT_DIR}/graph/outputs/yelpzip_fake_extractor_only"
  else
    OUTPUT_DIR="${ROOT_DIR}/graph/outputs/yelpzip_final"
  fi
fi

if [[ -n "${HF_ENDPOINT:-}" ]]; then
  export HF_ENDPOINT
fi
if [[ -n "${HF_HOME:-}" ]]; then
  export HF_HOME
fi
if [[ -n "${TRANSFORMERS_CACHE:-}" ]]; then
  export TRANSFORMERS_CACHE
fi

FINAL_ARGS=(
  --graph_data_dir "${ROOT_DIR}/graph data"
  --output_dir "${OUTPUT_DIR}"
  --mask_source "${MASK_SOURCE}"
  --primary_model_name_or_path "${PRIMARY_MODEL_NAME_OR_PATH:-roberta-base}"
  --time_bucket "${TIME_BUCKET:-week}"
  --relation_model "${RELATION_MODEL:-relation_attn}"
  --legacy_roberta_model_dir "${LEGACY_ROBERTA_MODEL_DIR:-roberta-base}"
)

if [[ "${SENIOR_PROTOCOL}" == "1" ]]; then
  FINAL_ARGS+=(--senior_protocol)
fi
if [[ "${MASK_SOURCE}" == "llm" ]]; then
  FINAL_ARGS+=(--llm_jsonl_path "${LLM_JSONL_PATH:-${DEFAULT_LLM_JSONL}}")
fi

if [[ -n "${SECONDARY_MODEL_NAME_OR_PATH+x}" ]]; then
  SECONDARY_MODEL="${SECONDARY_MODEL_NAME_OR_PATH}"
elif [[ "${FAKE_EXTRACTOR_ONLY}" == "1" ]]; then
  SECONDARY_MODEL=""
else
  SECONDARY_MODEL="pretrain_model/skep-base"
fi
if [[ -n "${SECONDARY_MODEL}" ]]; then
  FINAL_ARGS+=(--secondary_model_name_or_path "${SECONDARY_MODEL}")
fi

if [[ "${BALANCE_USER_LABELS:-0}" == "1" ]]; then
  FINAL_ARGS+=(--balance_user_labels)
fi
if [[ -n "${BALANCED_USER_COUNT:-}" ]]; then
  FINAL_ARGS+=(--balanced_user_count "${BALANCED_USER_COUNT}")
fi
if [[ -n "${GRAPH_MODE:-}" ]]; then
  FINAL_ARGS+=(--graph_mode "${GRAPH_MODE}")
fi
if [[ -n "${SENIOR_USU_RATIO:-}" ]]; then
  FINAL_ARGS+=(--senior_usu_ratio "${SENIOR_USU_RATIO}")
fi
DEFAULT_RUN_LEGACY_BASELINES="1"
if [[ "${FAKE_EXTRACTOR_ONLY}" == "1" ]]; then
  DEFAULT_RUN_LEGACY_BASELINES="0"
fi
if [[ "${RUN_LEGACY_BASELINES:-${DEFAULT_RUN_LEGACY_BASELINES}}" == "1" ]]; then
  FINAL_ARGS+=(--run_legacy_baselines)
fi

if [[ "${DISABLE_GRAPH_REWEIGHTING:-0}" == "1" ]]; then
  FINAL_ARGS+=(--disable_graph_reweighting)
fi
if [[ -n "${GRAPH_REWEIGHT_ALPHA:-}" ]]; then
  FINAL_ARGS+=(--graph_reweight_alpha "${GRAPH_REWEIGHT_ALPHA}")
fi
if [[ -n "${GRAPH_SUPPORT_TOP_K:-}" ]]; then
  FINAL_ARGS+=(--graph_support_top_k "${GRAPH_SUPPORT_TOP_K}")
fi
if [[ -n "${GRAPH_SUPPORT_NEIGHBOR_REVIEW_CAP:-}" ]]; then
  FINAL_ARGS+=(--graph_support_neighbor_review_cap "${GRAPH_SUPPORT_NEIGHBOR_REVIEW_CAP}")
fi
if [[ -n "${LOGIC_THRESHOLD_MODE:-}" ]]; then
  FINAL_ARGS+=(--logic_threshold_mode "${LOGIC_THRESHOLD_MODE}")
fi
if [[ -n "${LOGIC_THRESHOLD_QUANTILE:-}" ]]; then
  FINAL_ARGS+=(--logic_threshold_quantile "${LOGIC_THRESHOLD_QUANTILE}")
fi
if [[ -n "${LOGIC_THRESHOLD_VALUE:-}" ]]; then
  FINAL_ARGS+=(--logic_threshold_value "${LOGIC_THRESHOLD_VALUE}")
fi

"${PYTHON_BIN}" -m graph.run_final_experiment "${FINAL_ARGS[@]}" "$@"
