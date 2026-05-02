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

if [[ -n "${HF_ENDPOINT:-}" ]]; then
  export HF_ENDPOINT
fi
if [[ -n "${HF_HOME:-}" ]]; then
  export HF_HOME
fi
if [[ -n "${TRANSFORMERS_CACHE:-}" ]]; then
  export TRANSFORMERS_CACHE
fi

GRAPH_DATA_DIR="${GRAPH_DATA_DIR:-${ROOT_DIR}/graph data}"
FAKE_EXTRACTOR_ONLY="${FAKE_EXTRACTOR_ONLY:-0}"
SENIOR_PROTOCOL="${SENIOR_PROTOCOL:-0}"
MASK_SOURCE="${MASK_SOURCE:-llm}"
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
if [[ -z "${LLM_JSONL_PATH:-}" && "${SENIOR_PROTOCOL}" == "1" ]]; then
  LLM_JSONL="${ROOT_DIR}/graph/outputs/llm_cache/yelpzip_senior_llm_abnormal_patterns.jsonl"
else
  LLM_JSONL="${LLM_JSONL_PATH:-${ROOT_DIR}/graph/outputs/llm_cache/yelpzip_llm_abnormal_patterns.jsonl}"
fi
if [[ -z "${LLM_SEED_JSONL_PATH:-}" && "${SENIOR_PROTOCOL}" == "1" ]]; then
  LLM_SEED_JSONL="${ROOT_DIR}/graph/outputs/llm_cache/yelpzip_senior_llm_seed.jsonl"
else
  LLM_SEED_JSONL="${LLM_SEED_JSONL_PATH:-${ROOT_DIR}/graph/outputs/llm_cache/yelpzip_llm_seed.jsonl}"
fi

echo "[1/2] Preparing mask source: ${MASK_SOURCE}"
if [[ "${MASK_SOURCE}" == "llm" ]]; then
  echo "      LLM cache output: ${LLM_JSONL}"
else
  echo "      no LLM cache is required for this run"
fi

if [[ "${MASK_SOURCE}" == "llm" && "${RUN_LLM_CACHE:-1}" == "1" ]]; then
  LLM_OUTPUT_JSONL="${LLM_JSONL}"
  if [[ "${USE_LOCAL_ANNOTATOR:-0}" == "1" ]]; then
    LLM_OUTPUT_JSONL="${LLM_SEED_JSONL}"
    echo "      local annotator mode: seed cache ${LLM_OUTPUT_JSONL}"
  fi

  LLM_ARGS=(
    --graph_data_dir "${GRAPH_DATA_DIR}"
    --prepared_output_dir "${OUTPUT_DIR}/prepared_data"
    --output_jsonl "${LLM_OUTPUT_JSONL}"
    --prompt_path "${LLM_PROMPT_PATH:-${ROOT_DIR}/graph/prompts/llm_abnormal_pattern_extraction.txt}"
    --model "${LLM_MODEL:-gpt-4o-mini}"
    --base_url "${LLM_BASE_URL:-${OPENAI_BASE_URL:-https://api.openai.com/v1}}"
    --enable_thinking "${LLM_ENABLE_THINKING:-auto}"
    --max_tokens "${LLM_MAX_TOKENS:-512}"
    --workers "${LLM_WORKERS:-4}"
  )

  if [[ "${SENIOR_PROTOCOL}" == "1" ]]; then
    LLM_ARGS+=(--senior_protocol)
  fi
  if [[ -n "${DATA_PATH:-}" ]]; then
    LLM_ARGS+=(--data_path "${DATA_PATH}")
  fi
  if [[ "${BALANCE_USER_LABELS:-0}" == "1" ]]; then
    LLM_ARGS+=(--balance_user_labels)
  fi
  if [[ -n "${BALANCED_USER_COUNT:-}" ]]; then
    LLM_ARGS+=(--balanced_user_count "${BALANCED_USER_COUNT}")
  fi
  if [[ -n "${LLM_LIMIT:-}" ]]; then
    LLM_ARGS+=(--limit "${LLM_LIMIT}")
  fi
  if [[ "${USE_LOCAL_ANNOTATOR:-0}" == "1" ]]; then
    LLM_ARGS+=(--sample_size "${LLM_SEED_SIZE:-5000}")
    LLM_ARGS+=(--sample_strategy "${LLM_SAMPLE_STRATEGY:-balanced}")
    LLM_ARGS+=(--sample_seed "${LLM_SAMPLE_SEED:-42}")
  fi
  if [[ -n "${LLM_START_REVIEW_NODE_ID:-}" ]]; then
    LLM_ARGS+=(--start_review_node_id "${LLM_START_REVIEW_NODE_ID}")
  fi
  if [[ "${LLM_OVERWRITE_CACHE:-0}" == "1" ]]; then
    LLM_ARGS+=(--overwrite_cache)
  fi
  if [[ "${LLM_CONTINUE_ON_ERROR:-0}" == "1" ]]; then
    LLM_ARGS+=(--continue_on_error)
  fi
  if [[ "${LLM_NO_RESPONSE_FORMAT:-0}" == "1" ]]; then
    LLM_ARGS+=(--no_response_format)
  fi
  if [[ "${LLM_COMPACT_PROMPT:-0}" == "1" ]]; then
    LLM_ARGS+=(--compact_prompt)
  fi

  "${PYTHON_BIN}" -m graph.generate_llm_cache "${LLM_ARGS[@]}"
else
  echo "      skipped LLM cache generation"
fi

if [[ "${MASK_SOURCE}" == "llm" && -n "${LLM_LIMIT:-}" ]]; then
  echo "LLM_LIMIT=${LLM_LIMIT} was set, so run_all stops after cache generation to avoid a partial formal run."
  echo "Inspect the generated cache, then unset LLM_LIMIT and rerun bash graph/run_all.sh for the full experiment."
  exit 0
fi

if [[ "${MASK_SOURCE}" == "llm" && "${USE_LOCAL_ANNOTATOR:-0}" == "1" ]]; then
  echo "[local] Training/generating full cache with local seq2seq annotator"
  LOCAL_ARGS=(
    --reviews_csv "${OUTPUT_DIR}/prepared_data/reviews_canonical.csv"
    --seed_jsonl "${LLM_SEED_JSONL}"
    --output_jsonl "${LLM_JSONL}"
    --model_name_or_path "${LOCAL_ANNOTATOR_BASE_MODEL:-google/flan-t5-base}"
    --model_dir "${LOCAL_ANNOTATOR_MODEL_DIR:-${ROOT_DIR}/graph/outputs/local_annotator/t5_abnormal_extractor}"
    --num_train_epochs "${LOCAL_ANNOTATOR_EPOCHS:-3}"
    --batch_size "${LOCAL_ANNOTATOR_BATCH_SIZE:-4}"
    --learning_rate "${LOCAL_ANNOTATOR_LR:-3e-5}"
    --min_seed_rows "${LOCAL_ANNOTATOR_MIN_SEED_ROWS:-200}"
  )
  if [[ "${LOCAL_ANNOTATOR_OVERWRITE_OUTPUT:-0}" == "1" ]]; then
    LOCAL_ARGS+=(--overwrite_output)
  fi
  if [[ -n "${LOCAL_ANNOTATOR_LIMIT_GENERATE:-}" ]]; then
    LOCAL_ARGS+=(--limit_generate "${LOCAL_ANNOTATOR_LIMIT_GENERATE}")
  fi
  "${PYTHON_BIN}" -m graph.train_local_annotator "${LOCAL_ARGS[@]}"
fi

if [[ "${MASK_SOURCE}" == "llm" && ! -s "${LLM_JSONL}" ]]; then
  echo "LLM cache is missing or empty: ${LLM_JSONL}" >&2
  echo "Set OPENAI_API_KEY/LLM_API_KEY, or set RUN_LLM_CACHE=0 only when the cache already exists." >&2
  exit 1
fi

echo "[2/2] Running final YelpZip graph experiment"

FINAL_ARGS=(
  --graph_data_dir "${GRAPH_DATA_DIR}"
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
  FINAL_ARGS+=(--llm_jsonl_path "${LLM_JSONL}")
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

if [[ -n "${DATA_PATH:-}" ]]; then
  FINAL_ARGS+=(--data_path "${DATA_PATH}")
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
