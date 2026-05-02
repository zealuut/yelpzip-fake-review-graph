#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_OUTPUT_JSONL="${ROOT_DIR}/graph/outputs/llm_cache/yelpzip_llm_abnormal_patterns.jsonl"
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

ARGS=(
  --graph_data_dir "${ROOT_DIR}/graph data"
  --prepared_output_dir "${ROOT_DIR}/graph/outputs/yelpzip_final/prepared_data"
  --output_jsonl "${LLM_JSONL_PATH:-${DEFAULT_OUTPUT_JSONL}}"
  --prompt_path "${LLM_PROMPT_PATH:-${ROOT_DIR}/graph/prompts/llm_abnormal_pattern_extraction.txt}"
  --model "${LLM_MODEL:-gpt-4o-mini}"
  --base_url "${LLM_BASE_URL:-${OPENAI_BASE_URL:-https://api.openai.com/v1}}"
  --enable_thinking "${LLM_ENABLE_THINKING:-auto}"
  --max_tokens "${LLM_MAX_TOKENS:-512}"
  --timeout "${LLM_TIMEOUT:-60}"
  --retries "${LLM_RETRIES:-4}"
  --retry_sleep "${LLM_RETRY_SLEEP:-2.0}"
  --workers "${LLM_WORKERS:-4}"
)

if [[ -n "${LLM_LIMIT:-}" ]]; then
  ARGS+=(--limit "${LLM_LIMIT}")
fi
if [[ -n "${LLM_START_REVIEW_NODE_ID:-}" ]]; then
  ARGS+=(--start_review_node_id "${LLM_START_REVIEW_NODE_ID}")
fi
if [[ -n "${LLM_MAX_IN_FLIGHT:-}" ]]; then
  ARGS+=(--max_in_flight "${LLM_MAX_IN_FLIGHT}")
fi
if [[ -n "${LLM_RPM_LIMIT:-}" ]]; then
  ARGS+=(--rpm_limit "${LLM_RPM_LIMIT}")
fi
if [[ -n "${LLM_TPM_LIMIT:-}" ]]; then
  ARGS+=(--tpm_limit "${LLM_TPM_LIMIT}")
fi
if [[ -n "${LLM_RATE_LIMIT_SAFETY:-}" ]]; then
  ARGS+=(--rate_limit_safety "${LLM_RATE_LIMIT_SAFETY}")
fi
if [[ -n "${LLM_EXPECTED_OUTPUT_TOKENS:-}" ]]; then
  ARGS+=(--expected_output_tokens "${LLM_EXPECTED_OUTPUT_TOKENS}")
fi
if [[ -n "${LLM_MIN_RETRY_AFTER:-}" ]]; then
  ARGS+=(--min_retry_after "${LLM_MIN_RETRY_AFTER}")
fi
if [[ "${LLM_OVERWRITE_CACHE:-0}" == "1" ]]; then
  ARGS+=(--overwrite_cache)
fi
if [[ "${LLM_CONTINUE_ON_ERROR:-0}" == "1" ]]; then
  ARGS+=(--continue_on_error)
fi
if [[ "${LLM_NO_RESPONSE_FORMAT:-0}" == "1" ]]; then
  ARGS+=(--no_response_format)
fi
if [[ "${LLM_COMPACT_PROMPT:-0}" == "1" ]]; then
  ARGS+=(--compact_prompt)
fi

"${PYTHON_BIN}" -m graph.generate_llm_cache "${ARGS[@]}" "$@"
