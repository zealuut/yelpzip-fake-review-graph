#!/usr/bin/env bash

export PYTHON_BIN="python3"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="/home/xyz/HuChao (2)/Bert-TextClassification/.hf_cache"
export TRANSFORMERS_CACHE="/home/xyz/HuChao (2)/Bert-TextClassification/.hf_cache/transformers"

export PRIMARY_MODEL_NAME_OR_PATH="${PRIMARY_MODEL_NAME_OR_PATH:-roberta-base}"
export SECONDARY_MODEL_NAME_OR_PATH="${SECONDARY_MODEL_NAME_OR_PATH:-}"
export LOCAL_ANNOTATOR_BASE_MODEL="${LOCAL_ANNOTATOR_BASE_MODEL:-google/flan-t5-base}"
export LEGACY_ROBERTA_MODEL_DIR="${LEGACY_ROBERTA_MODEL_DIR:-roberta-base}"

export LLM_API_KEY="${LLM_API_KEY:-}"
export LLM_BASE_URL="https://api.siliconflow.cn/v1"
export LLM_MODEL="Qwen/Qwen2.5-7B-Instruct"
export LLM_ENABLE_THINKING="auto"
export LLM_MAX_TOKENS=160
export LLM_TIMEOUT=90
export LLM_RETRIES=5
export LLM_RETRY_SLEEP=2.5
export LLM_WORKERS=24
export LLM_MAX_IN_FLIGHT=48
export LLM_RPM_LIMIT=900
export LLM_TPM_LIMIT=42000
export LLM_RATE_LIMIT_SAFETY=0.80
export LLM_EXPECTED_OUTPUT_TOKENS=72
export LLM_MIN_RETRY_AFTER=8
export LLM_CONTINUE_ON_ERROR=1
export RUN_LEGACY_BASELINES=0
export LLM_PROMPT_PATH="/home/xyz/HuChao (2)/Bert-TextClassification/graph/prompts/llm_abnormal_pattern_extraction_qwen.txt"

export USE_LOCAL_ANNOTATOR=0
