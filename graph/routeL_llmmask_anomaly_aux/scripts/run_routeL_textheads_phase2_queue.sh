#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"
STAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
OUT_BASE="$ROOT_DIR/graph/outputs/routeL_textheads_phase2_${STAMP}"
LOG_BASE="$ROOT_DIR/graph/logs"
mkdir -p "$OUT_BASE" "$LOG_BASE"
run_pair() {
  local tag_a="$1"; shift
  local cfg_a="$1"; shift
  local tag_b="$1"; shift
  local cfg_b="$1"; shift
  nohup python3 -m graph.routeL_llmmask_anomaly_aux.scripts.run_routeL_textheads_phase2 \
    --output_root "$OUT_BASE/${tag_a}" --config_paths "$cfg_a" \
    > "$LOG_BASE/${tag_a}_${STAMP}.log" 2>&1 &
  pid_a=$!
  nohup python3 -m graph.routeL_llmmask_anomaly_aux.scripts.run_routeL_textheads_phase2 \
    --output_root "$OUT_BASE/${tag_b}" --config_paths "$cfg_b" \
    > "$LOG_BASE/${tag_b}_${STAMP}.log" 2>&1 &
  pid_b=$!
  wait "$pid_a"
  wait "$pid_b"
}
run_single() {
  local tag="$1"; shift
  local cfg="$1"; shift
  nohup python3 -m graph.routeL_llmmask_anomaly_aux.scripts.run_routeL_textheads_phase2 \
    --output_root "$OUT_BASE/${tag}" --config_paths "$cfg" \
    > "$LOG_BASE/${tag}_${STAMP}.log" 2>&1 &
  pid=$!
  wait "$pid"
}
CFG_DIR="$ROOT_DIR/graph/routeL_llmmask_anomaly_aux/configs"
run_pair exp0 "$CFG_DIR/E0_current_main_abnormal_head.yaml" exp1 "$CFG_DIR/E1_learned_token_evidence_head.yaml"
run_pair exp2k8 "$CFG_DIR/E2_topk_token_evidence_k8.yaml" exp2k16 "$CFG_DIR/E2_topk_token_evidence_k16.yaml"
run_pair exp3 "$CFG_DIR/E3_local_phrase_cnn_branch.yaml" exp4 "$CFG_DIR/E4_psycholinguistic_style_branch.yaml"
run_single exp5 "$CFG_DIR/E5_textual_semantic_drift_head.yaml"
