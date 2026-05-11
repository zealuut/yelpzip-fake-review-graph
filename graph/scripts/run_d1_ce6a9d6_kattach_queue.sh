#!/usr/bin/env bash
set -euo pipefail
WT='/home/xyz/HuChao (2)/d1_ce6a9d6_repro'
ROOT='/home/xyz/HuChao (2)/Bert-TextClassification'
TS=$(date +%Y%m%d_%H%M%S)
OUT="$ROOT/graph/outputs/routeD1_kattach_k2k4_ce6a9d6_${TS}"
LOG="$ROOT/graph/logs/routeD1_kattach_k2k4_ce6a9d6_${TS}.log"
mkdir -p "$OUT"
exec > "$LOG" 2>&1

echo "queue started out=$OUT"
export ROUTE_D1_KATTACH_OUT="$OUT"
export ROUTE_D1_WT="$WT"
export ROUTE_D1_ROOT="$ROOT"
python3 - <<'PY'
import json, os, sys
from pathlib import Path
out = Path(os.environ['ROUTE_D1_KATTACH_OUT'])
wt = Path(os.environ['ROUTE_D1_WT'])
root = Path(os.environ['ROUTE_D1_ROOT'])
os.chdir(str(wt))
sys.path.insert(0, str(wt))
from graph.scripts.route_runner import _load_base_artifacts as load_old, _build_route_edges as build_old
from graph.graph_pipeline import build_self_feature_matrix as build_sf_old, compute_edge_stats as edge_stats_old
from graph.relation_model import run_relation_aggregation_experiments as run_old

base_dir = root / 'graph' / 'outputs' / 'yelpzip_balanced_current_graph_no_reweight_20260502_160620'
ref_dir = root / 'graph' / 'outputs' / 'routeD_tns_guided_logic_egat_20260504_200855' / 'D1_EGAT_Base_LogicAE_CB'
arts = load_old(base_dir)
self_features = build_sf_old(arts['user_df'], arts['user_abnormal_vectors'])
k0_dir = out / 'K0_StrictSanity'
k0_dir.mkdir(parents=True, exist_ok=True)
(k0_dir/'metrics').mkdir(exist_ok=True)
edge_frames = build_old(base_artifacts=arts, graph_mode='current', top_k=20, senior_usu_ratio=0.10, route_output_dir=k0_dir)
edge_stats_old(edge_frames=edge_frames, user_df=arts['user_df'], output_dir=k0_dir)
result_df = run_old(
    user_df=arts['user_df'],
    self_features=self_features,
    edge_frames=edge_frames,
    output_dir=k0_dir / 'metrics',
    review_encoder_name='llm_masked_logic',
    model_kind='edge_aware_gat',
    seed=42,
    backbone='current_egat',
    relation_model='edge_aware_gat',
    use_abnormal_edge_weight=False,
    use_abnormal_gate=False,
    use_abnormal_value_gate=False,
    use_abnormal_attention_bias=False,
    abnormal_score_source='auto',
    abnormal_edge_lambda=1.0,
    abnormal_edge_eta=0.5,
    abnormal_gate_eta=0.5,
    abnormal_pair_mode='both_high',
    abnormal_gate_learnable=False,
    abnormal_attention_gamma=1.0,
    review_scores_df=arts['review_scores_df'],
    selected_edge_set='Base_LogicAE_CB',
    relation_topk=None,
    use_node_gat=False,
)
row = result_df[result_df['edge_set']=='Base_LogicAE_CB'].iloc[0].to_dict()
(k0_dir/'run_summary.json').write_text(json.dumps({'best_graph_model': row}, indent=2), encoding='utf-8')
(k0_dir/'config.json').write_text(json.dumps({'base_dir': str(base_dir), 'reference_dir': str(ref_dir), 'implementation': 'ce6a9d6'}, indent=2), encoding='utf-8')
(ref_row := json.loads((ref_dir/'run_summary.json').read_text(encoding='utf-8'))['best_graph_model'])
if abs(float(row['auc']) - float(ref_row['auc'])) > 0.003 or abs(float(row['ap']) - float(ref_row['ap'])) > 0.003:
    (out/'FAILED_K0_NOT_STRICT_D1.md').write_text('ce6a9d6 K0 still failed strict D1 tolerance\n', encoding='utf-8')
    print('K0 strict failed; stop before K2/K4')
    raise SystemExit(3)
print('K0 strict passed; continue to K2/K4')
PY
status=$?
if [ "$status" -ne 0 ]; then
  exit 0
fi
python3 "$ROOT/graph/scripts/run_routeD1_kattach_k2k4.py" --output_root "$OUT" --seed 42
