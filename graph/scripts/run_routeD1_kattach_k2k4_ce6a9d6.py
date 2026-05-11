from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path('/home/xyz/HuChao (2)/Bert-TextClassification')
OLD_ROOT = Path('/home/xyz/HuChao (2)/d1_ce6a9d6_repro')
sys.path.insert(0, str(PROJECT_ROOT))

from graph.baseline_comparison.src.data_loader import load_protocol_bundle
from graph.graph_pipeline import build_routek_d1main_rns_topk_graph_frames, compute_edge_stats


def _load_old_relation_model():
    module_path = OLD_ROOT / 'graph' / 'relation_model.py'
    spec = importlib.util.spec_from_file_location('old_relation_model_ce6a9d6', module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load old relation model from {module_path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


OLD_RELATION_MODEL = _load_old_relation_model()
run_relation_aggregation_experiments = OLD_RELATION_MODEL.run_relation_aggregation_experiments


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


def setup_logger(path: Path):
    import logging
    logger = logging.getLogger(str(path))
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(path, encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(fh)
    return logger

GRAPH_DIR = PROJECT_ROOT / 'graph'
BASE_PROTOCOL_DIR = GRAPH_DIR / 'outputs' / 'yelpzip_balanced_current_graph_no_reweight_20260502_160620'
REFERENCE_D1_DIR = GRAPH_DIR / 'outputs' / 'routeD_tns_guided_logic_egat_20260504_200855' / 'D1_EGAT_Base_LogicAE_CB'

K4_ARMS = [
    {"arm_id": 0, "UPU": 20, "UTU": 20, "USU": 20, "LogicAE_CB": 20},
    {"arm_id": 1, "UPU": 10, "UTU": 10, "USU": 20, "LogicAE_CB": 20},
    {"arm_id": 2, "UPU": 10, "UTU": 10, "USU": 10, "LogicAE_CB": 30},
    {"arm_id": 3, "UPU": 5, "UTU": 5, "USU": 20, "LogicAE_CB": 30},
    {"arm_id": 4, "UPU": 20, "UTU": 10, "USU": 10, "LogicAE_CB": 30},
    {"arm_id": 5, "UPU": 10, "UTU": 20, "USU": 10, "LogicAE_CB": 30},
    {"arm_id": 6, "UPU": 5, "UTU": 10, "USU": 10, "LogicAE_CB": 40},
    {"arm_id": 7, "UPU": 15, "UTU": 15, "USU": 15, "LogicAE_CB": 25},
    {"arm_id": 8, "UPU": 5, "UTU": 5, "USU": 10, "LogicAE_CB": 40},
    {"arm_id": 9, "UPU": 10, "UTU": 5, "USU": 20, "LogicAE_CB": 30},
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--output_root', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--alpha_abnormal', type=float, default=0.5)
    p.add_argument('--beta_tns', type=float, default=0.2)
    p.add_argument('--gamma_interaction', type=float, default=0.2)
    p.add_argument('--abnormal_score_source', default='auto')
    p.add_argument('--k4_warmup_epochs', type=int, default=15)
    p.add_argument('--bandit_lambda_density', type=float, default=0.02)
    return p.parse_args()


def _load_assets():
    bundle = load_protocol_bundle()
    review_scores_df = pd.read_csv(BASE_PROTOCOL_DIR / 'review_scores_enriched.csv')
    user_text_vectors = np.load(BASE_PROTOCOL_DIR / 'logic_vectors' / 'user_text_vectors.npy')
    user_abnormal_vectors = np.load(BASE_PROTOCOL_DIR / 'logic_vectors' / 'user_abnormal_vectors.npy')
    d1_run_summary = json.loads((REFERENCE_D1_DIR / 'run_summary.json').read_text(encoding='utf-8'))
    return bundle, review_scores_df, user_text_vectors, user_abnormal_vectors, d1_run_summary['best_graph_model']


def _k4_reward(val_auc: float, val_ap: float, edge_density_norm: float, lambda_density: float) -> float:
    return float(val_auc) + 0.5 * float(val_ap) - float(lambda_density) * float(edge_density_norm)


def _run_graph(exp_dir: Path, *, bundle, review_scores_df, user_text_vectors, user_abnormal_vectors, topk_mode: str, relation_k: dict[str, int], alpha: float, beta: float, gamma: float, abnormal_score_source: str, seed: int, max_epochs_override: int | None = None, patience_override: int | None = None):
    exp_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = exp_dir / 'metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    edge_frames = build_routek_d1main_rns_topk_graph_frames(
        user_df=bundle.user_df.copy(),
        review_features=review_scores_df.copy(),
        user_text_vectors=user_text_vectors,
        user_abnormal_vectors=user_abnormal_vectors,
        d1_edge_frames=bundle.edge_frames,
        output_dir=exp_dir,
        topk_mode=topk_mode,
        relation_k=relation_k,
        abnormal_score_source=abnormal_score_source,
        alpha_abnormal=alpha,
        beta_tns=beta,
        gamma_interaction=gamma,
        tns_phi_days=5,
    )
    compute_edge_stats(edge_frames=edge_frames, user_df=bundle.user_df, output_dir=exp_dir)
    result_df = run_relation_aggregation_experiments(
        user_df=bundle.user_df,
        self_features=bundle.node_features.astype(np.float32),
        edge_frames=edge_frames,
        output_dir=metrics_dir,
        review_encoder_name='llm_masked_logic',
        model_kind='edge_aware_gat',
        seed=seed,
        backbone='current_egat',
        relation_model='edge_aware_gat',
        review_scores_df=review_scores_df,
        selected_edge_set='Base_LogicAE_CB',
        relation_topk=None,
        use_node_gat=False,
        use_abnormal_edge_weight=False,
        use_abnormal_gate=False,
        use_abnormal_value_gate=False,
        use_abnormal_attention_bias=False,
        abnormal_score_source=abnormal_score_source,
    )
    row = result_df.loc[result_df['edge_set'] == 'Base_LogicAE_CB'].iloc[0].to_dict()
    edge_cfg_path = exp_dir / 'edges' / 'edge_build_config.json'
    edge_cfg = json.loads(edge_cfg_path.read_text(encoding='utf-8')) if edge_cfg_path.exists() else {}
    return row, edge_cfg


def main() -> None:
    args = parse_args()
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out / f"routeD1_kattach_ce6a9d6_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    bundle, review_scores_df, user_text_vectors, user_abnormal_vectors, d1_best = _load_assets()

    rows = [{
        'experiment_name': 'D1_EGAT_Base_LogicAE_CB',
        'strategy': 'reference_only',
        'selected_arm_id': '',
        'UPU_k': 20, 'UTU_k': 20, 'USU_k': 20, 'LogicAE_CB_k': 20,
        'AUC': d1_best['auc'], 'AP': d1_best['ap'], 'F1': d1_best['f1'], 'Recall': d1_best['recall'], 'Precision': d1_best['precision'],
        'test_threshold': d1_best['threshold'], 'notes': 'reference row',
    }]

    logger.info('running D1_K2_AbnormalTNSAware on ce6a9d6 implementation')
    k2_dir = out / 'D1_K2_AbnormalTNSAware'
    k2_row, k2_edge_cfg = _run_graph(k2_dir, bundle=bundle, review_scores_df=review_scores_df, user_text_vectors=user_text_vectors, user_abnormal_vectors=user_abnormal_vectors, topk_mode='abnormal_tns_aware', relation_k={'UPU':20,'UTU':20,'USU':20,'LogicAE_CB':20}, alpha=args.alpha_abnormal, beta=args.beta_tns, gamma=args.gamma_interaction, abnormal_score_source=args.abnormal_score_source, seed=args.seed)
    save_json(k2_dir / 'run_summary.json', {'experiment_name':'D1_K2_AbnormalTNSAware','best_graph_model':k2_row,'implementation':'ce6a9d6'})
    save_json(k2_dir / 'config.json', {'strategy':'K2_abnormal_tns_aware','implementation':'ce6a9d6'})
    rows.append({
        'experiment_name': 'D1_K2_AbnormalTNSAware',
        'strategy': 'K2_abnormal_tns_aware',
        'selected_arm_id': '',
        'UPU_k': 20, 'UTU_k': 20, 'USU_k': 20, 'LogicAE_CB_k': 20,
        'AUC': k2_row['auc'], 'AP': k2_row['ap'], 'F1': k2_row['f1'], 'Recall': k2_row['recall'], 'Precision': k2_row['precision'],
        'test_threshold': k2_row['threshold'], 'notes': 'ce6a9d6 implementation',
    })

    logger.info('running D1_K4_BanditSelected on ce6a9d6 implementation')
    k4_dir = out / 'D1_K4_BanditSelected'
    warm_rows = []
    for arm in K4_ARMS:
        arm_id = int(arm['arm_id'])
        relation_k = {k:int(v) for k,v in arm.items() if k!='arm_id'}
        arm_dir = k4_dir / f'arm_{arm_id:02d}' / 'warmup'
        row, _ = _run_graph(arm_dir, bundle=bundle, review_scores_df=review_scores_df, user_text_vectors=user_text_vectors, user_abnormal_vectors=user_abnormal_vectors, topk_mode='abnormal_tns_aware', relation_k=relation_k, alpha=args.alpha_abnormal, beta=args.beta_tns, gamma=args.gamma_interaction, abnormal_score_source=args.abnormal_score_source, seed=args.seed)
        num_edges = 0
        for rel in ['UPU','UTU','USU','LogicAE_CB']:
            p = arm_dir / 'edges' / f'{rel}_edges.csv'
            if p.exists():
                num_edges += len(pd.read_csv(p))
        warm_rows.append({'arm_id':arm_id,'UPU_k':relation_k['UPU'],'UTU_k':relation_k['UTU'],'USU_k':relation_k['USU'],'LogicAE_CB_k':relation_k['LogicAE_CB'],'num_edges':num_edges,'warmup_val_auc':row.get('val_auc'),'warmup_val_ap':row.get('val_ap')})
    arm_df = pd.DataFrame(warm_rows)
    arm0_edges = max(float(arm_df.loc[arm_df['arm_id']==0,'num_edges'].iloc[0]), 1.0)
    arm_df['edge_density_norm'] = arm_df['num_edges'].astype(float) / arm0_edges
    arm_df['warmup_reward'] = arm_df.apply(lambda r: _k4_reward(r['warmup_val_auc'], r['warmup_val_ap'], r['edge_density_norm'], args.bandit_lambda_density), axis=1)
    top3 = arm_df.sort_values('warmup_reward', ascending=False).head(3)['arm_id'].astype(int).tolist()
    arm_df['selected_for_full_train'] = arm_df['arm_id'].isin(top3)
    full_rows = []
    for arm_id in top3:
        arm = next(a for a in K4_ARMS if int(a['arm_id']) == arm_id)
        relation_k = {k:int(v) for k,v in arm.items() if k!='arm_id'}
        full_dir = k4_dir / f'arm_{arm_id:02d}' / 'full'
        row, _ = _run_graph(full_dir, bundle=bundle, review_scores_df=review_scores_df, user_text_vectors=user_text_vectors, user_abnormal_vectors=user_abnormal_vectors, topk_mode='abnormal_tns_aware', relation_k=relation_k, alpha=args.alpha_abnormal, beta=args.beta_tns, gamma=args.gamma_interaction, abnormal_score_source=args.abnormal_score_source, seed=args.seed)
        reward = _k4_reward(float(row.get('val_auc',0.0)), float(row.get('val_ap',0.0)), float(arm_df.loc[arm_df['arm_id']==arm_id,'edge_density_norm'].iloc[0]), args.bandit_lambda_density)
        full_rows.append({'arm_id':arm_id,'UPU_k':relation_k['UPU'],'UTU_k':relation_k['UTU'],'USU_k':relation_k['USU'],'LogicAE_CB_k':relation_k['LogicAE_CB'],'num_edges':int(arm_df.loc[arm_df['arm_id']==arm_id,'num_edges'].iloc[0]),'edge_density_norm':float(arm_df.loc[arm_df['arm_id']==arm_id,'edge_density_norm'].iloc[0]),'warmup_val_auc':float(arm_df.loc[arm_df['arm_id']==arm_id,'warmup_val_auc'].iloc[0]),'warmup_val_ap':float(arm_df.loc[arm_df['arm_id']==arm_id,'warmup_val_ap'].iloc[0]),'warmup_reward':float(arm_df.loc[arm_df['arm_id']==arm_id,'warmup_reward'].iloc[0]),'selected_for_full_train':True,'full_train_val_auc':row.get('val_auc'),'full_train_val_ap':row.get('val_ap'),'full_train_reward':reward,'test_auc':row.get('auc'),'test_ap':row.get('ap'),'test_f1':row.get('f1'),'test_recall':row.get('recall'),'test_precision':row.get('precision')})
    full_df = pd.DataFrame(full_rows)
    selected = full_df.sort_values('full_train_reward', ascending=False).iloc[0].to_dict()
    (k4_dir / 'metrics').mkdir(parents=True, exist_ok=True)
    arm_df.merge(full_df, on=['arm_id','UPU_k','UTU_k','USU_k','LogicAE_CB_k','num_edges','edge_density_norm','selected_for_full_train'], how='left').to_csv(k4_dir / 'metrics' / 'bandit_arm_search.csv', index=False)
    save_json(k4_dir / 'run_summary.json', {'experiment_name':'D1_K4_BanditSelected','selected_arm_id':int(selected['arm_id']),'selected_relation_k':{'UPU':int(selected['UPU_k']),'UTU':int(selected['UTU_k']),'USU':int(selected['USU_k']),'LogicAE_CB':int(selected['LogicAE_CB_k'])},'implementation':'ce6a9d6'})
    save_json(k4_dir / 'config.json', {'strategy':'K4_bandit','selected_arm_id':int(selected['arm_id']),'implementation':'ce6a9d6'})
    rows.append({
        'experiment_name': 'D1_K4_BanditSelected',
        'strategy': 'K4_bandit',
        'selected_arm_id': int(selected['arm_id']),
        'UPU_k': int(selected['UPU_k']), 'UTU_k': int(selected['UTU_k']), 'USU_k': int(selected['USU_k']), 'LogicAE_CB_k': int(selected['LogicAE_CB_k']),
        'AUC': selected['test_auc'], 'AP': selected['test_ap'], 'F1': selected['test_f1'], 'Recall': selected['test_recall'], 'Precision': selected['test_precision'],
        'test_threshold': 'UNKNOWN_FROM_D1', 'notes': 'ce6a9d6 implementation',
    })

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out / 'routeD1_kattach_summary.csv', index=False)
    (out / 'routeD1_kattach_summary.md').write_text(summary_df.to_csv(index=False), encoding='utf-8')


if __name__ == '__main__':
    main()
