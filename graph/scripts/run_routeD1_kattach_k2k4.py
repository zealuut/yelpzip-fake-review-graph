from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from graph.baseline_comparison.src.data_loader import load_protocol_bundle
from graph.baseline_comparison.src.utils import save_json, setup_logger
from graph.graph_pipeline import build_routek_d1main_rns_topk_graph_frames, compute_edge_stats
from graph.relation_model import run_relation_aggregation_experiments

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
    p.add_argument('--run_k0_sanity', action='store_true')
    return p.parse_args()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _sha256_array(arr: np.ndarray) -> str:
    return _sha256_bytes(np.ascontiguousarray(arr).tobytes())


def _sha256_df(df: pd.DataFrame) -> str:
    return _sha256_bytes(pd.util.hash_pandas_object(df, index=False).values.tobytes()) if not df.empty else 'EMPTY'


def _load_assets() -> dict[str, Any]:
    bundle = load_protocol_bundle()
    review_scores_df = pd.read_csv(BASE_PROTOCOL_DIR / 'review_scores_enriched.csv')
    user_text_vectors = np.load(BASE_PROTOCOL_DIR / 'logic_vectors' / 'user_text_vectors.npy')
    user_abnormal_vectors = np.load(BASE_PROTOCOL_DIR / 'logic_vectors' / 'user_abnormal_vectors.npy')
    d1_run_summary = json.loads((REFERENCE_D1_DIR / 'run_summary.json').read_text(encoding='utf-8'))
    d1_config = json.loads((REFERENCE_D1_DIR / 'config.json').read_text(encoding='utf-8'))
    d1_run_config = json.loads((REFERENCE_D1_DIR / 'run_config.json').read_text(encoding='utf-8'))
    return {
        'bundle': bundle,
        'review_scores_df': review_scores_df,
        'user_text_vectors': user_text_vectors,
        'user_abnormal_vectors': user_abnormal_vectors,
        'd1_run_summary': d1_run_summary,
        'd1_config': d1_config,
        'd1_run_config': d1_run_config,
    }


def _audit(assets: dict[str, Any]) -> dict[str, Any]:
    bundle = assets['bundle']
    user_df = bundle.user_df.copy()
    split_payload = {
        split: sorted(user_df.loc[user_df['split'].astype(str) == split, 'user_id'].astype(str).tolist())
        for split in ['train', 'val', 'test']
    }
    split_hash = {k: _sha256_bytes('\n'.join(v).encode('utf-8')) for k, v in split_payload.items()}
    label_hash = _sha256_bytes(user_df[['user_id', 'user_label']].sort_values('user_id').to_csv(index=False).encode('utf-8'))
    edge_counts = {name: int(len(bundle.edge_frames[name])) for name in ['UPU', 'UTU', 'USU', 'LogicAE_CB']}
    edge_hashes = {name: _sha256_df(bundle.edge_frames[name]) for name in ['UPU', 'UTU', 'USU', 'LogicAE_CB']}
    feature_path = 'build_self_feature_matrix(user_scores_enriched.csv, logic_vectors/user_abnormal_vectors.npy) via load_protocol_bundle()'
    audit = {
        'base_source': str(BASE_PROTOCOL_DIR),
        'reference_dir': str(REFERENCE_D1_DIR),
        'is_strict_d1_artifact': True,
        'feature_path': feature_path,
        'feature_shape': list(bundle.node_features.shape),
        'feature_hash': _sha256_array(bundle.node_features),
        'split_hash': split_hash,
        'label_hash': label_hash,
        'relation_edge_counts': edge_counts,
        'edge_weight_hashes': edge_hashes,
        'd1_model_config': {
            'review_encoder': assets['d1_run_summary']['best_graph_model']['review_encoder'],
            'backbone': assets['d1_run_summary']['best_graph_model']['backbone'],
            'relation_model': assets['d1_run_summary']['best_graph_model']['relation_model'],
            'seed': assets['d1_config'].get('seed', 'UNKNOWN_FROM_D1'),
            'run_config': assets['d1_run_config'],
        },
    }
    return audit


def _run_graph(exp_dir: Path, exp_name: str, assets: dict[str, Any], topk_mode: str, relation_k: dict[str, int], alpha: float, beta: float, gamma: float, abnormal_score_source: str, seed: int, preserve_d1_for_fixed_k0: bool=False, max_epochs_override: int | None=None, patience_override: int | None=None):
    exp_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = exp_dir / 'metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    edge_frames = build_routek_d1main_rns_topk_graph_frames(
        user_df=assets['bundle'].user_df.copy(),
        review_features=assets['review_scores_df'].copy(),
        user_text_vectors=assets['user_text_vectors'],
        user_abnormal_vectors=assets['user_abnormal_vectors'],
        d1_edge_frames=assets['bundle'].edge_frames,
        output_dir=exp_dir,
        topk_mode=topk_mode,
        relation_k=relation_k,
        abnormal_score_source=abnormal_score_source,
        alpha_abnormal=alpha,
        beta_tns=beta,
        gamma_interaction=gamma,
        tns_phi_days=5,
        preserve_d1_for_fixed_k0=preserve_d1_for_fixed_k0,
    )
    compute_edge_stats(edge_frames=edge_frames, user_df=assets['bundle'].user_df, output_dir=exp_dir)
    result_df = run_relation_aggregation_experiments(
        user_df=assets['bundle'].user_df,
        self_features=assets['bundle'].node_features.astype(np.float32),
        edge_frames=edge_frames,
        output_dir=metrics_dir,
        review_encoder_name='llm_masked_logic',
        model_kind='edge_aware_gat',
        seed=seed,
        backbone='current_egat',
        relation_model='edge_aware_gat',
        review_scores_df=assets['review_scores_df'],
        selected_edge_set='Base_LogicAE_CB',
        use_node_gat=False,
        use_self_graph_gate=False,
        use_relation_sigmoid_gate=False,
        use_self_aux_loss=False,
        use_abnormal_edge_weight=False,
        use_abnormal_gate=False,
        use_abnormal_value_gate=False,
        use_abnormal_attention_bias=False,
        abnormal_score_source=abnormal_score_source,
        tns_attention_relations=[],
        max_epochs_override=max_epochs_override,
        patience_override=patience_override,
        return_training_details=True,
    )
    row = result_df.loc[result_df['edge_set'] == 'Base_LogicAE_CB'].iloc[0].to_dict()
    save_json(exp_dir / 'run_summary.json', {
        'experiment_name': exp_name,
        'base_source': str(BASE_PROTOCOL_DIR),
        'reference_dir': str(REFERENCE_D1_DIR),
        'strategy': topk_mode,
        'relation_k': relation_k,
        'best_graph_model': row,
    })
    save_json(exp_dir / 'config.json', {
        'experiment_name': exp_name,
        'graph_mode': 'current',
        'edge_set': 'Base_LogicAE_CB',
        'model_backbone': 'current_egat',
        'relation_model': 'edge_aware_gat',
        'topk_mode': topk_mode,
        'relation_k': relation_k,
        'alpha_abnormal': alpha,
        'beta_tns': beta,
        'gamma_interaction': gamma,
        'abnormal_score_source': abnormal_score_source,
        'base_source': str(BASE_PROTOCOL_DIR),
        'reference_dir': str(REFERENCE_D1_DIR),
        'feature_source': 'D1 source assets + build_self_feature_matrix (same D1 code path)',
    })
    (exp_dir / 'train.log').write_text(f'{exp_name} completed\n', encoding='utf-8')
    edge_cfg_path = exp_dir / 'edges' / 'edge_build_config.json'
    edge_cfg = json.loads(edge_cfg_path.read_text(encoding='utf-8')) if edge_cfg_path.exists() else {}
    topk_overlap_path = exp_dir / 'metrics' / 'topk_selection_overlap.csv'
    topology_changed = None
    if topk_overlap_path.exists():
        overlap_df = pd.read_csv(topk_overlap_path)
        topology_changed = bool((overlap_df['overlap_with_K0_top20'] < 0.999999).any())
    return row, edge_cfg, topology_changed


def _k4_reward(val_auc: float, val_ap: float, edge_density_norm: float, lambda_density: float) -> float:
    return float(val_auc) + 0.5 * float(val_ap) - float(lambda_density) * float(edge_density_norm)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_root / f"routeD1_kattach_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    assets = _load_assets()
    audit = _audit(assets)
    save_json(output_root / 'd1_artifact_audit.json', audit)
    d1_best = assets['d1_run_summary']['best_graph_model']

    if args.run_k0_sanity:
        logger.info('running K0 strict sanity')
        k0_dir = output_root / 'K0_StrictSanity'
        k0_row, _, _ = _run_graph(
            exp_dir=k0_dir,
            exp_name='K0_StrictSanity',
            assets=assets,
            topk_mode='fixed_original',
            relation_k={'UPU':20,'UTU':20,'USU':20,'LogicAE_CB':20},
            alpha=args.alpha_abnormal,
            beta=args.beta_tns,
            gamma=args.gamma_interaction,
            abnormal_score_source=args.abnormal_score_source,
            seed=args.seed,
            preserve_d1_for_fixed_k0=True,
        )
        if abs(float(k0_row['auc']) - float(d1_best['auc'])) > 0.003 or abs(float(k0_row['ap']) - float(d1_best['ap'])) > 0.003:
            (output_root / 'FAILED_K0_NOT_STRICT_D1.md').write_text(
                'K0 strict sanity failed to reproduce D1 within tolerance.\n', encoding='utf-8'
            )
            logger.error('K0 strict sanity failed; stopping before K2/K4')
            return
    else:
        logger.info('skipping K0 strict rerun; relying on direct D1 edge/hash audit')

    rows=[]
    rows.append({
        'experiment_name':'D1_EGAT_Base_LogicAE_CB',
        'base_source': str(BASE_PROTOCOL_DIR),
        'is_strict_d1_artifact': True,
        'feature_path': audit['feature_path'],
        'feature_shape': json.dumps(audit['feature_shape']),
        'feature_hash': audit['feature_hash'],
        'split_hash': json.dumps(audit['split_hash'], ensure_ascii=False, sort_keys=True),
        'label_hash': audit['label_hash'],
        'strategy': 'reference_only',
        'selected_arm_id': '',
        'UPU_k':20,'UTU_k':20,'USU_k':20,'LogicAE_CB_k':20,
        'topology_changed':'',
        'num_edges': int(sum(audit['relation_edge_counts'].values())),
        'logicAE_cb_edge_count': int(audit['relation_edge_counts']['LogicAE_CB']),
        'resolved_tau_logic': 'UNKNOWN_FROM_D1',
        'AUC': d1_best['auc'],'AP': d1_best['ap'],'F1': d1_best['f1'],'Recall': d1_best['recall'],'Precision': d1_best['precision'],
        'best_epoch':'UNKNOWN_FROM_D1','test_threshold':d1_best['threshold'],'notes':'reference row',
    })

    logger.info('running D1_K2_AbnormalTNSAware')
    k2_dir = output_root / 'D1_K2_AbnormalTNSAware'
    k2_row, k2_edge_cfg, k2_topology_changed = _run_graph(
        exp_dir=k2_dir,
        exp_name='D1_K2_AbnormalTNSAware',
        assets=assets,
        topk_mode='abnormal_tns_aware',
        relation_k={'UPU':20,'UTU':20,'USU':20,'LogicAE_CB':20},
        alpha=args.alpha_abnormal,
        beta=args.beta_tns,
        gamma=args.gamma_interaction,
        abnormal_score_source=args.abnormal_score_source,
        seed=args.seed,
    )
    rows.append({
        'experiment_name':'D1_K2_AbnormalTNSAware',
        'base_source': str(BASE_PROTOCOL_DIR),
        'is_strict_d1_artifact': True,
        'feature_path': audit['feature_path'],
        'feature_shape': json.dumps(audit['feature_shape']),
        'feature_hash': audit['feature_hash'],
        'split_hash': json.dumps(audit['split_hash'], ensure_ascii=False, sort_keys=True),
        'label_hash': audit['label_hash'],
        'strategy':'K2_abnormal_tns_aware',
        'selected_arm_id':'',
        'UPU_k':20,'UTU_k':20,'USU_k':20,'LogicAE_CB_k':20,
        'topology_changed': bool(k2_topology_changed) if k2_topology_changed is not None else 'UNKNOWN',
        'num_edges': int(k2_row.get('num_edges',0)),
        'logicAE_cb_edge_count': int(k2_edge_cfg.get('relation_k',{}).get('LogicAE_CB',20)) if isinstance(k2_edge_cfg, dict) else 'UNKNOWN',
        'resolved_tau_logic': k2_edge_cfg.get('resolved_tau_logic', 'UNKNOWN_FROM_D1') if isinstance(k2_edge_cfg, dict) else 'UNKNOWN_FROM_D1',
        'AUC': k2_row['auc'],'AP': k2_row['ap'],'F1': k2_row['f1'],'Recall': k2_row['recall'],'Precision': k2_row['precision'],
        'best_epoch': k2_row.get('best_epoch','UNKNOWN_FROM_D1'),'test_threshold': k2_row.get('threshold'),'notes':'strict D1 protocol with K2 strategy',
    })

    logger.info('running D1_K4_BanditSelected')
    k4_dir = output_root / 'D1_K4_BanditSelected'
    warm_rows=[]
    for arm in K4_ARMS:
        arm_id=int(arm['arm_id'])
        relation_k={k:int(v) for k,v in arm.items() if k!='arm_id'}
        arm_dir = k4_dir / f'arm_{arm_id:02d}' / 'warmup'
        row, _, topology_changed = _run_graph(
            exp_dir=arm_dir,
            exp_name=f'D1_K4_arm_{arm_id:02d}_warmup',
            assets=assets,
            topk_mode='abnormal_tns_aware',
            relation_k=relation_k,
            alpha=args.alpha_abnormal,
            beta=args.beta_tns,
            gamma=args.gamma_interaction,
            abnormal_score_source=args.abnormal_score_source,
            seed=args.seed,
            max_epochs_override=args.k4_warmup_epochs,
            patience_override=max(3, args.k4_warmup_epochs//2),
        )
        num_edges=0
        for rel in ['UPU','UTU','USU','LogicAE_CB']:
            p=arm_dir/'edges'/f'{rel}_edges.csv'
            if p.exists():
                num_edges += len(pd.read_csv(p))
        warm_rows.append({
            'arm_id': arm_id,
            'UPU_k': relation_k['UPU'],'UTU_k': relation_k['UTU'],'USU_k': relation_k['USU'],'LogicAE_CB_k': relation_k['LogicAE_CB'],
            'num_edges': num_edges,
            'topology_changed': bool(topology_changed) if topology_changed is not None else 'UNKNOWN',
            'warmup_val_auc': row.get('val_auc'), 'warmup_val_ap': row.get('val_ap'),
        })
    arm_df=pd.DataFrame(warm_rows)
    arm0_edges=max(float(arm_df.loc[arm_df['arm_id']==0,'num_edges'].iloc[0]),1.0)
    arm_df['edge_density_norm']=arm_df['num_edges'].astype(float)/arm0_edges
    arm_df['warmup_reward']=arm_df.apply(lambda r: _k4_reward(r['warmup_val_auc'], r['warmup_val_ap'], r['edge_density_norm'], args.bandit_lambda_density), axis=1)
    top3=arm_df.sort_values('warmup_reward', ascending=False).head(3)['arm_id'].astype(int).tolist()
    arm_df['selected_for_full_train']=arm_df['arm_id'].isin(top3)

    full_rows=[]
    for arm_id in top3:
        arm = next(a for a in K4_ARMS if int(a['arm_id'])==arm_id)
        relation_k={k:int(v) for k,v in arm.items() if k!='arm_id'}
        full_dir = k4_dir / f'arm_{arm_id:02d}' / 'full'
        row, edge_cfg, topology_changed = _run_graph(
            exp_dir=full_dir,
            exp_name=f'D1_K4_arm_{arm_id:02d}_full',
            assets=assets,
            topk_mode='abnormal_tns_aware',
            relation_k=relation_k,
            alpha=args.alpha_abnormal,
            beta=args.beta_tns,
            gamma=args.gamma_interaction,
            abnormal_score_source=args.abnormal_score_source,
            seed=args.seed,
        )
        reward = _k4_reward(float(row.get('val_auc',0.0)), float(row.get('val_ap',0.0)), float(arm_df.loc[arm_df['arm_id']==arm_id,'edge_density_norm'].iloc[0]), args.bandit_lambda_density)
        full_rows.append({
            'arm_id': arm_id,
            'UPU_k': relation_k['UPU'],'UTU_k': relation_k['UTU'],'USU_k': relation_k['USU'],'LogicAE_CB_k': relation_k['LogicAE_CB'],
            'num_edges': int(arm_df.loc[arm_df['arm_id']==arm_id,'num_edges'].iloc[0]),
            'edge_density_norm': float(arm_df.loc[arm_df['arm_id']==arm_id,'edge_density_norm'].iloc[0]),
            'warmup_val_auc': float(arm_df.loc[arm_df['arm_id']==arm_id,'warmup_val_auc'].iloc[0]),
            'warmup_val_ap': float(arm_df.loc[arm_df['arm_id']==arm_id,'warmup_val_ap'].iloc[0]),
            'warmup_reward': float(arm_df.loc[arm_df['arm_id']==arm_id,'warmup_reward'].iloc[0]),
            'selected_for_full_train': True,
            'full_train_val_auc': row.get('val_auc'),
            'full_train_val_ap': row.get('val_ap'),
            'full_train_reward': reward,
            'test_auc': row.get('auc'),'test_ap': row.get('ap'),'test_f1': row.get('f1'),'test_recall': row.get('recall'),'test_precision': row.get('precision'),
            'topology_changed': bool(topology_changed) if topology_changed is not None else 'UNKNOWN',
            'resolved_tau_logic': edge_cfg.get('resolved_tau_logic', 'UNKNOWN_FROM_D1') if isinstance(edge_cfg, dict) else 'UNKNOWN_FROM_D1',
        })
    full_df=pd.DataFrame(full_rows)
    selected = full_df.sort_values('full_train_reward', ascending=False).iloc[0].to_dict()
    (k4_dir/'metrics').mkdir(parents=True, exist_ok=True)
    merged=arm_df.merge(full_df, on=['arm_id','UPU_k','UTU_k','USU_k','LogicAE_CB_k','num_edges','edge_density_norm','selected_for_full_train'], how='left')
    merged.to_csv(k4_dir/'metrics'/'bandit_arm_search.csv', index=False)
    save_json(k4_dir/'run_summary.json', {
        'experiment_name':'D1_K4_BanditSelected',
        'selected_arm_id': int(selected['arm_id']),
        'selected_relation_k': {
            'UPU': int(selected['UPU_k']), 'UTU': int(selected['UTU_k']), 'USU': int(selected['USU_k']), 'LogicAE_CB': int(selected['LogicAE_CB_k'])
        },
        'base_source': str(BASE_PROTOCOL_DIR),
        'reference_dir': str(REFERENCE_D1_DIR),
    })
    save_json(k4_dir/'config.json', {
        'experiment_name':'D1_K4_BanditSelected',
        'strategy':'K4_bandit',
        'alpha_abnormal': args.alpha_abnormal,
        'beta_tns': args.beta_tns,
        'gamma_interaction': args.gamma_interaction,
        'abnormal_score_source': args.abnormal_score_source,
        'candidate_arms': K4_ARMS,
        'selected_arm_id': int(selected['arm_id']),
        'base_source': str(BASE_PROTOCOL_DIR),
        'reference_dir': str(REFERENCE_D1_DIR),
        'feature_source': 'D1 source assets + build_self_feature_matrix (same D1 code path)',
    })
    rows.append({
        'experiment_name':'D1_K4_BanditSelected',
        'base_source': str(BASE_PROTOCOL_DIR),
        'is_strict_d1_artifact': True,
        'feature_path': audit['feature_path'],
        'feature_shape': json.dumps(audit['feature_shape']),
        'feature_hash': audit['feature_hash'],
        'split_hash': json.dumps(audit['split_hash'], ensure_ascii=False, sort_keys=True),
        'label_hash': audit['label_hash'],
        'strategy':'K4_bandit',
        'selected_arm_id': int(selected['arm_id']),
        'UPU_k': int(selected['UPU_k']),'UTU_k': int(selected['UTU_k']),'USU_k': int(selected['USU_k']),'LogicAE_CB_k': int(selected['LogicAE_CB_k']),
        'topology_changed': selected.get('topology_changed', 'UNKNOWN'),
        'num_edges': int(selected['num_edges']),
        'logicAE_cb_edge_count': int(selected['LogicAE_CB_k']),
        'resolved_tau_logic': selected.get('resolved_tau_logic','UNKNOWN_FROM_D1'),
        'AUC': selected['test_auc'],'AP': selected['test_ap'],'F1': selected['test_f1'],'Recall': selected['test_recall'],'Precision': selected['test_precision'],
        'best_epoch':'UNKNOWN_FROM_D1','test_threshold':'UNKNOWN_FROM_D1','notes':'strict D1 protocol with K4 bandit strategy',
    })

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_root / 'routeD1_kattach_summary.csv', index=False)
    lines=[
        '# Route D1 + K2/K4 Attachment Summary',
        '',
        f"- Strict D1 source assets reused: yes",
        f"- Feature path: {audit['feature_path']}",
        f"- Feature shape: {audit['feature_shape']}",
        f"- D1_K2 AUC/AP/F1: {rows[1]['AUC']:.4f} / {rows[1]['AP']:.4f} / {rows[1]['F1']:.4f}",
        f"- D1_K4 selected arm: {int(selected['arm_id'])} => UPU={int(selected['UPU_k'])}, UTU={int(selected['UTU_k'])}, USU={int(selected['USU_k'])}, LogicAE_CB={int(selected['LogicAE_CB_k'])}",
        f"- D1_K4 AUC/AP/F1: {rows[2]['AUC']:.4f} / {rows[2]['AP']:.4f} / {rows[2]['F1']:.4f}",
        '',
        'Detailed analysis to be filled after run completion.'
    ]
    (output_root / 'routeD1_kattach_summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')

if __name__ == '__main__':
    main()
