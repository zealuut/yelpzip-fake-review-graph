from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from graph.scripts.route_runner import _load_base_artifacts
from graph.graph_pipeline import (
    build_routek_d1main_rns_topk_graph_frames,
    build_self_feature_matrix,
    compute_edge_stats,
)
from graph.relation_model import run_relation_aggregation_experiments

MAIN_PROJECT_ROOT = Path("/home/xyz/HuChao (2)/Bert-TextClassification")
BASE_PROTOCOL_DIR = MAIN_PROJECT_ROOT / "graph" / "outputs" / "yelpzip_balanced_current_graph_no_reweight_20260502_160620"
REFERENCE_D1_DIR = MAIN_PROJECT_ROOT / "graph" / "outputs" / "routeD_tns_guided_logic_egat_20260504_200855" / "D1_EGAT_Base_LogicAE_CB"

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha_abnormal", type=float, default=0.5)
    parser.add_argument("--beta_tns", type=float, default=0.2)
    parser.add_argument("--gamma_interaction", type=float, default=0.2)
    parser.add_argument("--abnormal_score_source", default="auto")
    parser.add_argument("--k4_warmup_epochs", type=int, default=15)
    parser.add_argument("--bandit_lambda_density", type=float, default=0.02)
    return parser.parse_args()


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_assets():
    artifacts = _load_base_artifacts(BASE_PROTOCOL_DIR)
    d1_summary = json.loads((REFERENCE_D1_DIR / "run_summary.json").read_text(encoding="utf-8"))
    user_df = artifacts["user_df"].copy()
    user_text_vectors = artifacts["user_text_vectors"]
    user_abnormal_vectors = artifacts["user_abnormal_vectors"]
    review_scores_df = artifacts["review_scores_df"].copy()
    self_features = build_self_feature_matrix(user_df, user_abnormal_vectors)
    d1_edge_frames = {}
    for relation in ["UPU", "UTU", "USU", "LogicAE_CB"]:
        d1_edge_frames[relation] = pd.read_csv(REFERENCE_D1_DIR / "edges" / f"{relation}_edges.csv")
    return {
        "user_df": user_df,
        "user_text_vectors": user_text_vectors,
        "user_abnormal_vectors": user_abnormal_vectors,
        "review_scores_df": review_scores_df,
        "self_features": self_features,
        "d1_edge_frames": d1_edge_frames,
        "d1_best": d1_summary["best_graph_model"],
    }


def _reward(val_auc: float, val_ap: float, edge_density_norm: float, lambda_density: float) -> float:
    return float(val_auc) + 0.5 * float(val_ap) - float(lambda_density) * float(edge_density_norm)


def _run_graph(
    *,
    exp_dir: Path,
    assets: dict,
    experiment_name: str,
    topk_mode: str,
    relation_k: dict[str, int],
    alpha: float,
    beta: float,
    gamma: float,
    abnormal_score_source: str,
    seed: int,
    max_epochs_override: int | None = None,
    patience_override: int | None = None,
):
    exp_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = exp_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    edge_frames = build_routek_d1main_rns_topk_graph_frames(
        user_df=assets["user_df"].copy(),
        review_features=assets["review_scores_df"].copy(),
        user_text_vectors=assets["user_text_vectors"],
        user_abnormal_vectors=assets["user_abnormal_vectors"],
        d1_edge_frames=assets["d1_edge_frames"],
        output_dir=exp_dir,
        topk_mode=topk_mode,
        relation_k=relation_k,
        abnormal_score_source=abnormal_score_source,
        alpha_abnormal=alpha,
        beta_tns=beta,
        gamma_interaction=gamma,
        tns_phi_days=5,
    )
    compute_edge_stats(edge_frames=edge_frames, user_df=assets["user_df"], output_dir=exp_dir)

    result_df = run_relation_aggregation_experiments(
        user_df=assets["user_df"],
        self_features=assets["self_features"],
        edge_frames=edge_frames,
        output_dir=metrics_dir,
        review_encoder_name="llm_masked_logic",
        model_kind="edge_aware_gat",
        seed=seed,
        backbone="current_egat",
        relation_model="edge_aware_gat",
        use_abnormal_edge_weight=False,
        use_abnormal_gate=False,
        use_abnormal_value_gate=False,
        use_abnormal_attention_bias=False,
        abnormal_score_source=abnormal_score_source,
        abnormal_edge_lambda=1.0,
        abnormal_edge_eta=0.5,
        abnormal_gate_eta=0.5,
        abnormal_pair_mode="both_high",
        abnormal_gate_learnable=False,
        abnormal_attention_gamma=1.0,
        review_scores_df=assets["review_scores_df"],
        selected_edge_set="Base_LogicAE_CB",
        relation_topk=None,
        use_node_gat=False,
        max_epochs_override=max_epochs_override,
        patience_override=patience_override,
    )
    row = result_df.loc[result_df["edge_set"] == "Base_LogicAE_CB"].iloc[0].to_dict()
    _save_json(
        exp_dir / "run_summary.json",
        {
            "experiment_name": experiment_name,
            "implementation": "ce6a9d6-kattach-fix",
            "best_graph_model": row,
            "relation_k": relation_k,
            "strategy": topk_mode,
        },
    )
    _save_json(
        exp_dir / "config.json",
        {
            "experiment_name": experiment_name,
            "graph_mode": "current",
            "edge_set": "Base_LogicAE_CB",
            "model_backbone": "current_egat",
            "relation_model": "edge_aware_gat",
            "topk_mode": topk_mode,
            "relation_k": relation_k,
            "alpha_abnormal": alpha,
            "beta_tns": beta,
            "gamma_interaction": gamma,
            "abnormal_score_source": abnormal_score_source,
            "implementation": "ce6a9d6-kattach-fix",
        },
    )
    return row


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    assets = _load_assets()

    rows = [
        {
            "experiment_name": "D1_EGAT_Base_LogicAE_CB",
            "strategy": "reference_only",
            "selected_arm_id": "",
            "UPU_k": 20,
            "UTU_k": 20,
            "USU_k": 20,
            "LogicAE_CB_k": 20,
            "AUC": assets["d1_best"]["auc"],
            "AP": assets["d1_best"]["ap"],
            "F1": assets["d1_best"]["f1"],
            "Recall": assets["d1_best"]["recall"],
            "Precision": assets["d1_best"]["precision"],
            "test_threshold": assets["d1_best"]["threshold"],
            "notes": "reference row",
        }
    ]

    k2_dir = output_root / "D1_K2_AbnormalTNSAware"
    k2_row = _run_graph(
        exp_dir=k2_dir,
        assets=assets,
        experiment_name="D1_K2_AbnormalTNSAware",
        topk_mode="abnormal_tns_aware",
        relation_k={"UPU": 20, "UTU": 20, "USU": 20, "LogicAE_CB": 20},
        alpha=args.alpha_abnormal,
        beta=args.beta_tns,
        gamma=args.gamma_interaction,
        abnormal_score_source=args.abnormal_score_source,
        seed=args.seed,
    )
    rows.append(
        {
            "experiment_name": "D1_K2_AbnormalTNSAware",
            "strategy": "K2_abnormal_tns_aware",
            "selected_arm_id": "",
            "UPU_k": 20,
            "UTU_k": 20,
            "USU_k": 20,
            "LogicAE_CB_k": 20,
            "AUC": k2_row["auc"],
            "AP": k2_row["ap"],
            "F1": k2_row["f1"],
            "Recall": k2_row["recall"],
            "Precision": k2_row["precision"],
            "test_threshold": k2_row["threshold"],
            "notes": "ce6a9d6-kattach-fix",
        }
    )

    k4_dir = output_root / "D1_K4_BanditSelected"
    warm_rows = []
    for arm in K4_ARMS:
        arm_id = int(arm["arm_id"])
        relation_k = {k: int(v) for k, v in arm.items() if k != "arm_id"}
        arm_dir = k4_dir / f"arm_{arm_id:02d}" / "warmup"
        row = _run_graph(
            exp_dir=arm_dir,
            assets=assets,
            experiment_name=f"D1_K4_arm_{arm_id:02d}_warmup",
            topk_mode="abnormal_tns_aware",
            relation_k=relation_k,
            alpha=args.alpha_abnormal,
            beta=args.beta_tns,
            gamma=args.gamma_interaction,
            abnormal_score_source=args.abnormal_score_source,
            seed=args.seed,
            max_epochs_override=args.k4_warmup_epochs,
            patience_override=max(3, args.k4_warmup_epochs // 2),
        )
        num_edges = 0
        for relation in ["UPU", "UTU", "USU", "LogicAE_CB"]:
            path = arm_dir / "edges" / f"{relation}_edges.csv"
            if path.exists():
                num_edges += len(pd.read_csv(path))
        warm_rows.append(
            {
                "arm_id": arm_id,
                "UPU_k": relation_k["UPU"],
                "UTU_k": relation_k["UTU"],
                "USU_k": relation_k["USU"],
                "LogicAE_CB_k": relation_k["LogicAE_CB"],
                "num_edges": num_edges,
                "warmup_val_auc": row.get("val_auc"),
                "warmup_val_ap": row.get("val_ap"),
            }
        )

    arm_df = pd.DataFrame(warm_rows)
    arm0_edges = max(float(arm_df.loc[arm_df["arm_id"] == 0, "num_edges"].iloc[0]), 1.0)
    arm_df["edge_density_norm"] = arm_df["num_edges"].astype(float) / arm0_edges
    arm_df["warmup_reward"] = arm_df.apply(
        lambda r: _reward(r["warmup_val_auc"], r["warmup_val_ap"], r["edge_density_norm"], args.bandit_lambda_density),
        axis=1,
    )
    top3 = arm_df.sort_values("warmup_reward", ascending=False).head(3)["arm_id"].astype(int).tolist()
    arm_df["selected_for_full_train"] = arm_df["arm_id"].isin(top3)

    full_rows = []
    for arm_id in top3:
        arm = next(item for item in K4_ARMS if int(item["arm_id"]) == arm_id)
        relation_k = {k: int(v) for k, v in arm.items() if k != "arm_id"}
        full_dir = k4_dir / f"arm_{arm_id:02d}" / "full"
        row = _run_graph(
            exp_dir=full_dir,
            assets=assets,
            experiment_name=f"D1_K4_arm_{arm_id:02d}_full",
            topk_mode="abnormal_tns_aware",
            relation_k=relation_k,
            alpha=args.alpha_abnormal,
            beta=args.beta_tns,
            gamma=args.gamma_interaction,
            abnormal_score_source=args.abnormal_score_source,
            seed=args.seed,
        )
        reward = _reward(
            float(row.get("val_auc", 0.0)),
            float(row.get("val_ap", 0.0)),
            float(arm_df.loc[arm_df["arm_id"] == arm_id, "edge_density_norm"].iloc[0]),
            args.bandit_lambda_density,
        )
        full_rows.append(
            {
                "arm_id": arm_id,
                "UPU_k": relation_k["UPU"],
                "UTU_k": relation_k["UTU"],
                "USU_k": relation_k["USU"],
                "LogicAE_CB_k": relation_k["LogicAE_CB"],
                "num_edges": int(arm_df.loc[arm_df["arm_id"] == arm_id, "num_edges"].iloc[0]),
                "edge_density_norm": float(arm_df.loc[arm_df["arm_id"] == arm_id, "edge_density_norm"].iloc[0]),
                "warmup_val_auc": float(arm_df.loc[arm_df["arm_id"] == arm_id, "warmup_val_auc"].iloc[0]),
                "warmup_val_ap": float(arm_df.loc[arm_df["arm_id"] == arm_id, "warmup_val_ap"].iloc[0]),
                "warmup_reward": float(arm_df.loc[arm_df["arm_id"] == arm_id, "warmup_reward"].iloc[0]),
                "selected_for_full_train": True,
                "full_train_val_auc": row.get("val_auc"),
                "full_train_val_ap": row.get("val_ap"),
                "full_train_reward": reward,
                "test_auc": row.get("auc"),
                "test_ap": row.get("ap"),
                "test_f1": row.get("f1"),
                "test_recall": row.get("recall"),
                "test_precision": row.get("precision"),
            }
        )

    full_df = pd.DataFrame(full_rows)
    selected = full_df.sort_values("full_train_reward", ascending=False).iloc[0].to_dict()
    (k4_dir / "metrics").mkdir(parents=True, exist_ok=True)
    arm_df.merge(
        full_df,
        on=[
            "arm_id",
            "UPU_k",
            "UTU_k",
            "USU_k",
            "LogicAE_CB_k",
            "num_edges",
            "edge_density_norm",
            "selected_for_full_train",
        ],
        how="left",
    ).to_csv(k4_dir / "metrics" / "bandit_arm_search.csv", index=False)
    _save_json(
        k4_dir / "run_summary.json",
        {
            "experiment_name": "D1_K4_BanditSelected",
            "implementation": "ce6a9d6-kattach-fix",
            "selected_arm_id": int(selected["arm_id"]),
            "selected_relation_k": {
                "UPU": int(selected["UPU_k"]),
                "UTU": int(selected["UTU_k"]),
                "USU": int(selected["USU_k"]),
                "LogicAE_CB": int(selected["LogicAE_CB_k"]),
            },
        },
    )
    _save_json(
        k4_dir / "config.json",
        {
            "experiment_name": "D1_K4_BanditSelected",
            "strategy": "K4_bandit",
            "implementation": "ce6a9d6-kattach-fix",
            "selected_arm_id": int(selected["arm_id"]),
        },
    )
    rows.append(
        {
            "experiment_name": "D1_K4_BanditSelected",
            "strategy": "K4_bandit",
            "selected_arm_id": int(selected["arm_id"]),
            "UPU_k": int(selected["UPU_k"]),
            "UTU_k": int(selected["UTU_k"]),
            "USU_k": int(selected["USU_k"]),
            "LogicAE_CB_k": int(selected["LogicAE_CB_k"]),
            "AUC": selected["test_auc"],
            "AP": selected["test_ap"],
            "F1": selected["test_f1"],
            "Recall": selected["test_recall"],
            "Precision": selected["test_precision"],
            "test_threshold": "UNKNOWN_FROM_D1",
            "notes": "ce6a9d6-kattach-fix",
        }
    )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_root / "routeD1_kattach_summary.csv", index=False)
    (output_root / "routeD1_kattach_summary.md").write_text(summary_df.to_csv(index=False), encoding="utf-8")


if __name__ == "__main__":
    main()
