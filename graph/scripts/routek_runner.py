from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from graph.graph_pipeline import build_routek_adaptive_topk_graph_frames, compute_edge_stats
from graph.relation_model import run_relation_aggregation_experiments
from graph.baseline_comparison.src.data_loader import load_protocol_bundle
from graph.baseline_comparison.src.utils import save_json, save_yaml, setup_logger


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GRAPH_DIR = PROJECT_ROOT / "graph"
BASE_PROTOCOL_DIR = GRAPH_DIR / "outputs" / "yelpzip_balanced_current_graph_no_reweight_20260502_160620"
REFERENCE_D1_DIR = GRAPH_DIR / "outputs" / "routeD_tns_guided_logic_egat_20260504_200855" / "D1_EGAT_Base_LogicAE_CB"


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
    parser = argparse.ArgumentParser(description="Run Route K adaptive top-k graph denoising experiments.")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha_abnormal", type=float, default=0.5)
    parser.add_argument("--beta_tns", type=float, default=0.2)
    parser.add_argument("--gamma_interaction", type=float, default=0.2)
    parser.add_argument("--abnormal_score_source", default="auto")
    parser.add_argument("--bandit_warmup_epochs", type=int, default=15)
    parser.add_argument("--bandit_lambda_density", type=float, default=0.02)
    parser.add_argument("--detached_run", action="store_true", default=False)
    return parser.parse_args()


def _load_base_artifacts() -> dict[str, Any]:
    bundle = load_protocol_bundle()
    review_scores_df = pd.read_csv(BASE_PROTOCOL_DIR / "review_scores_enriched.csv")
    user_text_vectors = np.load(BASE_PROTOCOL_DIR / "logic_vectors" / "user_text_vectors.npy")
    user_abnormal_vectors = np.load(BASE_PROTOCOL_DIR / "logic_vectors" / "user_abnormal_vectors.npy")
    reference_payload = json.loads((REFERENCE_D1_DIR / "run_summary.json").read_text(encoding="utf-8"))
    return {
        "bundle": bundle,
        "review_scores_df": review_scores_df,
        "user_text_vectors": user_text_vectors,
        "user_abnormal_vectors": user_abnormal_vectors,
        "reference_payload": reference_payload,
    }


def _best_graph_row(result_df: pd.DataFrame, edge_set: str, metric: str = "auc") -> dict[str, Any]:
    if result_df.empty:
        return {}
    candidates = result_df[result_df["edge_set"] == edge_set].copy()
    if candidates.empty:
        return {}
    metric = str(metric or "auc")
    if metric not in candidates.columns:
        metric = "auc"
    return candidates.sort_values(metric, ascending=False).iloc[0].to_dict()


def _copy_selected_artifacts(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in ["run_summary.json", "config.json", "run_config.json", "train.log"]:
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, dst_dir / name)
    for rel in ["metrics/model_results.csv", "metrics/epoch_metrics.csv", "metrics/test_predictions.csv", "metrics/edge_stats.csv", "metrics/topk_edge_quality_by_relation.csv", "metrics/topk_rank_score_stats.csv", "metrics/topk_relation_degree_stats.csv", "edges/edge_build_config.json"]:
        src = src_dir / rel
        if src.exists():
            dst = dst_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def _train_one_experiment(
    *,
    exp_dir: Path,
    exp_name: str,
    topk_mode: str,
    relation_k: dict[str, int],
    base_artifacts: dict[str, Any],
    alpha_abnormal: float,
    beta_tns: float,
    gamma_interaction: float,
    abnormal_score_source: str,
    seed: int,
    warmup_epochs: int | None = None,
    bandit_mode: bool = False,
) -> tuple[dict[str, Any], Path]:
    metrics_dir = exp_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    exp_dir.mkdir(parents=True, exist_ok=True)

    edge_frames = build_routek_adaptive_topk_graph_frames(
        user_df=base_artifacts["bundle"].user_df.copy(),
        review_features=base_artifacts["review_scores_df"],
        user_text_vectors=base_artifacts["user_text_vectors"],
        user_abnormal_vectors=base_artifacts["user_abnormal_vectors"],
        output_dir=exp_dir,
        topk_mode=topk_mode,
        relation_k=relation_k,
        alpha_abnormal=alpha_abnormal,
        beta_tns=beta_tns,
        gamma_interaction=gamma_interaction,
        tns_phi_days=5,
        abnormal_score_source=abnormal_score_source,
    )
    edge_stats = compute_edge_stats(edge_frames=edge_frames, user_df=base_artifacts["bundle"].user_df, output_dir=exp_dir)
    self_features = base_artifacts["bundle"].node_features.astype(np.float32)

    train_kwargs = {
        "user_df": base_artifacts["bundle"].user_df,
        "self_features": self_features,
        "edge_frames": edge_frames,
        "output_dir": metrics_dir,
        "review_encoder_name": "d1_current_features",
        "model_kind": "edge_aware_gat",
        "seed": seed,
        "backbone": "current_egat",
        "relation_model": "edge_aware_gat",
        "review_scores_df": base_artifacts["review_scores_df"],
        "selected_edge_set": "Base_LogicAE_CB",
        "use_node_gat": False,
        "use_self_graph_gate": False,
        "use_relation_sigmoid_gate": False,
        "use_self_aux_loss": False,
        "use_abnormal_edge_weight": False,
        "use_abnormal_gate": False,
        "use_abnormal_value_gate": False,
        "use_abnormal_attention_bias": False,
        "abnormal_score_source": abnormal_score_source,
        "abnormal_edge_lambda": 1.0,
        "abnormal_edge_eta": 0.5,
        "abnormal_gate_eta": 0.5,
        "abnormal_pair_mode": "both_high",
        "abnormal_gate_learnable": False,
        "abnormal_attention_gamma": 1.0,
        "relation_topk": None,
        "return_training_details": True,
    }
    if warmup_epochs is not None and bandit_mode:
        train_kwargs["max_epochs_override"] = int(warmup_epochs)
        train_kwargs["patience_override"] = max(3, int(warmup_epochs) // 2)
    result_df = run_relation_aggregation_experiments(**train_kwargs)
    graph_row = _best_graph_row(result_df, "Base_LogicAE_CB", metric="val_auc" if bandit_mode else "auc")
    exp_config = {
        "experiment_name": exp_name,
        "graph_protocol": "current_topk",
        "graph_mode": "current",
        "edge_set": "Base_LogicAE_CB",
        "model": "EGAT",
        "model_backbone": "current_egat",
        "relation_model": "edge_aware_gat",
        "topk_mode": topk_mode,
        "relation_k": relation_k,
        "alpha_abnormal": float(alpha_abnormal),
        "beta_tns": float(beta_tns),
        "gamma_interaction": float(gamma_interaction),
        "abnormal_score_source": abnormal_score_source,
        "seed": int(seed),
        "feature_source": "D1 current node features from current-topk protocol",
        "feature_dim": int(self_features.shape[1]),
        "num_users": int(base_artifacts["bundle"].user_df.shape[0]),
        "num_edges": int(edge_stats["num_edges"].sum()),
        "notes": "Route K adaptive top-k graph denoising; current top-k graph only.",
    }
    if graph_row:
        exp_config["best_metrics"] = graph_row
        exp_config["AUC"] = graph_row.get("auc")
        exp_config["AP"] = graph_row.get("ap")
        exp_config["val_auc"] = graph_row.get("val_auc")
        exp_config["val_ap"] = graph_row.get("val_ap")
        exp_config["F1"] = graph_row.get("f1")
        exp_config["Recall"] = graph_row.get("recall")
        exp_config["Precision"] = graph_row.get("precision")
        exp_config["best_epoch"] = graph_row.get("best_epoch")
        exp_config["test_threshold"] = graph_row.get("threshold")
    save_json(exp_dir / "config.json", exp_config)
    save_json(exp_dir / "run_config.json", exp_config)
    save_json(exp_dir / "run_summary.json", {"config": exp_config, "best_graph_model": graph_row, "reference": base_artifacts["reference_payload"]})
    (exp_dir / "train.log").write_text(
        "\n".join(
            [
                f"experiment={exp_name}",
                f"topk_mode={topk_mode}",
                f"relation_k={relation_k}",
                f"auc={graph_row.get('auc') if graph_row else None}",
                f"ap={graph_row.get('ap') if graph_row else None}",
                f"f1={graph_row.get('f1') if graph_row else None}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "experiment_name": exp_name,
        "output_dir": str(exp_dir),
        "model": "EGAT",
        "graph_protocol": "current_topk",
        "edge_set": "Base_LogicAE_CB",
        "relation_handling": "typed_edge_aware_egat",
        "feature_source": exp_config["feature_source"],
        "num_users": exp_config["num_users"],
        "num_edges": exp_config["num_edges"],
        "hidden_dim": 144,
        "num_layers": 1,
        "heads": "UNKNOWN_FROM_D1",
        "num_bases": "UNKNOWN_FROM_D1",
        "use_neighbor_sampling": False,
        "optimizer": "AdamW",
        "lr": 0.001,
        "weight_decay": 0.0005,
        "dropout": 0.2,
        "epochs": 100 if warmup_epochs is None else warmup_epochs,
        "patience": 16 if warmup_epochs is None else max(3, int(warmup_epochs) // 2),
        "AUC": graph_row.get("auc") if graph_row else None,
        "AP": graph_row.get("ap") if graph_row else None,
        "val_auc": graph_row.get("val_auc") if graph_row else None,
        "val_ap": graph_row.get("val_ap") if graph_row else None,
        "F1": graph_row.get("f1") if graph_row else None,
        "Recall": graph_row.get("recall") if graph_row else None,
        "Precision": graph_row.get("precision") if graph_row else None,
        "best_epoch": graph_row.get("best_epoch") if graph_row else None,
        "test_threshold": graph_row.get("threshold") if graph_row else None,
        "notes": exp_config["notes"],
        "topk_mode": topk_mode,
        "relation_k": relation_k,
    }, exp_dir


def _route_k_rows() -> list[dict[str, Any]]:
    return [
        {
            "experiment_name": "K0_FixedTopK20_EGAT_Base_LogicAE_CB",
            "topk_mode": "fixed_original",
            "relation_k": {"UPU": 20, "UTU": 20, "USU": 20, "LogicAE_CB": 20},
            "use_bandit": False,
        },
        {
            "experiment_name": "K1_AbnormalAwareTopK_EGAT_Base_LogicAE_CB",
            "topk_mode": "abnormal_aware",
            "relation_k": {"UPU": 20, "UTU": 20, "USU": 20, "LogicAE_CB": 20},
            "use_bandit": False,
        },
        {
            "experiment_name": "K2_AbnormalTNSAwareTopK_EGAT_Base_LogicAE_CB",
            "topk_mode": "abnormal_tns_aware",
            "relation_k": {"UPU": 20, "UTU": 20, "USU": 20, "LogicAE_CB": 20},
            "use_bandit": False,
        },
        {
            "experiment_name": "K3_HandRelationWiseTopK_EGAT_Base_LogicAE_CB",
            "topk_mode": "abnormal_tns_aware",
            "relation_k": {"UPU": 10, "UTU": 10, "USU": 10, "LogicAE_CB": 30},
            "use_bandit": False,
        },
        {
            "experiment_name": "K4_BanditSelected_AbnormalTNSAwareTopK_EGAT_Base_LogicAE_CB",
            "topk_mode": "abnormal_tns_aware",
            "relation_k": {"UPU": 20, "UTU": 20, "USU": 20, "LogicAE_CB": 20},
            "use_bandit": True,
        },
    ]


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_root / f"routeK_{timestamp}.log")

    artifacts = _load_base_artifacts()
    route_rows: list[dict[str, Any]] = []

    bandit_rows: list[dict[str, Any]] = []
    selected_arm_id = None
    selected_arm_k = None
    selected_arm_dir = None

    for spec in _route_k_rows():
        exp_name = spec["experiment_name"]
        logger.info("starting %s", exp_name)
        if not spec["use_bandit"]:
            exp_dir = output_root / exp_name
            row, exp_dir = _train_one_experiment(
                exp_dir=exp_dir,
                exp_name=exp_name,
                topk_mode=spec["topk_mode"],
                relation_k=spec["relation_k"],
                base_artifacts=artifacts,
                alpha_abnormal=args.alpha_abnormal,
                beta_tns=args.beta_tns,
                gamma_interaction=args.gamma_interaction,
                abnormal_score_source=args.abnormal_score_source,
                seed=args.seed,
            )
            route_rows.append(row)
            logger.info("%s done auc=%s ap=%s f1=%s", exp_name, row.get("AUC"), row.get("AP"), row.get("F1"))
            continue

        warmup_results = []
        arm_outputs: dict[int, Path] = {}
        for arm in K4_ARMS:
            arm_id = int(arm["arm_id"])
            arm_dir = output_root / exp_name / f"arm_{arm_id:02d}" / "warmup"
            arm_outputs[arm_id] = arm_dir
            result, _ = _train_one_experiment(
                exp_dir=arm_dir,
                exp_name=f"{exp_name}_arm_{arm_id:02d}_warmup",
                topk_mode=spec["topk_mode"],
                relation_k={k: int(v) for k, v in arm.items() if k != "arm_id"},
                base_artifacts=artifacts,
                alpha_abnormal=args.alpha_abnormal,
                beta_tns=args.beta_tns,
                gamma_interaction=args.gamma_interaction,
                abnormal_score_source=args.abnormal_score_source,
                seed=args.seed,
                warmup_epochs=args.bandit_warmup_epochs,
                bandit_mode=True,
            )
            val_reward = float(result.get("val_auc") or 0.0) + 0.5 * float(result.get("val_ap") or 0.0)
            warmup_results.append(
                {
                    "arm_id": arm_id,
                    "UPU_k": int(arm["UPU"]),
                    "UTU_k": int(arm["UTU"]),
                    "USU_k": int(arm["USU"]),
                    "LogicAE_CB_k": int(arm["LogicAE_CB"]),
                    "num_edges": int(result.get("num_edges") or 0),
                    "edge_density_norm": 1.0,
                    "warmup_val_auc": result.get("val_auc"),
                    "warmup_val_ap": result.get("val_ap"),
                    "warmup_reward": val_reward,
                    "selected_for_full_train": False,
                    "full_train_val_auc": None,
                    "full_train_val_ap": None,
                    "full_train_reward": None,
                    "test_auc": None,
                    "test_ap": None,
                    "test_f1": None,
                    "test_recall": None,
                    "test_precision": None,
                }
            )
        warmup_df = pd.DataFrame(warmup_results).sort_values("warmup_reward", ascending=False).reset_index(drop=True)
        arm0_match = warmup_df.loc[warmup_df["arm_id"] == 0, "num_edges"]
        arm0_edges = float(arm0_match.iloc[0]) if not arm0_match.empty else float(warmup_df["num_edges"].max() or 1.0)
        if arm0_edges <= 0:
            arm0_edges = 1.0
        warmup_df["edge_density_norm"] = warmup_df["num_edges"].astype(float) / arm0_edges
        warmup_df["warmup_reward"] = warmup_df["warmup_val_auc"].astype(float) + 0.5 * warmup_df["warmup_val_ap"].astype(float) - args.bandit_lambda_density * warmup_df["edge_density_norm"].astype(float)
        warmup_df = warmup_df.sort_values("warmup_reward", ascending=False).reset_index(drop=True)
        top3 = warmup_df.head(3)["arm_id"].astype(int).tolist()
        for arm_id in top3:
            arm = next(item for item in K4_ARMS if int(item["arm_id"]) == int(arm_id))
            exp_dir = output_root / exp_name / f"arm_{arm_id:02d}" / "full"
            result, _ = _train_one_experiment(
                exp_dir=exp_dir,
                exp_name=f"{exp_name}_arm_{arm_id:02d}_full",
                topk_mode=spec["topk_mode"],
                relation_k={k: int(v) for k, v in arm.items() if k != "arm_id"},
                base_artifacts=artifacts,
                alpha_abnormal=args.alpha_abnormal,
                beta_tns=args.beta_tns,
                gamma_interaction=args.gamma_interaction,
                abnormal_score_source=args.abnormal_score_source,
                seed=args.seed,
            )
            reward = (float(result.get("val_auc") or 0.0) + 0.5 * float(result.get("val_ap") or 0.0)) - args.bandit_lambda_density * (float(result.get("num_edges") or 0.0) / max(arm0_edges, 1.0))
            warmup_df.loc[warmup_df["arm_id"] == arm_id, ["selected_for_full_train", "full_train_val_auc", "full_train_val_ap", "full_train_reward", "test_auc", "test_ap", "test_f1", "test_recall", "test_precision"]] = [
                True,
                result.get("val_auc"),
                result.get("val_ap"),
                reward,
                result.get("AUC"),
                result.get("AP"),
                result.get("F1"),
                result.get("Recall"),
                result.get("Precision"),
            ]
        warmup_df.to_csv(output_root / exp_name / "bandit_arm_search.csv", index=False)
        selected = warmup_df.sort_values("full_train_reward", ascending=False).iloc[0].to_dict()
        selected_arm_id = int(selected["arm_id"])
        selected_arm_k = {k: int(selected[k]) for k in ["UPU_k", "UTU_k", "USU_k", "LogicAE_CB_k"]}
        selected_arm_dir = output_root / exp_name / f"arm_{selected_arm_id:02d}" / "full"
        final_summary = {
            "experiment_name": exp_name,
            "selected_arm_id": selected_arm_id,
            "selected_arm_k": selected_arm_k,
            "bandit_lambda_density": args.bandit_lambda_density,
        }
        save_json(output_root / exp_name / "run_summary.json", final_summary)
        save_yaml(output_root / exp_name / "config.yaml", final_summary)
        route_rows.append(
            {
                "experiment_name": exp_name,
                "topk_mode": spec["topk_mode"],
                "UPU_k": selected_arm_k["UPU_k"],
                "UTU_k": selected_arm_k["UTU_k"],
                "USU_k": selected_arm_k["USU_k"],
                "LogicAE_CB_k": selected_arm_k["LogicAE_CB_k"],
                "alpha_abnormal": args.alpha_abnormal,
                "beta_tns": args.beta_tns,
                "gamma_interaction": args.gamma_interaction,
                "use_bandit": True,
                "selected_arm_id": selected_arm_id,
                "num_edges": int(selected.get("num_edges", 0)),
                "AUC": selected.get("test_auc"),
                "AP": selected.get("test_ap"),
                "val_auc": selected.get("full_train_val_auc"),
                "val_ap": selected.get("full_train_val_ap"),
                "F1": selected.get("test_f1"),
                "Recall": selected.get("test_recall"),
                "Precision": selected.get("test_precision"),
                "best_epoch": "UNKNOWN_FROM_D1",
                "test_threshold": "UNKNOWN_FROM_D1",
                "notes": "Bandit-selected adaptive top-k result.",
            }
        )
        _copy_selected_artifacts(selected_arm_dir, output_root / exp_name)
        logger.info("%s done selected_arm=%s auc=%s ap=%s f1=%s", exp_name, selected_arm_id, selected.get("test_auc"), selected.get("test_ap"), selected.get("test_f1"))

    summary_df = pd.DataFrame(route_rows)
    summary_df.to_csv(output_root / "routeK_summary.csv", index=False)
    (output_root / "routeK_summary.md").write_text(summary_df.to_csv(index=False), encoding="utf-8")
    save_json(
        output_root / "routeK_run_summary.json",
        {
            "timestamp": timestamp,
            "seed": args.seed,
            "alpha_abnormal": args.alpha_abnormal,
            "beta_tns": args.beta_tns,
            "gamma_interaction": args.gamma_interaction,
            "abnormal_score_source": args.abnormal_score_source,
            "bandit_warmup_epochs": args.bandit_warmup_epochs,
            "bandit_lambda_density": args.bandit_lambda_density,
            "reference": {
                "experiment_name": "CurrentTopK_EGAT_Base_LogicAE_CB",
                "AUC": 0.85637,
                "AP": 0.85837,
                "F1": 0.77817,
                "Recall": 0.82309,
                "Precision": 0.73790,
                "source": str(REFERENCE_D1_DIR),
            },
        },
    )


if __name__ == "__main__":
    main()
