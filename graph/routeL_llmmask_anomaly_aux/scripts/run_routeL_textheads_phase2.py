from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
import traceback

import numpy as np
import pandas as pd
import torch

from graph.graph_pipeline import (
    build_edge_frames,
    build_review_and_user_artifacts,
    build_self_feature_matrix,
    compute_edge_stats,
)
from graph.relation_model import run_relation_aggregation_experiments

from graph.routeL_llmmask_anomaly_aux.src.export_user_features_routeL import export_review_feature_frame
from graph.routeL_llmmask_anomaly_aux.src.review_training_routeL_text import (
    build_psycholinguistic_style_frame,
    build_routeL_dataloaders,
    build_routeL_text_model,
    build_semantic_drift_frame,
    compute_review_text_embeddings,
    encode_routeL_all_reviews,
    load_routeL_review_frames,
    make_llm_mask_stats_from_review_scores,
    train_routeL_text_encoder,
    write_feature_hash,
)
from graph.routeL_llmmask_anomaly_aux.src.routeL_utils import (
    ensure_dir,
    json_dump,
    load_d1_bundle,
    load_yaml_config,
    project_root_from_here,
)


def _sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _write_failed_precheck(output_root: Path, lines: list[str]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "FAILED_PRECHECK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _threshold_operating_points(labels: np.ndarray, probs: np.ndarray) -> pd.DataFrame:
    from graph.review_training import compute_binary_metrics

    thresholds = np.unique(np.clip(probs, 0.0, 1.0))
    thresholds = np.concatenate(([0.0], thresholds, [1.0]))
    rows: list[dict[str, Any]] = []
    for thr in thresholds:
        metrics = compute_binary_metrics(labels, probs, threshold=float(thr))
        precision = metrics["precision"]
        recall = metrics["recall"]
        f1 = metrics["f1"]
        if precision + recall == 0:
            f2 = 0.0
        else:
            beta2 = 4.0
            f2 = (1 + beta2) * precision * recall / (beta2 * precision + recall)
        rows.append(
            {
                "threshold": float(thr),
                "AUC": metrics["auc"],
                "AP": metrics["ap"],
                "F1": f1,
                "F2": f2,
                "Recall": recall,
                "Precision": precision,
            }
        )
    frame = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
    if frame.empty:
        return pd.DataFrame(columns=["threshold_mode", "threshold", "AUC", "AP", "F1", "Recall", "Precision"])
    f1_row = frame.sort_values(["F1", "Recall", "Precision"], ascending=[False, False, False]).iloc[0]
    f2_row = frame.sort_values(["F2", "Recall", "Precision"], ascending=[False, False, False]).iloc[0]
    p70 = frame[frame["Precision"] >= 0.70]
    if len(p70):
        p70_row = p70.sort_values(["Recall", "F1"], ascending=[False, False]).iloc[0]
    else:
        p70_row = frame.sort_values(["Precision", "Recall"], ascending=[False, False]).iloc[0]
    out = []
    for mode, row in [("threshold_f1", f1_row), ("threshold_f2", f2_row), ("threshold_p70", p70_row)]:
        out.append(
            {
                "threshold_mode": mode,
                "threshold": float(row["threshold"]),
                "AUC": float(row["AUC"]),
                "AP": float(row["AP"]),
                "F1": float(row["F1"]),
                "Recall": float(row["Recall"]),
                "Precision": float(row["Precision"]),
            }
        )
    return pd.DataFrame(out)


def _phase0_phase1_precheck(project_root: Path) -> tuple[bool, list[str]]:
    lines = ["# Route L Text-Heads Precheck", ""]
    route_l_dir = project_root / "graph/routeL_llmmask_anomaly_aux"
    phase1 = project_root / "graph/outputs/routeL_phase1_dryrun_20260508_124947/dryrun_report.json"
    d1_cfg = project_root / "graph/outputs/routeD_tns_guided_logic_egat_20260504_200855/D1_EGAT_Base_LogicAE_CB/config.json"
    d1_split = project_root / "graph/outputs/yelpzip_balanced_current_graph_no_reweight_20260502_160620/prepared_data/reviews_canonical.csv"
    checks = {
        "Route L isolated code dir": route_l_dir.exists(),
        "Phase 1 dry-run exists": phase1.exists(),
        "D1 config readable": d1_cfg.exists(),
        "D1 split readable": d1_split.exists(),
    }
    ok = all(checks.values())
    for name, status in checks.items():
        lines.append(f"- {name}: {'PASS' if status else 'FAIL'}")
    if phase1.exists():
        try:
            js = json.loads(phase1.read_text(encoding="utf-8"))
            lines.append(f"- Phase 1 status value: {js.get('status')}")
            ok = ok and js.get("status") == "pass"
        except Exception:
            ok = False
            lines.append("- Phase 1 status value: FAIL_TO_PARSE")
    return ok, lines


def _experiment_name_from_cfg(path: Path) -> str:
    mapping = {
        "E0_current_main_abnormal_head": "Exp0_CurrentMainAbnormalHead",
        "E1_learned_token_evidence_head": "Exp1_LearnedTokenEvidenceHead",
        "E2_topk_token_evidence_k8": "Exp2_TopKTokenEvidenceHead_k8",
        "E2_topk_token_evidence_k16": "Exp2_TopKTokenEvidenceHead_k16",
        "E3_local_phrase_cnn_branch": "Exp3_TransformerLocalPhraseCNN",
        "E4_psycholinguistic_style_branch": "Exp4_TransformerPsycholinguisticStyle",
        "E5_textual_semantic_drift_head": "Exp5_TextualSemanticDriftHead",
    }
    return mapping.get(path.stem, path.stem)


def _prepare_extra_feature_frame(
    cfg: dict[str, Any],
    review_df: pd.DataFrame,
    dataloaders: dict[str, Any],
    bundle: Any,
    device: torch.device,
) -> pd.DataFrame | None:
    experiment_kind = str(cfg.get("EXPERIMENT_KIND", ""))
    if experiment_kind == "exp4_psycholinguistic_style":
        return build_psycholinguistic_style_frame(review_df)
    if experiment_kind == "exp5_semantic_drift":
        temp_model = build_routeL_text_model(
            primary_model_name_or_path=bundle.run_config["primary_model_name_or_path"],
            vector_dim=int(bundle.run_config.get("vector_dim", 256)),
            experiment_kind="exp1_learned_token_evidence",
            extra_feature_dim=0,
            topk_tokens=8,
        )
        temp_model.to(device)
        text_embedding_frame = compute_review_text_embeddings(temp_model, dataloaders["all"], device)
        return build_semantic_drift_frame(review_df, text_embedding_frame)
    return None


def run_single(cfg_path: Path, output_root: Path, seed: int = 42) -> dict[str, Any]:
    cfg = load_yaml_config(cfg_path)
    project_root = project_root_from_here()
    bundle = load_d1_bundle(project_root)
    exp_name = _experiment_name_from_cfg(cfg_path)
    exp_dir = ensure_dir(output_root / exp_name)
    metrics_dir = ensure_dir(exp_dir / "metrics")
    review_encoder_dir = ensure_dir(exp_dir / "review_encoder")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        review_df, llm_feature_df, _ = load_routeL_review_frames(bundle.base_dir)
        dataloaders = build_routeL_dataloaders(
            base_dir=bundle.base_dir,
            primary_model_name_or_path=bundle.run_config["primary_model_name_or_path"],
            max_seq_length=int(bundle.run_config.get("max_seq_length", 256)),
            batch_size=int(bundle.run_config.get("batch_size", 16)),
        )

        extra_feature_frame = _prepare_extra_feature_frame(cfg, review_df, dataloaders, bundle, device)
        extra_feature_dim = 0
        if extra_feature_frame is not None:
            extra_feature_dim = len([c for c in extra_feature_frame.columns if c != "review_node_id"])

        run_config = {
            "experiment_name": exp_name,
            "review_encoder": "llm_masked_logic",
            "graph_mode": bundle.run_config.get("graph_mode", "current"),
            "edge_set": "Base_LogicAE_CB",
            "model_backbone": "current_egat",
            "relation_model": "edge_aware_gat",
            "same_d1_protocol": True,
            "seed": int(seed),
            "batch_size": int(bundle.run_config.get("batch_size", 16)),
            "learning_rate": float(bundle.run_config.get("learning_rate", 2e-5)),
            "num_epochs": int(bundle.run_config.get("num_epochs", 3)),
            "patience": int(bundle.run_config.get("patience", 2)),
            "max_seq_length": int(bundle.run_config.get("max_seq_length", 256)),
            "vector_dim": int(bundle.run_config.get("vector_dim", 256)),
            "review_label_source": "review_label",
            "weak_label": False,
            "feature_source": "RouteL_end_to_end_exported_user_abnormal_vectors",
            "d1_output_dir": str(bundle.d1_output_dir),
            "d1_base_dir": str(bundle.base_dir),
            "is_strict_d1_feature": False,
            "end_to_end_recomputed_logic_graph": True,
            "user_pooling_method": "D1_default_build_review_and_user_artifacts",
            "experiment_kind": str(cfg.get("EXPERIMENT_KIND")),
            "lambda_evidence": float(cfg.get("LAMBDA_EVIDENCE", 0.2)),
            "lambda_sparse": float(cfg.get("LAMBDA_SPARSE", 0.0)),
            "topk_tokens": int(cfg.get("TOPK_TOKENS", 8)),
            "extra_feature_dim": int(extra_feature_dim),
        }
        json_dump(exp_dir / "config.json", run_config)
        json_dump(exp_dir / "review_encoder_config.json", run_config)

        model = build_routeL_text_model(
            primary_model_name_or_path=bundle.run_config["primary_model_name_or_path"],
            vector_dim=run_config["vector_dim"],
            experiment_kind=run_config["experiment_kind"],
            extra_feature_dim=extra_feature_dim,
            topk_tokens=run_config["topk_tokens"],
        )
        ckpt_path, review_metrics_csv, review_epoch_df = train_routeL_text_encoder(
            model=model,
            dataloaders=dataloaders,
            output_dir=review_encoder_dir,
            device=device,
            learning_rate=run_config["learning_rate"],
            num_epochs=run_config["num_epochs"],
            patience=run_config["patience"],
            lambda_evidence=run_config["lambda_evidence"],
            lambda_sparse=run_config["lambda_sparse"],
            extra_feature_frame=extra_feature_frame,
        )
        (exp_dir / "review_encoder_train.log").write_text("review encoder training completed\n", encoding="utf-8")

        review_output_df, review_vectors, text_vectors = encode_routeL_all_reviews(
            model=model,
            dataloader=dataloaders["all"],
            review_df=review_df,
            checkpoint_path=ckpt_path,
            device=device,
            extra_feature_frame=extra_feature_frame,
        )
        export_review_feature_frame(review_output_df, review_vectors, text_vectors, review_encoder_dir)
        review_epoch_df.to_csv(metrics_dir / "review_encoder_metrics.csv", index=False)
        review_output_df.to_csv(metrics_dir / "review_level_predictions.csv", index=False)
        np.save(review_encoder_dir / "exported_user_feature.npy", review_vectors.astype(np.float32))
        write_feature_hash(review_encoder_dir / "exported_user_feature_hash.txt", review_vectors)

        graph_review_output_df = review_output_df.rename(
            columns={
                "review_prob": "p_fake_review",
                "gate": "review_gate",
            }
        ).copy()
        review_df_for_graph = review_df.sort_values("review_node_id").reset_index(drop=True).copy()
        if "review_datetime" in review_df_for_graph.columns:
            review_df_for_graph["review_datetime"] = pd.to_datetime(review_df_for_graph["review_datetime"], errors="coerce")

        review_scores_df, user_df, _, user_abnormal_vectors, user_text_vectors = build_review_and_user_artifacts(
            review_df=review_df_for_graph,
            llm_feature_df=llm_feature_df,
            review_output_df=graph_review_output_df,
            review_vectors=review_vectors,
            text_vectors=text_vectors,
            output_dir=exp_dir,
            top_m=int(bundle.run_config.get("top_m", 3)),
            time_bucket=str(bundle.run_config.get("time_bucket", "week")),
        )

        edge_frames = build_edge_frames(
            user_df=user_df,
            user_text_vectors=user_text_vectors,
            user_abnormal_vectors=user_abnormal_vectors,
            output_dir=exp_dir,
            top_k=int(bundle.run_config.get("top_k", 20)),
            review_features=review_scores_df,
            logic_threshold_mode=str(bundle.run_config.get("logic_threshold_mode", "quantile")),
            logic_threshold_quantile=float(bundle.run_config.get("logic_threshold_quantile", 0.6)),
            logic_threshold_value=float(bundle.run_config.get("logic_threshold_value", 0.3)),
            graph_mode="current",
            senior_usu_ratio=0.10,
            use_tns_guided_logic=False,
            tns_phi_days=5,
            tns_logic_mode="boost",
            tns_logic_lambda=1.0,
            logic_tns_topk=20,
        )
        edge_stats_df = compute_edge_stats(edge_frames=edge_frames, user_df=user_df, output_dir=exp_dir)
        self_features = build_self_feature_matrix(user_df, user_abnormal_vectors)

        graph_results_df = run_relation_aggregation_experiments(
            user_df=user_df,
            self_features=self_features,
            edge_frames=edge_frames,
            output_dir=metrics_dir,
            review_encoder_name="llm_masked_logic",
            model_kind="relation_attn",
            seed=seed,
            backbone="current_egat",
            relation_model="edge_aware_gat",
            selected_edge_set="Base_LogicAE_CB",
            review_scores_df=review_scores_df,
            use_abnormal_edge_weight=False,
            use_abnormal_gate=False,
            use_abnormal_value_gate=False,
            use_abnormal_attention_bias=False,
            abnormal_score_source="auto",
            use_node_gat=False,
            return_training_details=True,
        )

        if not (metrics_dir / "epoch_metrics.csv").exists():
            graph_results_df.to_csv(metrics_dir / "epoch_metrics.csv", index=False)
        if not (metrics_dir / "test_predictions.csv").exists():
            graph_results_df.to_csv(metrics_dir / "test_predictions.csv", index=False)
        graph_results_df.to_csv(metrics_dir / "model_results.csv", index=False)
        edge_stats_df.to_csv(metrics_dir / "edge_stats.csv", index=False)
        make_llm_mask_stats_from_review_scores(review_scores_df).to_csv(metrics_dir / "llm_mask_stats.csv", index=False)

        val_pred_path = metrics_dir / "val_predictions.csv"
        if val_pred_path.exists():
            val_pred_df = pd.read_csv(val_pred_path)
            _threshold_operating_points(
                val_pred_df["label"].to_numpy(dtype=np.int64),
                val_pred_df["prob"].to_numpy(dtype=np.float32),
            ).to_csv(metrics_dir / "threshold_operating_points.csv", index=False)
        else:
            pd.DataFrame(columns=["threshold_mode", "threshold", "AUC", "AP", "F1", "Recall", "Precision"]).to_csv(
                metrics_dir / "threshold_operating_points.csv", index=False
            )

        review_metric_row = review_epoch_df.sort_values("val_auc", ascending=False).iloc[0]
        target_edge_set = graph_results_df[graph_results_df["edge_set"] == "Base_LogicAE_CB"]
        if len(target_edge_set):
            graph_best = target_edge_set.sort_values(["auc", "ap", "f1"], ascending=[False, False, False]).iloc[0]
        else:
            graph_best = graph_results_df.sort_values(["auc", "ap", "f1"], ascending=[False, False, False]).iloc[0]

        feature_hashes = {
            "exported_user_feature_path": str(review_encoder_dir / "exported_user_feature.npy"),
            "exported_user_feature_shape": list(review_vectors.shape),
            "exported_user_feature_hash": _sha256_array(review_vectors.astype(np.float32)),
            "D1_reference_feature_info_if_available": {
                "review_encoder": bundle.run_summary.get("best_graph_model", {}).get("review_encoder", "UNKNOWN_FROM_D1"),
                "d1_output_dir": str(bundle.d1_output_dir),
                "feature_dim": 288,
            },
        }
        json_dump(metrics_dir / "feature_hashes.json", feature_hashes)

        summary = {
            "experiment_name": exp_name,
            "review_encoder": "llm_masked_logic",
            "feature_dim": int(review_vectors.shape[1]),
            "feature_hash": feature_hashes["exported_user_feature_hash"],
            "review_auc": float(review_metric_row["val_auc"]),
            "review_ap": float(review_metric_row["val_ap"]),
            "review_f1": float(review_metric_row["val_f1"]),
            "review_recall": float(review_metric_row["val_recall"]),
            "review_precision": float(review_metric_row["val_precision"]),
            "graph_auc": float(graph_best["auc"]),
            "graph_ap": float(graph_best["ap"]),
            "graph_f1": float(graph_best["f1"]),
            "graph_recall": float(graph_best["recall"]),
            "graph_precision": float(graph_best["precision"]),
            "graph_best_epoch": int(graph_best.get("best_epoch", 0)) if "best_epoch" in graph_best else 0,
            "graph_threshold": float(graph_best["threshold"]),
            "notes": "",
        }
        json_dump(exp_dir / "run_summary.json", summary)
        (exp_dir / "train.log").write_text("Route L text-head experiment completed\n", encoding="utf-8")
        return summary
    except Exception:
        (exp_dir / "train_error_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config_paths", nargs="*", default=None)
    args = parser.parse_args()

    project_root = project_root_from_here()
    output_root = ensure_dir(args.output_root)
    ok, lines = _phase0_phase1_precheck(project_root)
    if not ok:
        _write_failed_precheck(output_root, lines)
        raise SystemExit(1)

    config_dir = project_root / "graph/routeL_llmmask_anomaly_aux/configs"
    default_cfg_paths = [
        config_dir / "E0_current_main_abnormal_head.yaml",
        config_dir / "E1_learned_token_evidence_head.yaml",
        config_dir / "E2_topk_token_evidence_k8.yaml",
        config_dir / "E2_topk_token_evidence_k16.yaml",
        config_dir / "E3_local_phrase_cnn_branch.yaml",
        config_dir / "E4_psycholinguistic_style_branch.yaml",
        config_dir / "E5_textual_semantic_drift_head.yaml",
    ]
    cfg_paths = [Path(p) for p in args.config_paths] if args.config_paths else default_cfg_paths
    summaries = []
    for cfg_path in cfg_paths:
        summaries.append(run_single(cfg_path, output_root, seed=args.seed))

    summaries.append(
        {
            "experiment_name": "D1_EGAT_Base_LogicAE_CB",
            "review_encoder": "llm_masked_logic",
            "feature_dim": 288,
            "feature_hash": "UNKNOWN_FROM_D1",
            "review_auc": None,
            "review_ap": None,
            "review_f1": None,
            "review_recall": None,
            "review_precision": None,
            "graph_auc": 0.8563709149922789,
            "graph_ap": 0.858368711617606,
            "graph_f1": 0.7781715095676824,
            "graph_recall": 0.823088455772114,
            "graph_precision": 0.7379032258064516,
            "graph_best_epoch": None,
            "graph_threshold": 0.42550843954086304,
            "notes": "reference only; source=graph/outputs/routeD_tns_guided_logic_egat_20260504_200855/D1_EGAT_Base_LogicAE_CB",
        }
    )
    pd.DataFrame(summaries).to_csv(output_root / "routeL_summary.csv", index=False)
    md_lines = ["# Route L Text-Head Summary", ""]
    best_graph = max([row for row in summaries if row["experiment_name"].startswith("Exp")], key=lambda row: row["graph_auc"])
    md_lines.append(f"- Best graph AUC experiment: {best_graph['experiment_name']} ({best_graph['graph_auc']:.4f})")
    md_lines.append(f"- D1 reference graph AUC: 0.8564")
    (output_root / "routeL_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
