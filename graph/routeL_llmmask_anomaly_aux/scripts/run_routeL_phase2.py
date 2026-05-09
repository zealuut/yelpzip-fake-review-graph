from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from graph.graph_pipeline import (
    build_edge_frames,
    build_review_and_user_artifacts,
    build_self_feature_matrix,
    compute_edge_stats,
)
from graph.llm_utils import numeric_feature_columns
from graph.relation_model import run_relation_aggregation_experiments
from graph.review_training import compute_binary_metrics

from graph.routeL_llmmask_anomaly_aux.src.export_user_features_routeL import (
    assert_aux_not_exported,
    export_review_feature_frame,
)
from graph.routeL_llmmask_anomaly_aux.src.review_training_routeL import (
    build_routeL_dataloaders,
    build_routeL_model,
    encode_routeL_all_reviews,
    load_routeL_review_frames,
    train_routeL_review_encoder,
)
from graph.routeL_llmmask_anomaly_aux.src.routeL_utils import (
    ensure_dir,
    json_dump,
    load_d1_bundle,
    load_yaml_config,
    project_root_from_here,
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_array(arr: np.ndarray) -> str:
    return _sha256_bytes(np.ascontiguousarray(arr).tobytes())


def _write_failed_precheck(output_root: Path, lines: list[str]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "FAILED_PRECHECK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _review_label_stats(review_scores_df: pd.DataFrame) -> dict[str, float]:
    hit = (
        review_scores_df["num_abnormal_patterns"].gt(0)
        if "num_abnormal_patterns" in review_scores_df.columns
        else pd.Series(False, index=review_scores_df.index)
    )
    fake_mask_hit_rate = 0.0
    real_mask_hit_rate = 0.0
    if "review_label" in review_scores_df.columns:
        fake = review_scores_df["review_label"].astype(int) == 1
        real = review_scores_df["review_label"].astype(int) == 0
        if fake.any():
            fake_mask_hit_rate = float(hit[fake].mean())
        if real.any():
            real_mask_hit_rate = float(hit[real].mean())
    return {
        "mask_hit_rate": float(hit.mean()) if len(hit) else 0.0,
        "avg_pattern_count": float(
            pd.to_numeric(
                review_scores_df.get(
                    "num_abnormal_patterns",
                    pd.Series(0.0, index=review_scores_df.index),
                ),
                errors="coerce",
            )
            .fillna(0.0)
            .mean()
        )
        if len(review_scores_df)
        else 0.0,
        "llm_error_rate": float(
            review_scores_df.get("mask_source", pd.Series("", index=review_scores_df.index))
            .astype(str)
            .eq("LLM_ERROR")
            .mean()
        )
        if len(review_scores_df)
        else 0.0,
        "fake_mask_hit_rate": fake_mask_hit_rate,
        "real_mask_hit_rate": real_mask_hit_rate,
    }


def _pattern_type_distribution(review_scores_df: pd.DataFrame) -> dict[str, float]:
    dist: dict[str, float] = {}
    for col in review_scores_df.columns:
        if col.startswith("pattern_type__"):
            dist[col] = float(pd.to_numeric(review_scores_df[col], errors="coerce").fillna(0.0).mean())
    return dist


def _write_llm_mask_stats(metrics_dir: Path, review_scores_df: pd.DataFrame) -> None:
    stats = _review_label_stats(review_scores_df)
    row = {
        "num_reviews": int(len(review_scores_df)),
        **stats,
        "pattern_type_distribution": json.dumps(_pattern_type_distribution(review_scores_df), ensure_ascii=False),
    }
    pd.DataFrame([row]).to_csv(metrics_dir / "llm_mask_stats.csv", index=False)


def _threshold_operating_points(labels: np.ndarray, probs: np.ndarray) -> pd.DataFrame:
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


def _phase1_precheck(project_root: Path) -> tuple[bool, list[str]]:
    lines = ["# Route L Phase 2 Precheck", ""]
    phase1 = project_root / "graph/outputs/routeL_phase1_dryrun_20260508_124947/dryrun_report.json"
    route_l_dir = project_root / "graph/routeL_llmmask_anomaly_aux"
    d1_cfg = project_root / "graph/outputs/routeD_tns_guided_logic_egat_20260504_200855/D1_EGAT_Base_LogicAE_CB/config.json"
    d1_split = project_root / "graph/outputs/yelpzip_balanced_current_graph_no_reweight_20260502_160620/prepared_data/reviews_canonical.csv"
    d1_mask = project_root / "graph/outputs/yelpzip_balanced_current_graph_no_reweight_20260502_160620/llm_mask/llm_review_features.csv"
    checks = {
        "Phase 1 dry-run": phase1.exists(),
        "Route L isolated code dir": route_l_dir.exists(),
        "D1 config readable": d1_cfg.exists(),
        "D1 split readable": d1_split.exists(),
        "llm_masked_logic flow usable": d1_mask.exists(),
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
        "L1_early_noaux": "L1_EarlyFusion_NoAux",
        "L2_early_aux": "L2_EarlyFusion_Aux",
        "L3_late_noaux": "L3_LateFusion_NoAux",
        "L4_late_aux": "L4_LateFusion_Aux",
    }
    return mapping.get(path.stem, path.stem)


def run_single(cfg_path: Path, output_root: Path, seed: int = 42) -> dict[str, Any]:
    cfg = load_yaml_config(cfg_path)
    project_root = project_root_from_here()
    bundle = load_d1_bundle(project_root)
    exp_name = _experiment_name_from_cfg(cfg_path)
    exp_dir = ensure_dir(output_root / exp_name)
    metrics_dir = ensure_dir(exp_dir / "metrics")
    review_encoder_dir = ensure_dir(exp_dir / "review_encoder")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_config = {
        "experiment_name": exp_name,
        "fusion_mode": str(cfg.get("FUSION_MODE", "early")),
        "use_anomaly_aux_loss": bool(int(cfg.get("USE_ANOMALY_AUX_LOSS", 0))),
        "lambda_aux": float(cfg.get("lambda_aux", 0.0)),
        "anomaly_warmup_ratio": float(cfg.get("ANOMALY_WARMUP_RATIO", 0.3)),
        "review_encoder": "llm_masked_logic",
        "graph_mode": bundle.run_config.get("graph_mode", "current"),
        "edge_set": "Base_LogicAE_CB",
        "model_backbone": "current_egat",
        "relation_model": "edge_aware_gat",
        "seed": int(seed),
        "batch_size": int(bundle.run_config.get("batch_size", 16)),
        "learning_rate": float(bundle.run_config.get("learning_rate", 2e-5)),
        "num_epochs": int(bundle.run_config.get("num_epochs", 3)),
        "patience": int(bundle.run_config.get("patience", 2)),
        "max_seq_length": int(bundle.run_config.get("max_seq_length", 256)),
        "vector_dim": int(bundle.run_config.get("vector_dim", 256)),
        "review_label_source": "review_label",
        "weak_label": False,
        "feature_source": str(bundle.base_dir / "logic_vectors/user_abnormal_vectors.npy"),
        "d1_output_dir": str(bundle.d1_output_dir),
        "d1_base_dir": str(bundle.base_dir),
        "is_strict_d1_feature": True,
        "user_pooling_method": "D1_default_build_review_and_user_artifacts",
    }
    json_dump(exp_dir / "config.json", run_config)
    json_dump(exp_dir / "review_encoder_config.json", run_config)

    review_df, llm_feature_df, _ = load_routeL_review_frames(bundle.base_dir)
    dataloaders = build_routeL_dataloaders(
        base_dir=bundle.base_dir,
        primary_model_name_or_path=bundle.run_config["primary_model_name_or_path"],
        max_seq_length=run_config["max_seq_length"],
        batch_size=run_config["batch_size"],
    )
    model = build_routeL_model(
        primary_model_name_or_path=bundle.run_config["primary_model_name_or_path"],
        numeric_feature_dim=len(numeric_feature_columns()),
        vector_dim=run_config["vector_dim"],
        secondary_model_name_or_path=bundle.run_config.get("secondary_model_name_or_path"),
        freeze_primary=bool(bundle.run_config.get("freeze_primary", False)),
        freeze_secondary=bool(bundle.run_config.get("freeze_secondary", False)),
        fusion_mode=run_config["fusion_mode"],
        use_anomaly_aux_loss=run_config["use_anomaly_aux_loss"],
        anomaly_warmup_ratio=run_config["anomaly_warmup_ratio"],
        lambda_aux=run_config["lambda_aux"],
    )
    ckpt_path, review_metrics_csv, review_epoch_df = train_routeL_review_encoder(
        model=model,
        dataloaders=dataloaders,
        output_dir=review_encoder_dir,
        device=device,
        learning_rate=run_config["learning_rate"],
        num_epochs=run_config["num_epochs"],
        patience=run_config["patience"],
        lambda_aux=run_config["lambda_aux"],
        fusion_mode=run_config["fusion_mode"],
        anomaly_warmup_ratio=run_config["anomaly_warmup_ratio"],
    )
    (exp_dir / "review_encoder_train.log").write_text("review encoder training completed\n", encoding="utf-8")

    review_output_df, review_vectors, text_vectors = encode_routeL_all_reviews(
        model=model,
        dataloader=dataloaders["all"],
        review_df=review_df,
        checkpoint_path=ckpt_path,
        metrics_path=review_metrics_csv,
        device=device,
        fusion_mode=run_config["fusion_mode"],
        anomaly_warmup_ratio=run_config["anomaly_warmup_ratio"],
    )
    export_frame = export_review_feature_frame(review_output_df, review_vectors, text_vectors, review_encoder_dir)
    if not assert_aux_not_exported(export_frame):
        raise RuntimeError("aux_logit leaked into exported graph features")

    review_epoch_df.to_csv(metrics_dir / "review_encoder_metrics.csv", index=False)
    review_output_df.to_csv(metrics_dir / "review_level_predictions.csv", index=False)
    review_vectors_path = review_encoder_dir / "exported_user_feature.npy"
    np.save(review_vectors_path, review_vectors.astype(np.float32))
    (review_encoder_dir / "exported_user_feature_hash.txt").write_text(_sha256_array(review_vectors), encoding="utf-8")

    # Keep aux outputs for review-level analysis, but adapt the graph-facing review
    # output frame to the D1-compatible column contract expected by
    # build_review_and_user_artifacts().
    graph_review_output_df = review_output_df.rename(
        columns={
            "review_prob": "p_fake_review",
            "gate": "review_gate",
        }
    ).copy()
    review_df_for_graph = review_df.sort_values("review_node_id").reset_index(drop=True).copy()
    if "review_datetime" in review_df_for_graph.columns:
        review_df_for_graph["review_datetime"] = pd.to_datetime(
            review_df_for_graph["review_datetime"], errors="coerce"
        )

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
    _write_llm_mask_stats(metrics_dir, review_scores_df)

    val_pred_path = metrics_dir / "val_predictions.csv"
    if val_pred_path.exists():
        val_pred_df = pd.read_csv(val_pred_path)
        threshold_points = _threshold_operating_points(
            val_pred_df["label"].to_numpy(dtype=np.int64),
            val_pred_df["prob"].to_numpy(dtype=np.float32),
        )
        threshold_points.to_csv(metrics_dir / "threshold_operating_points.csv", index=False)
    else:
        pd.DataFrame(
            columns=["threshold_mode", "threshold", "AUC", "AP", "F1", "Recall", "Precision"]
        ).to_csv(metrics_dir / "threshold_operating_points.csv", index=False)

    review_metric_row = review_epoch_df.sort_values("val_auc", ascending=False).iloc[0]
    target_edge_set = graph_results_df[graph_results_df["edge_set"] == "Base_LogicAE_CB"]
    if len(target_edge_set):
        graph_best = target_edge_set.sort_values(["auc", "ap", "f1"], ascending=[False, False, False]).iloc[0]
    else:
        graph_best = graph_results_df.sort_values(["auc", "ap", "f1"], ascending=[False, False, False]).iloc[0]
    feature_hashes = {
        "exported_user_feature_path": str(review_vectors_path),
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
        "fusion_mode": run_config["fusion_mode"],
        "use_aux_loss": run_config["use_anomaly_aux_loss"],
        "lambda_aux": run_config["lambda_aux"],
        "warmup_ratio": run_config["anomaly_warmup_ratio"],
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
    (exp_dir / "train.log").write_text("Route L experiment completed\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config_paths", nargs="*", default=None)
    args = parser.parse_args()

    project_root = project_root_from_here()
    output_root = ensure_dir(args.output_root)
    ok, lines = _phase1_precheck(project_root)
    if not ok:
        _write_failed_precheck(output_root, lines)
        raise SystemExit(1)

    config_dir = project_root / "graph/routeL_llmmask_anomaly_aux/configs"
    default_cfg_paths = [
        config_dir / "L1_early_noaux.yaml",
        config_dir / "L2_early_aux.yaml",
        config_dir / "L3_late_noaux.yaml",
        config_dir / "L4_late_aux.yaml",
    ]
    cfg_paths = [Path(p) for p in args.config_paths] if args.config_paths else default_cfg_paths
    summaries = []
    for cfg_path in cfg_paths:
        summaries.append(run_single(cfg_path, output_root, seed=args.seed))

    summaries.append(
        {
            "experiment_name": "D1_EGAT_Base_LogicAE_CB",
            "fusion_mode": "reference_only",
            "use_aux_loss": False,
            "lambda_aux": 0.0,
            "warmup_ratio": 0.0,
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

    def best(name: str) -> dict[str, Any] | None:
        for row in summaries:
            if row["experiment_name"] == name:
                return row
        return None

    l1 = best("L1_EarlyFusion_NoAux")
    l2 = best("L2_EarlyFusion_Aux")
    l3 = best("L3_LateFusion_NoAux")
    l4 = best("L4_LateFusion_Aux")
    best_graph = max([r for r in summaries if r["experiment_name"].startswith("L")], key=lambda r: r["graph_auc"])
    md = [
        "# Route L Summary",
        "",
        f"- Early fusion > late fusion by graph AUC: {((l1 or {}).get('graph_auc', -1) > (l3 or {}).get('graph_auc', -1)) or ((l2 or {}).get('graph_auc', -1) > (l4 or {}).get('graph_auc', -1))}",
        f"- Aux helps early fusion: {((l2 or {}).get('graph_auc', -1) > (l1 or {}).get('graph_auc', -1))}",
        f"- Aux helps late fusion: {((l4 or {}).get('graph_auc', -1) > (l3 or {}).get('graph_auc', -1))}",
        f"- Best graph experiment: {best_graph['experiment_name']} (AUC={best_graph['graph_auc']:.6f}, AP={best_graph['graph_ap']:.6f}, F1={best_graph['graph_f1']:.6f})",
        f"- Exceeds D1 reference: {best_graph['graph_auc'] > 0.8563709149922789 or best_graph['graph_ap'] > 0.858368711617606 or best_graph['graph_f1'] > 0.7781715095676824}",
        "",
        "No data leakage checks found beyond D1-protocol reuse; blocked label columns are inherited from the D1 base artifacts.",
    ]
    (output_root / "routeL_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
