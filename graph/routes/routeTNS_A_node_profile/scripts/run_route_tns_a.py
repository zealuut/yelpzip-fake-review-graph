from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from graph.graph_pipeline import build_self_feature_matrix, compute_edge_stats
from graph.relation_model import run_relation_aggregation_experiments
from graph.routes.routeTNS_A_node_profile.src.tns_node_profile import (
    TNSConfig,
    append_tns_profile,
    build_tns_events,
    build_tns_group_stats,
    build_tns_groups,
    build_tns_user_profile,
    save_json,
)

ASSET_PROJECT_ROOT = Path("/home/xyz/HuChao (2)/Bert-TextClassification")
BASE_PROTOCOL_DIR = ASSET_PROJECT_ROOT / "graph" / "outputs" / "yelpzip_balanced_current_graph_no_reweight_20260502_160620"
D1_REFERENCE_DIR = ASSET_PROJECT_ROOT / "graph" / "outputs" / "routeD_tns_guided_logic_egat_20260504_200855" / "D1_EGAT_Base_LogicAE_CB"
REPRO_PACK_DIR = PROJECT_ROOT / "graph" / "outputs" / "D1_REPRO_PACK"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_d1_assets() -> dict[str, Any]:
    if not REPRO_PACK_DIR.exists():
        raise FileNotFoundError(f"D1_REPRO_PACK not found: {REPRO_PACK_DIR}")
    user_df = pd.read_csv(BASE_PROTOCOL_DIR / "user_scores_enriched.csv")
    review_df = pd.read_csv(BASE_PROTOCOL_DIR / "prepared_data" / "reviews_canonical.csv")
    review_scores_df = pd.read_csv(BASE_PROTOCOL_DIR / "review_scores_enriched.csv")
    user_abnormal_vectors = np.load(BASE_PROTOCOL_DIR / "logic_vectors" / "user_abnormal_vectors.npy")
    self_features = build_self_feature_matrix(user_df.copy(), user_abnormal_vectors)

    edge_frames = {}
    for relation in ["UPU", "UTU", "USU", "LogicAE_CB"]:
        edge_frames[relation] = pd.read_csv(REPRO_PACK_DIR / "edge_pack" / f"{relation}_edges.csv")

    return {
        "user_df": user_df,
        "review_df": review_df,
        "review_scores_df": review_scores_df,
        "self_features": self_features,
        "edge_frames": edge_frames,
        "feature_path": str(REPRO_PACK_DIR / "final_self_feature_matrix.npy"),
        "feature_shape": list(self_features.shape),
        "feature_hash": _sha256_file(REPRO_PACK_DIR / "feature_hashes.json"),
        "split_hash": _sha256_file(REPRO_PACK_DIR / "split_indices.json"),
        "label_hash": _sha256_file(REPRO_PACK_DIR / "label_vector.npy"),
    }


def _write_train_log(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _find_graph_row(results_df: pd.DataFrame) -> dict[str, Any]:
    graph_rows = results_df[results_df["edge_set"] == "Base_LogicAE_CB"].copy()
    if graph_rows.empty:
        return {}
    return graph_rows.sort_values("auc", ascending=False).iloc[0].to_dict()


def run_experiment(config_path: Path, output_root: Path) -> Path:
    cfg_raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    cfg = TNSConfig(
        delta_days=int(cfg_raw.get("delta_days", 3)),
        session_threshold_days=int(cfg_raw.get("session_threshold_days", 3)),
        min_group_size=int(cfg_raw.get("min_group_size", 3)),
        max_group_duration_days=int(cfg_raw.get("max_group_duration_days", 3)),
    )
    assets = _load_d1_assets()
    exp_name = str(cfg_raw["experiment_name"])
    exp_dir = output_root / exp_name
    metrics_dir = exp_dir / "metrics"
    edges_dir = exp_dir / "edges"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    edges_dir.mkdir(parents=True, exist_ok=True)

    events_df = build_tns_events(assets["review_df"], cfg)
    groups_df, members_df = build_tns_groups(assets["review_df"], events_df, cfg)
    profile_df = build_tns_user_profile(assets["user_df"], groups_df, members_df)
    group_stats_df = build_tns_group_stats(groups_df, members_df, assets["user_df"])

    events_df.to_csv(exp_dir / "tns_events.csv", index=False)
    groups_df.to_csv(exp_dir / "tns_groups.csv", index=False)
    profile_df.to_csv(exp_dir / "tns_user_profile.csv", index=False)
    group_stats_df.to_csv(exp_dir / "tns_group_stats.csv", index=False)

    feature_hash_payload = {
        "base_feature_path": assets["feature_path"],
        "base_feature_shape": assets["feature_shape"],
        "base_feature_hash": assets["feature_hash"],
        "split_hash": assets["split_hash"],
        "label_hash": assets["label_hash"],
    }

    use_tns_node_profile = bool(cfg_raw.get("use_tns_node_profile", False))
    if use_tns_node_profile:
        exp_self_features, tns_stats = append_tns_profile(assets["self_features"], assets["user_df"], profile_df)
    else:
        exp_self_features = assets["self_features"].astype(np.float32)
        tns_stats = {
            "feature_dim_before": int(assets["self_features"].shape[1]),
            "feature_dim_after": int(assets["self_features"].shape[1]),
            "tns_feature_dim": 0,
            "tns_feature_hash": "EMPTY",
        }
    feature_hash_payload.update(tns_stats)
    save_json(metrics_dir / "feature_hashes.json", feature_hash_payload)

    for relation, frame in assets["edge_frames"].items():
        frame.to_csv(edges_dir / f"{relation}_edges.csv", index=False)

    edge_stats_df = compute_edge_stats(
        edge_frames=assets["edge_frames"],
        user_df=assets["user_df"],
        output_dir=exp_dir,
    )
    if (exp_dir / "edge_stats.csv").exists():
        (exp_dir / "edge_stats.csv").replace(metrics_dir / "edge_stats.csv")

    edge_build_config = {
        "base_source": str(D1_REFERENCE_DIR),
        "graph_protocol": "D1 fixed Base_LogicAE_CB",
        "use_tns_node_profile": use_tns_node_profile,
        "delta_days": cfg.delta_days,
        "session_threshold_days": cfg.session_threshold_days,
        "min_group_size": cfg.min_group_size,
        "max_group_duration_days": cfg.max_group_duration_days,
    }
    save_json(edges_dir / "edge_build_config.json", edge_build_config)

    results_df = run_relation_aggregation_experiments(
        user_df=assets["user_df"],
        self_features=exp_self_features,
        edge_frames=assets["edge_frames"],
        output_dir=metrics_dir,
        review_encoder_name="llm_masked_logic",
        model_kind="edge_aware_gat",
        seed=int(cfg_raw.get("seed", 42)),
        backbone="current_egat",
        relation_model="edge_aware_gat",
        review_scores_df=assets["review_scores_df"],
        selected_edge_set="Base_LogicAE_CB",
    )

    best_row = _find_graph_row(results_df)
    threshold = float(best_row.get("threshold", 0.5)) if best_row else 0.5
    summary = {
        "experiment_name": exp_name,
        "delta_days": cfg.delta_days,
        "session_threshold": cfg.session_threshold_days,
        "num_events": int(len(events_df)),
        "num_groups": int(groups_df["group_id"].nunique()) if not groups_df.empty else 0,
        "num_users_with_tns": int((profile_df["tns_group_count"] > 0).sum()) if "tns_group_count" in profile_df.columns else 0,
        "avg_groups_per_user": float(profile_df["tns_group_count"].mean()) if "tns_group_count" in profile_df.columns else 0.0,
        "feature_dim_before": int(tns_stats["feature_dim_before"]),
        "feature_dim_after": int(tns_stats["feature_dim_after"]),
        "AUC": float(best_row.get("auc", 0.0)) if best_row else 0.0,
        "AP": float(best_row.get("ap", 0.0)) if best_row else 0.0,
        "F1": float(best_row.get("f1", 0.0)) if best_row else 0.0,
        "Recall": float(best_row.get("recall", 0.0)) if best_row else 0.0,
        "Precision": float(best_row.get("precision", 0.0)) if best_row else 0.0,
        "best_epoch": "UNKNOWN_FROM_CORE",
        "threshold": threshold,
        "notes": "D1 base graph fixed; TNS profile appended to node feature only.",
    }
    save_json(exp_dir / "config.json", {**cfg_raw, **edge_build_config, **feature_hash_payload})
    save_json(exp_dir / "run_summary.json", summary)
    _write_train_log(
        exp_dir / "train.log",
        [
            f"experiment={exp_name}",
            f"use_tns_node_profile={use_tns_node_profile}",
            f"num_events={summary['num_events']}",
            f"num_groups={summary['num_groups']}",
            f"feature_dim_before={summary['feature_dim_before']}",
            f"feature_dim_after={summary['feature_dim_after']}",
        ],
    )

    # compatibility placeholders required by route contract
    if not (metrics_dir / "epoch_metrics.csv").exists():
        pd.DataFrame(
            [
                {
                    "epoch": -1,
                    "status": "UNKNOWN_FROM_CORE",
                    "note": "Legacy core runner does not emit per-epoch metrics.",
                }
            ]
        ).to_csv(metrics_dir / "epoch_metrics.csv", index=False)
    if not (metrics_dir / "test_predictions.csv").exists():
        pd.DataFrame(
            [
                {
                    "status": "UNKNOWN_FROM_CORE",
                    "note": "Legacy core runner does not emit per-user test predictions.",
                }
            ]
        ).to_csv(metrics_dir / "test_predictions.csv", index=False)

    return exp_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_paths", nargs="+", required=True)
    parser.add_argument("--output_root", required=True)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = [
        {
            "experiment_name": "D1_EGAT_Base_LogicAE_CB",
            "delta_days": 3,
            "session_threshold": 3,
            "num_events": 0,
            "num_groups": 0,
            "num_users_with_tns": 0,
            "avg_groups_per_user": 0.0,
            "feature_dim_before": 288,
            "feature_dim_after": 288,
            "AUC": 0.8563709149922789,
            "AP": 0.858368711617606,
            "F1": 0.7781715095676824,
            "Recall": 0.823088455772114,
            "Precision": 0.7379032258064516,
            "best_epoch": "REFERENCE",
            "threshold": 0.42550843954086304,
            "notes": f"source={D1_REFERENCE_DIR}",
        }
    ]
    for config_text in args.config_paths:
        exp_dir = run_experiment(Path(config_text), output_root)
        rows.append(json.loads((exp_dir / "run_summary.json").read_text(encoding="utf-8")))
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_root / "routeTNSA_summary.csv", index=False)
    (output_root / "routeTNSA_summary.md").write_text(summary_df.to_csv(index=False), encoding="utf-8")


if __name__ == "__main__":
    main()
