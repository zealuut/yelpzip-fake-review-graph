from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from graph.graph_pipeline import build_self_feature_matrix, compute_edge_stats

from .utils import standardize_like_d1


PROJECT_ROOT = Path(__file__).resolve().parents[3]
GRAPH_DIR = PROJECT_ROOT / "graph"
REFERENCE_DIR = (
    GRAPH_DIR
    / "outputs"
    / "routeD_tns_guided_logic_egat_20260504_200855"
    / "D1_EGAT_Base_LogicAE_CB"
)
BASE_PROTOCOL_DIR = GRAPH_DIR / "outputs" / "yelpzip_balanced_current_graph_no_reweight_20260502_160620"
EDGE_TYPES = ["UPU", "UTU", "USU", "LogicAE_CB"]


@dataclass
class ProtocolBundle:
    config: dict[str, Any]
    user_df: pd.DataFrame
    node_features: np.ndarray
    labels: np.ndarray
    splits: np.ndarray
    user_ids: list[str]
    user_index: dict[str, int]
    edge_frames: dict[str, pd.DataFrame]
    union_edges: pd.DataFrame
    relation_edges: pd.DataFrame
    relation_id_map: dict[str, int]
    reference_metrics: dict[str, Any]
    notes: str


def _load_reference_metrics() -> dict[str, Any]:
    payload = json.loads((REFERENCE_DIR / "run_summary.json").read_text(encoding="utf-8"))
    best = payload["best_graph_model"]
    edge_stats = pd.read_csv(REFERENCE_DIR / "metrics" / "edge_stats.csv")
    edge_types = {"UPU", "UTU", "USU", "LogicAE_CB"}
    num_edges = int(edge_stats.loc[edge_stats["edge_type"].isin(edge_types), "num_edges"].sum())
    return {
        "experiment_name": "CurrentTopK_EGAT_Base_LogicAE_CB",
        "model": "EGAT_REFERENCE",
        "graph_protocol": "current_topk",
        "edge_set": "Base_LogicAE_CB",
        "relation_handling": "typed_edge_aware_egat",
        "feature_source": "user_scores_enriched.csv + user_abnormal_vectors.npy",
        "num_users": best["num_train_users"] + best["num_val_users"] + best["num_test_users"],
        "num_edges": num_edges,
        "hidden_dim": 144,
        "num_layers": 1,
        "heads": "UNKNOWN_FROM_D1",
        "num_bases": "UNKNOWN_FROM_D1",
        "optimizer": "AdamW",
        "lr": 0.001,
        "weight_decay": 0.0005,
        "dropout": 0.2,
        "epochs": 100,
        "patience": 16,
        "early_stopping_metric": "val_auc",
        "AUC": best["auc"],
        "AP": best["ap"],
        "F1": best["f1"],
        "Recall": best["recall"],
        "Precision": best["precision"],
        "best_epoch": "UNKNOWN_FROM_D1",
        "test_threshold": best["threshold"],
        "output_dir": str(REFERENCE_DIR),
        "notes": "Reference row only, not trained in baseline_comparison.",
        "source": str(REFERENCE_DIR),
    }


def _load_edge_frames() -> dict[str, pd.DataFrame]:
    edge_frames: dict[str, pd.DataFrame] = {}
    for edge_type in EDGE_TYPES:
        frame = pd.read_csv(REFERENCE_DIR / "edges" / f"{edge_type}_edges.csv")
        frame["src_user_id"] = frame["src_user_id"].astype(str)
        frame["dst_user_id"] = frame["dst_user_id"].astype(str)
        edge_frames[edge_type] = frame
    return edge_frames


def _build_union_edges(edge_frames: dict[str, pd.DataFrame], user_index: dict[str, int]) -> pd.DataFrame:
    union = pd.concat([edge_frames[name][["src_user_id", "dst_user_id", "edge_weight", "edge_type"]] for name in EDGE_TYPES], ignore_index=True)
    union["edge_weight"] = pd.to_numeric(union["edge_weight"], errors="coerce").fillna(0.0).astype(np.float32)
    union = (
        union.groupby(["src_user_id", "dst_user_id"], as_index=False)
        .agg(edge_weight=("edge_weight", "sum"))
        .sort_values(["src_user_id", "dst_user_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    union["src_index"] = union["src_user_id"].map(user_index).astype(np.int64)
    union["dst_index"] = union["dst_user_id"].map(user_index).astype(np.int64)
    return union


def _build_relation_edges(edge_frames: dict[str, pd.DataFrame], user_index: dict[str, int]) -> tuple[pd.DataFrame, dict[str, int]]:
    relation_id_map = {name: idx for idx, name in enumerate(EDGE_TYPES)}
    rows = []
    for relation_name in EDGE_TYPES:
        frame = edge_frames[relation_name][["src_user_id", "dst_user_id", "edge_weight"]].copy()
        frame["relation_name"] = relation_name
        frame["relation_id"] = relation_id_map[relation_name]
        rows.append(frame)
    relation_df = pd.concat(rows, ignore_index=True)
    relation_df["edge_weight"] = pd.to_numeric(relation_df["edge_weight"], errors="coerce").fillna(0.0).astype(np.float32)
    relation_df["src_index"] = relation_df["src_user_id"].map(user_index).astype(np.int64)
    relation_df["dst_index"] = relation_df["dst_user_id"].map(user_index).astype(np.int64)
    relation_df = relation_df.sort_values(["relation_id", "src_user_id", "dst_user_id"], kind="mergesort").reset_index(drop=True)
    return relation_df, relation_id_map


def load_protocol_bundle() -> ProtocolBundle:
    d1_config = json.loads((REFERENCE_DIR / "config.json").read_text(encoding="utf-8"))
    user_df = pd.read_csv(BASE_PROTOCOL_DIR / "user_scores_enriched.csv")
    user_df["user_id"] = user_df["user_id"].astype(str)
    user_abnormal_vectors = np.load(BASE_PROTOCOL_DIR / "logic_vectors" / "user_abnormal_vectors.npy")
    node_features = build_self_feature_matrix(user_df, user_abnormal_vectors)
    node_features = standardize_like_d1(node_features)
    labels = user_df["user_label"].to_numpy(dtype=np.int64)
    splits = user_df["split"].astype(str).to_numpy()
    user_ids = user_df["user_id"].astype(str).tolist()
    user_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
    edge_frames = _load_edge_frames()
    union_edges = _build_union_edges(edge_frames, user_index)
    relation_edges, relation_id_map = _build_relation_edges(edge_frames, user_index)
    reference_metrics = _load_reference_metrics()
    notes = (
        "Aligned to D1 current-topk protocol where confirmed from config/code. "
        "blocked_label_columns=UNKNOWN_FROM_D1; heads/num_bases are model-specific."
    )
    return ProtocolBundle(
        config=d1_config,
        user_df=user_df,
        node_features=node_features,
        labels=labels,
        splits=splits,
        user_ids=user_ids,
        user_index=user_index,
        edge_frames=edge_frames,
        union_edges=union_edges,
        relation_edges=relation_edges,
        relation_id_map=relation_id_map,
        reference_metrics=reference_metrics,
        notes=notes,
    )


def write_edge_stats(bundle: ProtocolBundle, experiment_dir: str | Path) -> pd.DataFrame:
    return compute_edge_stats(bundle.edge_frames, bundle.user_df, experiment_dir)
