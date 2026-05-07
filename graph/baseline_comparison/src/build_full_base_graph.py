from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from graph.data_utils import LABEL_LEAKAGE_COLUMNS
from graph.graph_pipeline import compute_edge_stats

from .utils import standardize_like_d1


PROJECT_ROOT = Path(__file__).resolve().parents[3]
GRAPH_DIR = PROJECT_ROOT / "graph"
FULL_BASE_DIR = GRAPH_DIR / "outputs" / "yelpzip_senior_backbone_clean_formal_20260502_101931"
REFERENCE_DIR = (
    GRAPH_DIR
    / "outputs"
    / "routeD_tns_guided_logic_egat_20260504_200855"
    / "D1_EGAT_Base_LogicAE_CB"
)
FULL_BASE_RELATIONS = ["UPU", "UTU", "USU"]
NON_FEATURE_COLUMNS = {
    "user_id",
    "split",
    "product_set",
    "time_bucket_set",
    *LABEL_LEAKAGE_COLUMNS,
}


@dataclass
class FullBaseProtocolBundle:
    config: dict[str, Any]
    user_df: pd.DataFrame
    feature_frame: pd.DataFrame
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
    feature_source: str
    feature_dim: int
    blocked_label_columns: list[str]
    split_stats: dict[str, int]


def _load_run_config() -> dict[str, Any]:
    return json.loads((FULL_BASE_DIR / "run_config.json").read_text(encoding="utf-8"))


def _load_user_feature_frame() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    user_df = pd.read_csv(FULL_BASE_DIR / "logic_vectors" / "user_summary.csv")
    user_df["user_id"] = user_df["user_id"].astype(str)
    user_df["split"] = user_df["split"].astype(str)

    numeric_feature_cols: list[str] = []
    for column in user_df.columns:
        if column in NON_FEATURE_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(user_df[column]):
            numeric_feature_cols.append(column)

    feature_frame = user_df[["user_id", "split", "user_label", *numeric_feature_cols]].copy()
    return user_df, feature_frame, numeric_feature_cols


def _load_edge_frames() -> dict[str, pd.DataFrame]:
    edge_frames: dict[str, pd.DataFrame] = {}
    for relation_name in FULL_BASE_RELATIONS:
        frame = pd.read_csv(FULL_BASE_DIR / "edges" / f"{relation_name}_edges.csv")
        frame["src_user_id"] = frame["src_user_id"].astype(str)
        frame["dst_user_id"] = frame["dst_user_id"].astype(str)
        frame["edge_type"] = relation_name
        frame["edge_weight"] = pd.to_numeric(frame["edge_weight"], errors="coerce").fillna(0.0).astype(np.float32)
        edge_frames[relation_name] = frame
    return edge_frames


def _build_union_edges(edge_frames: dict[str, pd.DataFrame], user_index: dict[str, int]) -> pd.DataFrame:
    union = pd.concat(
        [edge_frames[name][["src_user_id", "dst_user_id", "edge_weight", "edge_type"]] for name in FULL_BASE_RELATIONS],
        ignore_index=True,
    )
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


def _build_relation_edges(
    edge_frames: dict[str, pd.DataFrame],
    user_index: dict[str, int],
) -> tuple[pd.DataFrame, dict[str, int]]:
    relation_id_map = {name: idx for idx, name in enumerate(FULL_BASE_RELATIONS)}
    rows: list[pd.DataFrame] = []
    for relation_name in FULL_BASE_RELATIONS:
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


def _build_split_stats(user_df: pd.DataFrame) -> dict[str, int]:
    split_series = user_df["split"].astype(str)
    label_series = pd.to_numeric(user_df["user_label"], errors="coerce").fillna(0).astype(int)

    def _count(split_name: str, label_value: int | None = None) -> int:
        mask = split_series.eq(split_name)
        if label_value is not None:
            mask = mask & label_series.eq(label_value)
        return int(mask.sum())

    return {
        "num_users": int(len(user_df)),
        "num_train": _count("train"),
        "num_val": _count("val"),
        "num_test": _count("test"),
        "num_fake_train": _count("train", 1),
        "num_real_train": _count("train", 0),
        "num_fake_val": _count("val", 1),
        "num_real_val": _count("val", 0),
        "num_fake_test": _count("test", 1),
        "num_real_test": _count("test", 0),
    }


def _load_reference_metrics() -> dict[str, Any]:
    payload = json.loads((REFERENCE_DIR / "run_summary.json").read_text(encoding="utf-8"))
    best = payload["best_graph_model"]
    edge_stats = pd.read_csv(REFERENCE_DIR / "metrics" / "edge_stats.csv")
    target_relations = {"UPU", "UTU", "USU", "LogicAE_CB"}
    num_edges = int(edge_stats.loc[edge_stats["edge_type"].isin(target_relations), "num_edges"].sum())
    return {
        "experiment_name": "CurrentTopK_EGAT_Base_LogicAE_CB",
        "model": "EGAT_REFERENCE",
        "graph_protocol": "current_topk",
        "relations": "UPU,UTU,USU,LogicAE_CB",
        "relation_handling": "typed_edge_aware_egat",
        "feature_source": "current mainline D1 cached user features",
        "feature_dim": "UNKNOWN_FROM_D1",
        "blocked_label_columns": ",".join(sorted(LABEL_LEAKAGE_COLUMNS)),
        "num_users": 6664,
        "num_train": int(best["num_train_users"]),
        "num_val": int(best["num_val_users"]),
        "num_test": int(best["num_test_users"]),
        "num_fake_train": int(best["num_fake_train"]),
        "num_real_train": int(best["num_train_users"] - best["num_fake_train"]),
        "num_fake_val": int(best["num_fake_val"]),
        "num_real_val": int(best["num_val_users"] - best["num_fake_val"]),
        "num_fake_test": int(best["num_fake_test"]),
        "num_real_test": int(best["num_test_users"] - best["num_fake_test"]),
        "num_edges": num_edges,
        "hidden_dim": "REFERENCE_ONLY",
        "num_layers": "REFERENCE_ONLY",
        "heads": "REFERENCE_ONLY",
        "num_bases": "REFERENCE_ONLY",
        "use_neighbor_sampling": False,
        "optimizer": "REFERENCE_ONLY",
        "lr": "REFERENCE_ONLY",
        "weight_decay": "REFERENCE_ONLY",
        "dropout": "REFERENCE_ONLY",
        "epochs": "REFERENCE_ONLY",
        "patience": "REFERENCE_ONLY",
        "AUC": 0.85637,
        "AP": 0.85837,
        "F1": 0.77817,
        "Recall": 0.82309,
        "Precision": 0.73790,
        "best_epoch": "REFERENCE_ONLY",
        "test_threshold": float(best["threshold"]),
        "output_dir": str(REFERENCE_DIR),
        "notes": "reference only; not trained in this baseline run",
        "source": "graph/outputs/routeD_tns_guided_logic_egat_20260504_200855/D1_EGAT_Base_LogicAE_CB",
    }


def load_full_base_bundle() -> FullBaseProtocolBundle:
    config = _load_run_config()
    user_df, feature_frame, feature_columns = _load_user_feature_frame()
    node_features = standardize_like_d1(feature_frame[feature_columns].to_numpy(dtype=np.float32))
    labels = pd.to_numeric(user_df["user_label"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    splits = user_df["split"].astype(str).to_numpy()
    user_ids = user_df["user_id"].astype(str).tolist()
    user_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
    edge_frames = _load_edge_frames()
    union_edges = _build_union_edges(edge_frames, user_index)
    relation_edges, relation_id_map = _build_relation_edges(edge_frames, user_index)
    split_stats = _build_split_stats(user_df)
    reference_metrics = _load_reference_metrics()
    blocked = sorted(LABEL_LEAKAGE_COLUMNS)
    notes = (
        "FullBase_UPU_UTU_USU graph baseline. Reuses the same balanced user set, labels, and split as the current D1 protocol. "
        "Node features are clean behavior/text-profile numeric features from logic_vectors/user_summary.csv only; "
        "no LogicAE embedding, no LLM mask embedding, no TNS feature, no abnormal compression feature."
    )
    return FullBaseProtocolBundle(
        config=config,
        user_df=user_df,
        feature_frame=feature_frame,
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
        feature_source="logic_vectors/user_summary.csv numeric behavior features only",
        feature_dim=len(feature_columns),
        blocked_label_columns=blocked,
        split_stats=split_stats,
    )


def write_full_base_edge_stats(bundle: FullBaseProtocolBundle, experiment_dir: str | Path) -> pd.DataFrame:
    renamed_frames: dict[str, pd.DataFrame] = {}
    for relation_name, frame in bundle.edge_frames.items():
        relation_frame = frame.copy()
        relation_frame["edge_type"] = relation_name
        renamed_frames[relation_name] = relation_frame
    return compute_edge_stats(renamed_frames, bundle.user_df, experiment_dir)
