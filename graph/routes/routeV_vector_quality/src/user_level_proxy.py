"""V0: user-level proxy metrics for checkpoint selection.

The checkpoint selector must not fit and evaluate on the same val users. The
main route helper below aggregates review vectors to user vectors, fits a small
linear probe on train users, and evaluates it on val users. This keeps V0 a
selection proxy rather than a leaked validation classifier.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return 0.0
    return float(roc_auc_score(labels, scores))


def _safe_ap(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return 0.0
    return float(average_precision_score(labels, scores))


def aggregate_user_vectors(
    review_vectors: np.ndarray,
    review_df: pd.DataFrame,
    top_m: int = 3,
    score_column: str = "p_fake_review",
    label_column: str = "user_label",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate review vectors to user vectors using D1-style top-m reviews.

    RouteV strict mode requires explicit user-level labels. The label column is
    expected to come from prepared.user_df.user_label and be attached to each
    review row before this function is called.
    """
    if len(review_vectors) != len(review_df):
        raise ValueError("review_vectors and review_df must have the same row count")
    df = review_df.reset_index(drop=True).copy()
    required_columns = {"user_id", "split", label_column}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(
            "RouteV user-vector proxy requires explicit user-level labels; "
            f"missing columns: {missing_columns}"
        )
    df["_vec_idx"] = np.arange(len(df))
    if score_column not in df.columns:
        df[score_column] = 0.0

    user_vectors: list[np.ndarray] = []
    user_labels: list[int] = []
    user_ids: list[str] = []
    user_splits: list[str] = []
    top_m = max(int(top_m), 1)
    for user_id, group in df.groupby("user_id", sort=True):
        ranked = group.sort_values(score_column, ascending=False)
        selected_indices = ranked.head(top_m)["_vec_idx"].to_numpy(dtype=np.int64)
        user_vectors.append(review_vectors[selected_indices].mean(axis=0))
        labels = group[label_column]
        if labels.isna().any() or labels.nunique(dropna=False) != 1:
            raise ValueError(
                f"RouteV proxy found inconsistent {label_column} values for user_id={user_id!r}"
            )
        user_labels.append(int(labels.iloc[0]))
        user_ids.append(str(user_id))
        user_splits.append(str(group["split"].iloc[0]))

    return (
        np.asarray(user_vectors, dtype=np.float32),
        np.asarray(user_labels, dtype=np.int32),
        np.asarray(user_ids, dtype=object),
        np.asarray(user_splits, dtype=object),
    )


def compute_user_vector_proxy_train_eval(
    review_vectors: np.ndarray,
    review_df: pd.DataFrame,
    train_split: str = "train",
    eval_split: str = "val",
    top_m: int = 3,
    score_column: str = "p_fake_review",
    label_column: str = "user_label",
) -> dict[str, Any]:
    """Fit a small user-vector probe on train users and evaluate on val users."""
    user_vectors, user_labels, _user_ids, user_splits = aggregate_user_vectors(
        review_vectors=review_vectors,
        review_df=review_df,
        top_m=top_m,
        score_column=score_column,
        label_column=label_column,
    )
    train_mask = user_splits == train_split
    eval_mask = user_splits == eval_split
    x_train = user_vectors[train_mask]
    y_train = user_labels[train_mask]
    x_eval = user_vectors[eval_mask]
    y_eval = user_labels[eval_mask]

    if len(x_eval) == 0 or len(np.unique(y_eval)) < 2:
        return {
            "user_auc": 0.0,
            "user_ap": 0.0,
            "label_source": label_column,
            "proxy_metric_method": "train_user_linear_probe_eval_val_auc",
            "linear_probe_val_auc": 0.0,
            "linear_probe_val_ap": 0.0,
            "centroid_val_auc": 0.0,
            "norm_val_auc": 0.0,
            "train_split": train_split,
            "eval_split": eval_split,
            "top_m": int(top_m),
            "num_train_users": int(len(x_train)),
            "num_eval_users": int(len(x_eval)),
        }

    norm_scores = np.linalg.norm(x_eval, axis=1)
    norm_auc = _safe_auc(y_eval, norm_scores)
    if norm_auc < 0.5:
        norm_auc = 1.0 - norm_auc
        norm_scores = -norm_scores

    centroid_auc = 0.0
    centroid_ap = 0.0
    if len(x_train) > 0 and len(np.unique(y_train)) == 2:
        fake_centroid = x_train[y_train == 1].mean(axis=0)
        real_centroid = x_train[y_train == 0].mean(axis=0)
        fake_centroid = fake_centroid / max(np.linalg.norm(fake_centroid), 1e-8)
        real_centroid = real_centroid / max(np.linalg.norm(real_centroid), 1e-8)
        x_eval_norm = x_eval / np.clip(np.linalg.norm(x_eval, axis=1, keepdims=True), 1e-8, None)
        centroid_scores = x_eval_norm @ fake_centroid - x_eval_norm @ real_centroid
        centroid_auc = _safe_auc(y_eval, centroid_scores)
        centroid_ap = _safe_ap(y_eval, centroid_scores)

    linear_auc = 0.0
    linear_ap = 0.0
    if len(x_train) >= 4 and len(np.unique(y_train)) == 2:
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_eval_scaled = scaler.transform(x_eval)
        clf = LogisticRegression(max_iter=500, C=0.5, solver="lbfgs", class_weight="balanced")
        clf.fit(x_train_scaled, y_train)
        eval_probs = clf.predict_proba(x_eval_scaled)[:, 1]
        linear_auc = _safe_auc(y_eval, eval_probs)
        linear_ap = _safe_ap(y_eval, eval_probs)

    user_auc = linear_auc or centroid_auc or norm_auc
    user_ap = linear_ap or centroid_ap or _safe_ap(y_eval, norm_scores)
    return {
        "user_auc": float(user_auc),
        "user_ap": float(user_ap),
        "label_source": label_column,
        "proxy_metric_method": "train_user_linear_probe_eval_val_auc",
        "linear_probe_val_auc": float(linear_auc),
        "linear_probe_val_ap": float(linear_ap),
        "centroid_val_auc": float(centroid_auc),
        "centroid_val_ap": float(centroid_ap),
        "norm_val_auc": float(norm_auc),
        "train_split": train_split,
        "eval_split": eval_split,
        "top_m": int(top_m),
        "num_train_users": int(len(x_train)),
        "num_eval_users": int(len(x_eval)),
    }


def compute_user_vector_proxy(
    review_vectors: np.ndarray,
    review_df: pd.DataFrame,
    top_m: int = 3,
    score_column: str = "p_fake_review",
    label_column: str = "user_label",
) -> dict[str, Any]:
    """Backward-compatible self-contained proxy for diagnostics only."""
    user_vectors, user_labels, _user_ids, _splits = aggregate_user_vectors(
        review_vectors=review_vectors,
        review_df=review_df,
        top_m=top_m,
        score_column=score_column,
        label_column=label_column,
    )
    if len(np.unique(user_labels)) < 2:
        return {
            "user_auc": 0.0,
            "user_ap": 0.0,
            "user_norm_score": 0.0,
            "label_source": label_column,
            "proxy_metric_method": "all_user_norm_auc",
            "top_m": int(top_m),
        }
    user_norms = np.linalg.norm(user_vectors, axis=1)
    norm_auc = _safe_auc(user_labels, user_norms)
    if norm_auc < 0.5:
        norm_auc = 1.0 - norm_auc
        user_norms = -user_norms
    return {
        "user_auc": float(norm_auc),
        "user_ap": _safe_ap(user_labels, user_norms),
        "user_norm_score": float(norm_auc),
        "label_source": label_column,
        "proxy_metric_method": "all_user_norm_auc",
        "top_m": int(top_m),
    }


def compute_user_vector_proxy_from_split(
    review_vectors: np.ndarray,
    review_df: pd.DataFrame,
    split: str = "val",
    top_m: int = 3,
    score_column: str = "p_fake_review",
    label_column: str = "user_label",
) -> dict[str, Any]:
    """Backward-compatible split diagnostic using the non-parametric norm score."""
    split_mask = review_df["split"] == split
    split_df = review_df[split_mask].reset_index(drop=True)
    split_vectors = review_vectors[split_mask.to_numpy()]
    return compute_user_vector_proxy(
        split_vectors,
        split_df,
        top_m=top_m,
        score_column=score_column,
        label_column=label_column,
    )

