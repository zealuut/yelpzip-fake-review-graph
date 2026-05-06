from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def safe_binary_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, float]:
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs).astype(np.float32)
    preds = (probs >= threshold).astype(int)
    unique_labels = np.unique(labels)
    return {
        "auc": float(roc_auc_score(labels, probs)) if len(unique_labels) > 1 else 0.0,
        "ap": float(average_precision_score(labels, probs)) if len(unique_labels) > 1 else 0.0,
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "accuracy": float(accuracy_score(labels, preds)),
    }


def select_threshold_from_validation(labels: np.ndarray, probs: np.ndarray) -> float:
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs).astype(np.float32)
    if len(labels) == 0:
        return 0.5
    candidate_thresholds = np.unique(np.clip(probs, 0.01, 0.99))
    candidate_thresholds = np.concatenate([[0.5], candidate_thresholds])
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in candidate_thresholds:
        f1 = f1_score(labels, (probs >= threshold).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_threshold = float(threshold)
    return best_threshold


def epoch_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if "epoch" in frame.columns:
        frame = frame.sort_values("epoch").reset_index(drop=True)
    return frame
