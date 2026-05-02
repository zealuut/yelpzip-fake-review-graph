from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


EDGE_SET_DEFINITIONS = OrderedDict(
    [
        ("MLP_no_graph", []),
        ("Base", ["UPU", "UTU", "USU"]),
        ("Base_TextSim", ["UPU", "UTU", "USU", "TextSim"]),
        ("Base_CB", ["UPU", "UTU", "USU", "CB"]),
        ("Base_LogicAE_CB", ["UPU", "UTU", "USU", "LogicAE_CB"]),
        ("Full", ["UPU", "UTU", "USU", "TextSim", "CB", "LogicAE_CB"]),
    ]
)

BEHAVIOR_FEATURE_COLUMNS = [
    "total_reviews",
    "avg_rating",
    "rating_std",
    "rating_entropy",
    "rating_deviation_avg",
    "rating_deviation_std",
    "positive_ratio",
    "negative_ratio",
    "extreme_rating_ratio",
    "max_daily_reviews",
    "burst_ratio",
    "active_days",
    "user_tenure_days",
    "avg_review_gap_days",
    "std_review_gap_days",
    "avg_review_time_lag_days",
    "std_review_time_lag_days",
    "avg_review_length",
    "RD",
    "EXR",
    "MRO",
    "AD",
    "ATR",
    "behavior_anomaly_score",
]


def _safe_binary_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, float]:
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


def _select_threshold_from_validation(labels: np.ndarray, probs: np.ndarray) -> float:
    if len(labels) == 0:
        return 0.5
    candidate_thresholds = np.unique(np.clip(probs, 0.01, 0.99))
    candidate_thresholds = np.concatenate([[0.5], candidate_thresholds])
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in candidate_thresholds:
        f1 = f1_score(labels, (probs >= threshold).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold


def aggregate_relation_features(
    self_features: np.ndarray,
    edges_df: pd.DataFrame,
    user_index: dict[str, int],
) -> np.ndarray:
    aggregated = np.zeros_like(self_features, dtype=np.float32)
    weight_sums = np.zeros((self_features.shape[0], 1), dtype=np.float32)

    if edges_df.empty:
        return aggregated

    for row in edges_df.itertuples(index=False):
        src_idx = user_index[str(row.src_user_id)]
        dst_idx = user_index[str(row.dst_user_id)]
        weight = float(max(row.edge_weight, 1e-6))
        aggregated[src_idx] += weight * self_features[dst_idx]
        weight_sums[src_idx, 0] += weight

    non_zero = weight_sums[:, 0] > 0
    aggregated[non_zero] = aggregated[non_zero] / weight_sums[non_zero]
    return aggregated


def _build_model(model_kind: str, seed: int) -> Pipeline:
    if model_kind == "mlp":
        classifier = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            learning_rate_init=1e-3,
            max_iter=300,
            random_state=seed,
        )
    else:
        classifier = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=seed,
        )

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", classifier),
        ]
    )


def _fit_feature_only_baseline(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    model_kind: str,
    seed: int,
) -> dict[str, Any]:
    model = _build_model(model_kind=model_kind, seed=seed)
    model.fit(feature_matrix[train_mask], labels[train_mask])
    val_probs = model.predict_proba(feature_matrix[val_mask])[:, 1]
    test_probs = model.predict_proba(feature_matrix[test_mask])[:, 1]
    threshold = _select_threshold_from_validation(labels[val_mask], val_probs)
    val_metrics = _safe_binary_metrics(labels[val_mask], val_probs, threshold=threshold)
    test_metrics = _safe_binary_metrics(labels[test_mask], test_probs, threshold=threshold)
    return {
        "model_name": model_kind,
        "threshold": threshold,
        "val_auc": val_metrics["auc"],
        "val_ap": val_metrics["ap"],
        "auc": test_metrics["auc"],
        "ap": test_metrics["ap"],
        "recall": test_metrics["recall"],
        "precision": test_metrics["precision"],
        "f1": test_metrics["f1"],
        "accuracy": test_metrics["accuracy"],
    }


def _import_torch_modules():
    try:
        import torch
        from torch import nn
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover - depends on server runtime
        raise ImportError(
            "relation_attn requires torch. Use --relation_model logreg/mlp for a sklearn-only fallback."
        ) from exc
    return torch, nn, F


def _predict_relation_attention_probs(
    model: Any,
    blocks: np.ndarray,
    block_mask: np.ndarray,
    torch: Any,
    device: Any,
    batch_size: int = 4096,
) -> np.ndarray:
    model.eval()
    probs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(blocks), batch_size):
            end = min(start + batch_size, len(blocks))
            batch_blocks = torch.as_tensor(blocks[start:end], dtype=torch.float32, device=device)
            batch_mask = torch.as_tensor(block_mask[start:end], dtype=torch.bool, device=device)
            logits = model(batch_blocks, batch_mask)
            probs.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(probs, axis=0)


def _fit_relation_attention_model(
    feature_blocks: list[np.ndarray],
    active_masks: list[np.ndarray],
    labels: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    seed: int,
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
    torch, nn, F = _import_torch_modules()

    class RelationAttentionClassifier(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.25) -> None:
            super().__init__()
            self.block_encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.relation_scorer = nn.Linear(hidden_dim, 1)
            self.classifier = nn.Sequential(
                nn.LayerNorm(hidden_dim * 2),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, blocks: Any, block_mask: Any) -> Any:
            encoded = self.block_encoder(blocks)
            scores = self.relation_scorer(encoded).squeeze(-1)
            scores = scores.masked_fill(~block_mask.bool(), -1e9)
            weights = F.softmax(scores, dim=1)
            weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
            pooled = torch.sum(encoded * weights.unsqueeze(-1), dim=1)
            fused = torch.cat([encoded[:, 0, :], pooled], dim=-1)
            return self.classifier(fused).squeeze(-1)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    blocks = np.stack(feature_blocks, axis=1).astype(np.float32)
    block_mask = np.stack(active_masks, axis=1).astype(bool)
    block_mask[:, 0] = True

    scaler = StandardScaler()
    train_blocks = blocks[train_mask].reshape(-1, blocks.shape[-1])
    scaler.fit(train_blocks)
    scaled_blocks = scaler.transform(blocks.reshape(-1, blocks.shape[-1])).reshape(blocks.shape).astype(np.float32)

    input_dim = scaled_blocks.shape[-1]
    hidden_dim = min(256, max(64, input_dim // 2))
    model = RelationAttentionClassifier(input_dim=input_dim, hidden_dim=hidden_dim).to(device)

    train_indices = np.where(train_mask)[0]
    y_train = labels[train_mask].astype(np.float32)
    pos_count = float(y_train.sum())
    neg_count = float(len(y_train) - pos_count)
    pos_weight = torch.tensor([neg_count / max(pos_count, 1.0)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)

    best_state = None
    best_val_auc = -1.0
    bad_epochs = 0
    batch_size = 2048
    max_epochs = 260
    patience = 35
    rng = np.random.default_rng(seed)

    for _epoch in range(max_epochs):
        model.train()
        rng.shuffle(train_indices)
        for start in range(0, len(train_indices), batch_size):
            batch_idx = train_indices[start:start + batch_size]
            batch_blocks = torch.as_tensor(scaled_blocks[batch_idx], dtype=torch.float32, device=device)
            batch_mask = torch.as_tensor(block_mask[batch_idx], dtype=torch.bool, device=device)
            batch_labels = torch.as_tensor(labels[batch_idx], dtype=torch.float32, device=device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_blocks, batch_mask)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

        val_probs = _predict_relation_attention_probs(
            model=model,
            blocks=scaled_blocks[val_mask],
            block_mask=block_mask[val_mask],
            torch=torch,
            device=device,
        )
        val_auc = _safe_binary_metrics(labels[val_mask], val_probs, threshold=0.5)["auc"]
        if val_auc > best_val_auc + 1e-5:
            best_val_auc = val_auc
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, scaled_blocks, block_mask, np.asarray([best_val_auc], dtype=np.float32)


def run_relation_aggregation_experiments(
    user_df: pd.DataFrame,
    self_features: np.ndarray,
    edge_frames: dict[str, pd.DataFrame],
    output_dir: str | Path,
    review_encoder_name: str,
    model_kind: str,
    seed: int,
    results_filename: str = "model_results.csv",
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ordered_users = user_df.drop_duplicates(subset=["user_id"]).reset_index(drop=True)
    user_ids = ordered_users["user_id"].astype(str).tolist()
    user_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
    labels = ordered_users["user_label"].to_numpy(dtype=np.int64)
    splits = ordered_users["split"].astype(str).to_numpy()

    if self_features.shape[0] != len(ordered_users):
        raise ValueError("Self feature matrix row count must match ordered user count.")

    results: list[dict[str, Any]] = []
    train_mask = splits == "train"
    val_mask = splits == "val"
    test_mask = splits == "test"
    required_relations = sorted({relation for relations in EDGE_SET_DEFINITIONS.values() for relation in relations})
    relation_aggregates: dict[str, np.ndarray] = {}
    relation_active_masks: dict[str, np.ndarray] = {}
    for relation_name in required_relations:
        relation_edges = edge_frames.get(relation_name, pd.DataFrame())
        aggregated = aggregate_relation_features(self_features, relation_edges, user_index)
        relation_aggregates[relation_name] = aggregated
        relation_active_masks[relation_name] = np.linalg.norm(aggregated, axis=1) > 0

    behavior_feature_matrix = (
        ordered_users.reindex(columns=BEHAVIOR_FEATURE_COLUMNS, fill_value=0.0)
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    for baseline_name, baseline_kind in [("Behavior_LR", "logreg"), ("Behavior_MLP", "mlp")]:
        baseline_metrics = _fit_feature_only_baseline(
            feature_matrix=behavior_feature_matrix,
            labels=labels,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            model_kind=baseline_kind,
            seed=seed,
        )
        results.append(
            {
                "review_encoder": review_encoder_name,
                "model_name": f"feature_only_{baseline_kind}",
                "edge_set": baseline_name,
                "threshold": baseline_metrics["threshold"],
                "val_auc": baseline_metrics["val_auc"],
                "val_ap": baseline_metrics["val_ap"],
                "auc": baseline_metrics["auc"],
                "ap": baseline_metrics["ap"],
                "recall": baseline_metrics["recall"],
                "precision": baseline_metrics["precision"],
                "f1": baseline_metrics["f1"],
                "accuracy": baseline_metrics["accuracy"],
                "num_train_users": int(train_mask.sum()),
                "num_val_users": int(val_mask.sum()),
                "num_test_users": int(test_mask.sum()),
                "num_fake_train": int(labels[train_mask].sum()),
                "num_fake_val": int(labels[val_mask].sum()),
                "num_fake_test": int(labels[test_mask].sum()),
            }
        )

    for edge_set_name, relations in EDGE_SET_DEFINITIONS.items():
        feature_blocks = [self_features]
        active_masks = [np.ones(self_features.shape[0], dtype=bool)]
        for relation_name in relations:
            feature_blocks.append(relation_aggregates[relation_name])
            active_masks.append(relation_active_masks[relation_name])

        if model_kind == "relation_attn":
            model, scaled_blocks, block_mask, _ = _fit_relation_attention_model(
                feature_blocks=feature_blocks,
                active_masks=active_masks,
                labels=labels,
                train_mask=train_mask,
                val_mask=val_mask,
                seed=seed,
            )
            torch, _, _ = _import_torch_modules()
            device = next(model.parameters()).device
            val_probs = _predict_relation_attention_probs(
                model=model,
                blocks=scaled_blocks[val_mask],
                block_mask=block_mask[val_mask],
                torch=torch,
                device=device,
            )
            test_probs = _predict_relation_attention_probs(
                model=model,
                blocks=scaled_blocks[test_mask],
                block_mask=block_mask[test_mask],
                torch=torch,
                device=device,
            )
            model_name = "relation_attention"
        else:
            feature_matrix = np.concatenate(feature_blocks, axis=1).astype(np.float32)
            model = _build_model(model_kind=model_kind, seed=seed)
            model.fit(feature_matrix[train_mask], labels[train_mask])
            val_probs = model.predict_proba(feature_matrix[val_mask])[:, 1]
            test_probs = model.predict_proba(feature_matrix[test_mask])[:, 1]
            model_name = f"relation_agg_{model_kind}"

        threshold = _select_threshold_from_validation(labels[val_mask], val_probs)

        val_metrics = _safe_binary_metrics(labels[val_mask], val_probs, threshold=threshold)
        test_metrics = _safe_binary_metrics(labels[test_mask], test_probs, threshold=threshold)

        results.append(
            {
                "review_encoder": review_encoder_name,
                "model_name": model_name,
                "edge_set": edge_set_name,
                "threshold": threshold,
                "val_auc": val_metrics["auc"],
                "val_ap": val_metrics["ap"],
                "auc": test_metrics["auc"],
                "ap": test_metrics["ap"],
                "recall": test_metrics["recall"],
                "precision": test_metrics["precision"],
                "f1": test_metrics["f1"],
                "accuracy": test_metrics["accuracy"],
                "num_train_users": int(train_mask.sum()),
                "num_val_users": int(val_mask.sum()),
                "num_test_users": int(test_mask.sum()),
                "num_fake_train": int(labels[train_mask].sum()),
                "num_fake_val": int(labels[val_mask].sum()),
                "num_fake_test": int(labels[test_mask].sum()),
            }
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / results_filename, index=False)
    return results_df
