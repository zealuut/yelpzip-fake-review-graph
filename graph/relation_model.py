from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
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

TEXTUAL_FEATURE_COLUMNS = [
    "avg_sentence_length",
    "lexical_diversity",
    "text_similarity",
    "pronoun_count",
    "self_reference_diversity",
    "affective_diversity",
    "cognitive_diversity",
    "perceptual_diversity",
]

EDGE_FEATURE_CANDIDATES = [
    "edge_weight",
    "pair_abnormal_score",
    "abnormal_score_src",
    "abnormal_score_dst",
    "abnormal_gate",
    "shared_product_count",
    "shared_time_count",
    "shared_user_count",
    "shared_entity_count",
    "S_product_idf",
    "S_rating",
    "S_time",
    "S_time_idf",
    "S_product",
    "S_text",
    "S_logic",
    "tau_logic",
]


@dataclass
class EdgePack:
    relation_name: str
    src: np.ndarray
    dst: np.ndarray
    weight: np.ndarray
    edge_features: np.ndarray
    active_nodes: np.ndarray


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


def _standardize_feature_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return ((matrix - mean) / std).astype(np.float32)


def _build_edge_pack(
    relation_name: str,
    edges_df: pd.DataFrame,
    user_index: dict[str, int],
    relation_topk: int | None = None,
) -> EdgePack:
    if edges_df is None or edges_df.empty:
        empty = np.zeros((0,), dtype=np.int64)
        return EdgePack(
            relation_name=relation_name,
            src=empty,
            dst=empty,
            weight=np.zeros((0,), dtype=np.float32),
            edge_features=np.zeros((0, 1), dtype=np.float32),
            active_nodes=np.zeros((len(user_index),), dtype=bool),
        )

    frame = edges_df.copy()
    frame["src_user_id"] = frame["src_user_id"].astype(str)
    frame["dst_user_id"] = frame["dst_user_id"].astype(str)
    valid = frame["src_user_id"].isin(user_index) & frame["dst_user_id"].isin(user_index)
    frame = frame[valid].reset_index(drop=True)
    if relation_topk is not None and relation_topk > 0 and not frame.empty:
        sort_candidates = [
            "edge_weight",
            "shared_entity_count",
            "shared_product_count",
            "shared_time_count",
            "shared_user_count",
            "S_logic",
            "S_text",
            "S_product",
            "S_time",
            "S_product_idf",
            "S_time_idf",
            "tau_logic",
        ]
        sort_column = next((column for column in sort_candidates if column in frame.columns), None)
        if sort_column is not None:
            frame[sort_column] = pd.to_numeric(frame[sort_column], errors="coerce").fillna(0.0)
            frame = frame.sort_values(
                by=["src_user_id", sort_column, "dst_user_id"],
                ascending=[True, False, True],
                kind="mergesort",
            )
        else:
            frame = frame.sort_values(by=["src_user_id", "dst_user_id"], ascending=[True, True], kind="mergesort")
        frame = frame.groupby("src_user_id", group_keys=False).head(int(relation_topk)).reset_index(drop=True)
    if frame.empty:
        empty = np.zeros((0,), dtype=np.int64)
        return EdgePack(
            relation_name=relation_name,
            src=empty,
            dst=empty,
            weight=np.zeros((0,), dtype=np.float32),
            edge_features=np.zeros((0, 1), dtype=np.float32),
            active_nodes=np.zeros((len(user_index),), dtype=bool),
        )

    src = frame["src_user_id"].map(user_index).to_numpy(dtype=np.int64)
    dst = frame["dst_user_id"].map(user_index).to_numpy(dtype=np.int64)
    weights = pd.to_numeric(frame.get("edge_weight", 1.0), errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=np.float32)
    edge_feature_columns = [column for column in EDGE_FEATURE_CANDIDATES if column in frame.columns]
    if not edge_feature_columns:
        edge_feature_columns = ["edge_weight"]
    edge_features = frame.reindex(columns=edge_feature_columns, fill_value=0.0).fillna(0.0).to_numpy(dtype=np.float32)
    active_nodes = np.zeros((len(user_index),), dtype=bool)
    active_nodes[src] = True
    active_nodes[dst] = True
    return EdgePack(
        relation_name=relation_name,
        src=src,
        dst=dst,
        weight=weights,
        edge_features=_standardize_feature_matrix(edge_features),
        active_nodes=active_nodes,
    )


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


def _segment_softmax(scores: Any, src_index: Any, num_nodes: int, torch: Any) -> Any:
    max_per_node = torch.full((num_nodes,), -1e9, dtype=scores.dtype, device=scores.device)
    max_per_node.scatter_reduce_(0, src_index, scores, reduce="amax", include_self=True)
    exp_scores = torch.exp(scores - max_per_node[src_index])
    denom = torch.zeros((num_nodes,), dtype=scores.dtype, device=scores.device)
    denom.index_add_(0, src_index, exp_scores)
    return exp_scores / (denom[src_index] + 1e-9)


class _RelationEdgeAttention(torch.nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.edge_proj = torch.nn.Linear(edge_dim, hidden_dim, bias=False)
        self.attn = torch.nn.Linear(hidden_dim * 3, 1, bias=False)
        self.out_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.norm = torch.nn.LayerNorm(hidden_dim)
        self.activation = torch.nn.LeakyReLU(0.2)

    def forward(self, node_repr: Any, edge_pack: EdgePack, torch: Any, abnormal_bias: Any | None = None) -> Any:
        if edge_pack.src.size == 0:
            return torch.zeros_like(node_repr)

        src = torch.as_tensor(edge_pack.src, dtype=torch.long, device=node_repr.device)
        dst = torch.as_tensor(edge_pack.dst, dtype=torch.long, device=node_repr.device)
        edge_features = torch.as_tensor(edge_pack.edge_features, dtype=node_repr.dtype, device=node_repr.device)
        weights = torch.as_tensor(edge_pack.weight, dtype=node_repr.dtype, device=node_repr.device).clamp(min=1e-6)

        src_repr = node_repr[src]
        dst_repr = node_repr[dst]
        edge_repr = self.edge_proj(edge_features)
        scores = self.attn(self.activation(torch.cat([src_repr, dst_repr, edge_repr], dim=-1))).squeeze(-1)
        scores = scores + torch.log(weights)
        if abnormal_bias is not None:
            if not torch.is_tensor(abnormal_bias):
                abnormal_bias = torch.as_tensor(abnormal_bias, dtype=scores.dtype, device=scores.device)
            else:
                abnormal_bias = abnormal_bias.to(device=scores.device, dtype=scores.dtype)
            scores = scores + abnormal_bias
        alpha = _segment_softmax(scores, src, node_repr.shape[0], torch)
        alpha = self.dropout(alpha)
        projected = self.out_proj(dst_repr)
        alpha = alpha.to(dtype=projected.dtype)
        messages = projected * alpha.unsqueeze(-1)
        out = torch.zeros_like(node_repr)
        out.index_add_(0, src, messages)
        return self.norm(out)


class _RouteGraphClassifier(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        relations: Sequence[str],
        edge_dim_map: dict[str, int],
        hidden_dim: int = 128,
        dropout: float = 0.2,
        senior_style: bool = False,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.relations = list(relations)
        self.hidden_dim = hidden_dim
        self.senior_style = senior_style
        self.num_layers = max(1, int(num_layers))
        classifier_input_dim = hidden_dim * 3 if (senior_style and self.num_layers > 1) else hidden_dim * 2
        self.input_proj = torch.nn.Linear(input_dim, hidden_dim)
        self.self_proj = torch.nn.Linear(input_dim, hidden_dim)
        self.relation_layers = torch.nn.ModuleDict(
            {
                relation: _RelationEdgeAttention(hidden_dim=hidden_dim, edge_dim=max(1, edge_dim_map.get(relation, 1)), dropout=dropout)
                for relation in self.relations
            }
        )
        self.relation_gate = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, len(self.relations)),
        )
        self.fuse1 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.LayerNorm(hidden_dim),
        )
        self.fuse2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.LayerNorm(hidden_dim),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.LayerNorm(classifier_input_dim),
            torch.nn.Linear(classifier_input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_features: Any, edge_packs: dict[str, EdgePack], torch: Any, abnormal_biases: dict[str, Any] | None = None) -> Any:
        self_repr = self.self_proj(node_features)
        core = self.input_proj(node_features)
        relation_outputs: list[Any] = []
        for relation_name in self.relations:
            pack = edge_packs.get(relation_name)
            if pack is None:
                continue
            bias = None if abnormal_biases is None else abnormal_biases.get(relation_name)
            relation_outputs.append(self.relation_layers[relation_name](core, pack, torch=torch, abnormal_bias=bias))

        if relation_outputs:
            relation_stack = torch.stack(relation_outputs, dim=1)
            gate = torch.softmax(self.relation_gate(torch.cat([self_repr, core], dim=-1)), dim=-1)
            relation_context = torch.sum(relation_stack * gate.unsqueeze(-1), dim=1)
        else:
            relation_context = torch.zeros_like(self_repr)

        stage1 = self.fuse1(torch.cat([self_repr, relation_context], dim=-1))
        if not self.senior_style or self.num_layers == 1:
            fused = torch.cat([self_repr, stage1], dim=-1)
            return self.classifier(fused).squeeze(-1)

        relation_outputs_2: list[Any] = []
        for relation_name in self.relations:
            pack = edge_packs.get(relation_name)
            if pack is None:
                continue
            bias = None if abnormal_biases is None else abnormal_biases.get(relation_name)
            relation_outputs_2.append(self.relation_layers[relation_name](stage1, pack, torch=torch, abnormal_bias=bias))
        if relation_outputs_2:
            relation_stack_2 = torch.stack(relation_outputs_2, dim=1)
            gate_2 = torch.softmax(self.relation_gate(torch.cat([self_repr, stage1], dim=-1)), dim=-1)
            relation_context_2 = torch.sum(relation_stack_2 * gate_2.unsqueeze(-1), dim=1)
        else:
            relation_context_2 = torch.zeros_like(self_repr)
        stage2 = self.fuse2(torch.cat([stage1, relation_context_2], dim=-1))
        fused = torch.cat([self_repr, stage1, stage2], dim=-1)
        return self.classifier(fused).squeeze(-1)


def _fit_graph_backbone_model(
    node_features: np.ndarray,
    edge_packs: dict[str, EdgePack],
    labels: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    seed: int,
    backbone: str,
    abnormal_biases: dict[str, np.ndarray] | None = None,
) -> tuple[Any, np.ndarray, np.ndarray]:
    torch, _, _ = _import_torch_modules()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(torch.cuda.is_available() and backbone == "senior_topk")
    if amp_enabled and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    base_input = _standardize_feature_matrix(node_features.astype(np.float32))
    input_dim = base_input.shape[1]
    if backbone == "senior_topk":
        hidden_dim = 64
        dropout = 0.30
        num_layers = 1
    else:
        hidden_dim = min(192, max(64, input_dim // 2))
        dropout = 0.25 if backbone == "senior_exact" else 0.20
        num_layers = 2 if backbone == "senior_exact" else 1
    edge_dim_map = {relation: max(1, pack.edge_features.shape[1]) for relation, pack in edge_packs.items()}
    senior_style = backbone in {"senior_exact", "senior_topk"}
    model = _RouteGraphClassifier(
        input_dim=input_dim,
        relations=sorted(edge_packs.keys()),
        edge_dim_map=edge_dim_map,
        hidden_dim=hidden_dim,
        dropout=dropout,
        senior_style=senior_style,
        num_layers=num_layers,
    ).to(device)

    feature_tensor = torch.as_tensor(base_input, dtype=torch.float32, device=device)
    y_train = labels[train_mask].astype(np.float32)
    pos_count = float(y_train.sum())
    neg_count = float(len(y_train) - pos_count)
    pos_weight = torch.tensor([neg_count / max(pos_count, 1.0)], dtype=torch.float32, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4 if backbone == "senior_topk" else 1e-3, weight_decay=5e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_state = None
    best_val_auc = -1.0
    bad_epochs = 0
    max_epochs = 120 if backbone == "senior_topk" else (140 if senior_style else 100)
    patience = 18 if backbone == "senior_topk" else (20 if senior_style else 16)
    rng = np.random.default_rng(seed)

    for _epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = model(feature_tensor, edge_packs, torch=torch, abnormal_biases=abnormal_biases)
            train_logits = logits[train_mask]
            train_labels = torch.as_tensor(labels[train_mask], dtype=torch.float32, device=device)
            loss = criterion(train_logits, train_labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                val_logits = model(feature_tensor, edge_packs, torch=torch, abnormal_biases=abnormal_biases)[val_mask]
            val_probs = torch.sigmoid(val_logits).detach().cpu().numpy()
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
    return model, base_input, feature_tensor.detach().cpu().numpy()


def _predict_graph_backbone_probs(
    model: Any,
    node_features: np.ndarray,
    edge_packs: dict[str, EdgePack],
    abnormal_biases: dict[str, np.ndarray] | None,
    mask: np.ndarray,
    torch: Any,
    device: Any,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        features = torch.as_tensor(node_features, dtype=torch.float32, device=device)
        bias_tensors = None
        if abnormal_biases is not None:
            bias_tensors = {
                relation: torch.as_tensor(values, dtype=torch.float32, device=device)
                for relation, values in abnormal_biases.items()
            }
        logits = model(features, edge_packs, torch=torch, abnormal_biases=bias_tensors)
        return torch.sigmoid(logits[mask]).detach().cpu().numpy()


def run_relation_aggregation_experiments(
    user_df: pd.DataFrame,
    self_features: np.ndarray,
    edge_frames: dict[str, pd.DataFrame],
    output_dir: str | Path,
    review_encoder_name: str,
    model_kind: str,
    seed: int,
    results_filename: str = "model_results.csv",
    backbone: str = "current_relation",
    relation_model: str = "mean",
    use_abnormal_edge_weight: bool = False,
    use_abnormal_gate: bool = False,
    use_abnormal_attention_bias: bool = False,
    abnormal_score_source: str = "auto",
    abnormal_edge_lambda: float = 1.0,
    abnormal_attention_gamma: float = 1.0,
    review_scores_df: pd.DataFrame | None = None,
    selected_edge_set: str | None = None,
    relation_topk: int | None = None,
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

    user_abnormal_scores = None
    if review_scores_df is not None:
        try:
            from .graph_pipeline import build_user_abnormal_score_vector, annotate_edges_with_pair_scores, apply_abnormal_score_edge_transform

            user_abnormal_scores = build_user_abnormal_score_vector(
                user_df=user_df,
                review_scores_df=review_scores_df,
                source=abnormal_score_source,
                aggregate="mean",
                top_k=3,
            )
            if use_abnormal_edge_weight or use_abnormal_gate:
                edge_frames = apply_abnormal_score_edge_transform(
                    edge_frames=edge_frames,
                    user_df=user_df,
                    user_abnormal_scores=user_abnormal_scores,
                    abnormal_edge_lambda=abnormal_edge_lambda,
                    use_abnormal_gate=use_abnormal_gate,
                )
            if use_abnormal_attention_bias:
                edge_frames = annotate_edges_with_pair_scores(
                    edge_frames=edge_frames,
                    user_df=user_df,
                    user_abnormal_scores=user_abnormal_scores,
                )
        except Exception:
            user_abnormal_scores = None

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

    edge_set_items = (
        [(selected_edge_set, EDGE_SET_DEFINITIONS[selected_edge_set])]
        if selected_edge_set in EDGE_SET_DEFINITIONS
        else list(EDGE_SET_DEFINITIONS.items())
    )

    for edge_set_name, relations in edge_set_items:
        feature_blocks = [self_features]
        active_masks = [np.ones(self_features.shape[0], dtype=bool)]
        for relation_name in relations:
            feature_blocks.append(relation_aggregates[relation_name])
            active_masks.append(relation_active_masks[relation_name])

        if backbone in {"current_egat", "senior_exact", "senior_topk"}:
            edge_packs = {
                relation: _build_edge_pack(
                    relation,
                    edge_frames.get(relation, pd.DataFrame()),
                    user_index,
                    relation_topk=relation_topk if backbone == "senior_topk" else None,
                )
                for relation in relations
            }
            abnormal_biases = None
            if use_abnormal_attention_bias and user_abnormal_scores is not None:
                abnormal_biases = {}
                for relation in relations:
                    frame = edge_frames.get(relation, pd.DataFrame())
                    if frame.empty or "pair_abnormal_score" not in frame.columns:
                        abnormal_biases[relation] = np.zeros((0,), dtype=np.float32)
                    else:
                        abnormal_biases[relation] = (
                            frame["pair_abnormal_score"].fillna(0.0).astype(float).to_numpy(dtype=np.float32)
                            * float(abnormal_attention_gamma)
                        )
            model, base_input, _ = _fit_graph_backbone_model(
                node_features=self_features,
                edge_packs=edge_packs,
                labels=labels,
                train_mask=train_mask,
                val_mask=val_mask,
                seed=seed,
                backbone=backbone,
                abnormal_biases=abnormal_biases,
            )
            torch_runtime, _, _ = _import_torch_modules()
            device = next(model.parameters()).device
            val_probs = _predict_graph_backbone_probs(
                model=model,
                node_features=base_input,
                edge_packs=edge_packs,
                abnormal_biases=abnormal_biases,
                mask=val_mask,
                torch=torch_runtime,
                device=device,
            )
            test_probs = _predict_graph_backbone_probs(
                model=model,
                node_features=base_input,
                edge_packs=edge_packs,
                abnormal_biases=abnormal_biases,
                mask=test_mask,
                torch=torch_runtime,
                device=device,
            )
            model_name = f"{backbone}_{relation_model}"
        elif model_kind == "relation_attn":
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
                "backbone": backbone,
                "relation_model": relation_model,
                "use_abnormal_edge_weight": bool(use_abnormal_edge_weight),
                "use_abnormal_gate": bool(use_abnormal_gate),
                "use_abnormal_attention_bias": bool(use_abnormal_attention_bias),
                "abnormal_score_source": abnormal_score_source,
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
