from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


RELATION_ID = {"UPU": 0, "UTU": 1, "USU": 2}
NON_FEATURE_COLUMNS = {
    "user_id",
    "user_label",
    "label",
    "review_label",
    "is_fake",
    "is_real",
    "is_fake_user",
    "fake_ratio",
    "fake_review_count",
    "split",
    "product_set",
    "time_bucket_set",
}
EXCLUDED_FEATURE_TOKENS = (
    "abnormal",
    "logic",
    "llm",
    "mask",
    "tns",
    "head",
    "gate",
    "embedding",
    "vector",
    "text",
)


@dataclass
class GraphBatch:
    x: torch.Tensor
    src: torch.Tensor
    dst: torch.Tensor
    relation_id: torch.Tensor | None
    num_nodes: int


@dataclass
class Bundle:
    node_features: np.ndarray
    labels: np.ndarray
    splits: np.ndarray
    user_ids: list[str]
    union_edges: pd.DataFrame
    relation_edges: pd.DataFrame
    feature_columns: list[str]
    excluded_columns: dict[str, list[str]]
    edge_stats: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_binary_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, float]:
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs).astype(np.float32)
    preds = (probs >= threshold).astype(int)
    unique = np.unique(labels)
    return {
        "auc": float(roc_auc_score(labels, probs)) if len(unique) > 1 else 0.0,
        "ap": float(average_precision_score(labels, probs)) if len(unique) > 1 else 0.0,
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "accuracy": float(accuracy_score(labels, preds)),
    }


def select_threshold_from_validation(labels: np.ndarray, probs: np.ndarray) -> float:
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs).astype(np.float32)
    thresholds = np.concatenate([[0.5], np.unique(np.clip(probs, 0.01, 0.99))])
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in thresholds:
        f1 = f1_score(labels, (probs >= threshold).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_threshold = float(threshold)
    return best_threshold


def standardize_train_only(matrix: np.ndarray, splits: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    train_mask = splits.astype(str) == "train"
    mean = matrix[train_mask].mean(axis=0, keepdims=True)
    std = matrix[train_mask].std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return ((matrix - mean) / std).astype(np.float32)


def load_features(full_base_dir: Path) -> tuple[pd.DataFrame, np.ndarray, list[str], dict[str, list[str]]]:
    user_df = pd.read_csv(full_base_dir / "logic_vectors" / "user_summary.csv")
    user_df["user_id"] = user_df["user_id"].astype(str)
    user_df["split"] = user_df["split"].astype(str)
    feature_columns: list[str] = []
    excluded: dict[str, list[str]] = {"non_feature": [], "abnormal_text_head": [], "non_numeric": []}
    for column in user_df.columns:
        lower = column.lower()
        if column in NON_FEATURE_COLUMNS:
            excluded["non_feature"].append(column)
            continue
        if any(token in lower for token in EXCLUDED_FEATURE_TOKENS):
            excluded["abnormal_text_head"].append(column)
            continue
        if not pd.api.types.is_numeric_dtype(user_df[column]):
            excluded["non_numeric"].append(column)
            continue
        feature_columns.append(column)
    if not feature_columns:
        raise ValueError("No clean numeric features were selected.")
    raw = user_df[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    features = standardize_train_only(raw, user_df["split"].astype(str).to_numpy())
    return user_df, features, feature_columns, excluded


def load_edges(full_base_dir: Path, user_index: dict[str, int], relations: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    union_parts: list[pd.DataFrame] = []
    relation_parts: list[pd.DataFrame] = []
    edge_stat_rows: list[dict[str, Any]] = []
    for relation in relations:
        frame = pd.read_csv(full_base_dir / "edges" / f"{relation}_edges.csv", usecols=["src_user_id", "dst_user_id"])
        frame["src_user_id"] = frame["src_user_id"].astype(str)
        frame["dst_user_id"] = frame["dst_user_id"].astype(str)
        frame = frame[frame["src_user_id"].isin(user_index) & frame["dst_user_id"].isin(user_index)].copy()
        frame["src_index"] = frame["src_user_id"].map(user_index).astype(np.int64)
        frame["dst_index"] = frame["dst_user_id"].map(user_index).astype(np.int64)
        frame["relation_name"] = relation
        frame["relation_id"] = RELATION_ID[relation]
        edge_stat_rows.append(
            {
                "relation": relation,
                "num_edges": int(len(frame)),
                "num_src_users": int(frame["src_user_id"].nunique()),
                "num_dst_users": int(frame["dst_user_id"].nunique()),
            }
        )
        union_parts.append(frame[["src_user_id", "dst_user_id", "src_index", "dst_index"]])
        relation_parts.append(frame[["src_user_id", "dst_user_id", "src_index", "dst_index", "relation_name", "relation_id"]])
    union = pd.concat(union_parts, ignore_index=True)
    union = union.drop_duplicates(["src_index", "dst_index"]).sort_values(["src_index", "dst_index"], kind="mergesort").reset_index(drop=True)
    relation_edges = pd.concat(relation_parts, ignore_index=True).sort_values(["relation_id", "src_index", "dst_index"], kind="mergesort").reset_index(drop=True)
    edge_stats = pd.DataFrame(edge_stat_rows)
    edge_stats.loc[len(edge_stats)] = {
        "relation": "homogeneous_union",
        "num_edges": int(len(union)),
        "num_src_users": int(union["src_user_id"].nunique()),
        "num_dst_users": int(union["dst_user_id"].nunique()),
    }
    return union, relation_edges, edge_stats


def load_bundle(config: dict[str, Any]) -> Bundle:
    full_base_dir = Path(config["full_base_dir"])
    user_df, features, feature_columns, excluded = load_features(full_base_dir)
    user_ids = user_df["user_id"].astype(str).tolist()
    user_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
    relations = [str(item) for item in config["relations"]]
    union_edges, relation_edges, edge_stats = load_edges(full_base_dir, user_index, relations)
    return Bundle(
        node_features=features,
        labels=pd.to_numeric(user_df["user_label"], errors="coerce").fillna(0).to_numpy(dtype=np.int64),
        splits=user_df["split"].astype(str).to_numpy(),
        user_ids=user_ids,
        union_edges=union_edges,
        relation_edges=relation_edges,
        feature_columns=feature_columns,
        excluded_columns=excluded,
        edge_stats=edge_stats,
    )


def segment_softmax(scores: torch.Tensor, target_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    if scores.ndim == 1:
        max_per_node = torch.full((num_nodes,), -1e9, dtype=scores.dtype, device=scores.device)
        max_per_node.scatter_reduce_(0, target_index, scores, reduce="amax", include_self=True)
        exp_scores = torch.exp(scores - max_per_node[target_index])
        denom = torch.zeros((num_nodes,), dtype=scores.dtype, device=scores.device)
        denom.index_add_(0, target_index, exp_scores)
        return exp_scores / (denom[target_index] + 1e-9)
    return torch.stack([segment_softmax(scores[:, idx], target_index, num_nodes) for idx in range(scores.shape[1])], dim=1)


class StandardGATLayer(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, heads: int, dropout: float, activation: bool) -> None:
        super().__init__()
        if output_dim % heads != 0:
            raise ValueError("output_dim must be divisible by heads.")
        self.heads = heads
        self.head_dim = output_dim // heads
        self.linear = torch.nn.Linear(input_dim, output_dim, bias=False)
        self.attn_src = torch.nn.Parameter(torch.empty(heads, self.head_dim))
        self.attn_dst = torch.nn.Parameter(torch.empty(heads, self.head_dim))
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.ELU() if activation else torch.nn.Identity()
        torch.nn.init.xavier_uniform_(self.linear.weight)
        torch.nn.init.xavier_uniform_(self.attn_src)
        torch.nn.init.xavier_uniform_(self.attn_dst)

    def forward(self, x: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
        loops = torch.arange(num_nodes, device=x.device, dtype=torch.long)
        src_all = torch.cat([src, loops], dim=0)
        dst_all = torch.cat([dst, loops], dim=0)
        h = self.linear(self.dropout(x)).view(num_nodes, self.heads, self.head_dim)
        h_target = h[src_all]
        h_neighbor = h[dst_all]
        scores = torch.nn.functional.leaky_relu(
            (h_target * self.attn_src.unsqueeze(0)).sum(-1) + (h_neighbor * self.attn_dst.unsqueeze(0)).sum(-1),
            negative_slope=0.2,
        )
        alpha = self.dropout(segment_softmax(scores, src_all, num_nodes))
        out = torch.zeros((num_nodes, self.heads, self.head_dim), dtype=x.dtype, device=x.device)
        for head in range(self.heads):
            msg = h_neighbor[:, head, :] * alpha[:, head].unsqueeze(-1)
            out[:, head, :].index_add_(0, src_all, msg)
        return self.activation(out.reshape(num_nodes, self.heads * self.head_dim))


class StandardGAT(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, heads: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        dims = [input_dim] + [hidden_dim] * num_layers
        self.layers = torch.nn.ModuleList(
            [
                StandardGATLayer(dims[idx], dims[idx + 1], heads=heads, dropout=dropout, activation=idx < num_layers - 1)
                for idx in range(num_layers)
            ]
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(hidden_dim, 1)

    def forward(self, batch: GraphBatch) -> torch.Tensor:
        x = batch.x
        for layer in self.layers:
            x = layer(x, batch.src, batch.dst, batch.num_nodes)
        return self.classifier(self.dropout(x)).squeeze(-1)


class StandardGraphSAGELayer(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(input_dim * 2, output_dim)

    def forward(self, x: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
        msg = x[dst]
        agg = torch.zeros((num_nodes, x.shape[1]), dtype=x.dtype, device=x.device)
        deg = torch.zeros((num_nodes, 1), dtype=x.dtype, device=x.device)
        agg.index_add_(0, src, msg)
        deg.index_add_(0, src, torch.ones((src.shape[0], 1), dtype=x.dtype, device=x.device))
        neigh = agg / deg.clamp(min=1.0)
        return self.linear(torch.cat([x, neigh], dim=1))


class StandardGraphSAGE(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        dims = [input_dim] + [hidden_dim] * num_layers
        self.layers = torch.nn.ModuleList([StandardGraphSAGELayer(dims[idx], dims[idx + 1]) for idx in range(num_layers)])
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.ReLU()
        self.classifier = torch.nn.Linear(hidden_dim, 1)

    def forward(self, batch: GraphBatch) -> torch.Tensor:
        x = batch.x
        for idx, layer in enumerate(self.layers):
            x = layer(x, batch.src, batch.dst, batch.num_nodes)
            if idx < len(self.layers) - 1:
                x = self.dropout(self.activation(x))
        return self.classifier(self.dropout(x)).squeeze(-1)


class StandardRGCNLayer(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_relations: int, num_bases: int) -> None:
        super().__init__()
        self.num_relations = num_relations
        self.num_bases = min(max(1, num_bases), num_relations)
        self.bases = torch.nn.Parameter(torch.empty(self.num_bases, input_dim, output_dim))
        self.coefficients = torch.nn.Parameter(torch.empty(num_relations, self.num_bases))
        self.self_loop = torch.nn.Linear(input_dim, output_dim)
        torch.nn.init.xavier_uniform_(self.bases)
        torch.nn.init.xavier_uniform_(self.coefficients)

    def forward(self, x: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, rel: torch.Tensor, num_nodes: int) -> torch.Tensor:
        out = self.self_loop(x)
        weights = torch.einsum("rb,bio->rio", self.coefficients, self.bases)
        for relation_id in range(self.num_relations):
            mask = rel == relation_id
            if int(mask.sum()) == 0:
                continue
            src_r = src[mask]
            dst_r = dst[mask]
            msg = x[dst_r] @ weights[relation_id]
            agg = torch.zeros((num_nodes, weights.shape[-1]), dtype=x.dtype, device=x.device)
            deg = torch.zeros((num_nodes, 1), dtype=x.dtype, device=x.device)
            agg.index_add_(0, src_r, msg)
            deg.index_add_(0, src_r, torch.ones((src_r.shape[0], 1), dtype=x.dtype, device=x.device))
            out = out + agg / deg.clamp(min=1.0)
        return out


class StandardRGCN(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_relations: int, num_bases: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        dims = [input_dim] + [hidden_dim] * num_layers
        self.layers = torch.nn.ModuleList(
            [StandardRGCNLayer(dims[idx], dims[idx + 1], num_relations=num_relations, num_bases=num_bases) for idx in range(num_layers)]
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.ReLU()
        self.classifier = torch.nn.Linear(hidden_dim, 1)

    def forward(self, batch: GraphBatch) -> torch.Tensor:
        if batch.relation_id is None:
            raise ValueError("RGCN requires relation ids.")
        x = batch.x
        for idx, layer in enumerate(self.layers):
            x = layer(x, batch.src, batch.dst, batch.relation_id, batch.num_nodes)
            if idx < len(self.layers) - 1:
                x = self.dropout(self.activation(x))
        return self.classifier(self.dropout(x)).squeeze(-1)


def build_batch(bundle: Bundle, config: dict[str, Any], device: torch.device) -> GraphBatch:
    model = str(config["model"]).lower()
    if model == "rgcn":
        frame = bundle.relation_edges
        rel = torch.as_tensor(frame["relation_id"].to_numpy(np.int64), dtype=torch.long, device=device)
    else:
        frame = bundle.union_edges
        rel = None
    return GraphBatch(
        x=torch.as_tensor(bundle.node_features, dtype=torch.float32, device=device),
        src=torch.as_tensor(frame["src_index"].to_numpy(np.int64), dtype=torch.long, device=device),
        dst=torch.as_tensor(frame["dst_index"].to_numpy(np.int64), dtype=torch.long, device=device),
        relation_id=rel,
        num_nodes=len(bundle.user_ids),
    )


def build_model(config: dict[str, Any], input_dim: int) -> torch.nn.Module:
    model = str(config["model"]).lower()
    if model == "gat":
        return StandardGAT(input_dim, int(config["hidden_dim"]), int(config["heads"]), int(config["num_layers"]), float(config["dropout"]))
    if model == "graphsage":
        return StandardGraphSAGE(input_dim, int(config["hidden_dim"]), int(config["num_layers"]), float(config["dropout"]))
    if model == "rgcn":
        return StandardRGCN(input_dim, int(config["hidden_dim"]), len(RELATION_ID), int(config["num_bases"]), int(config["num_layers"]), float(config["dropout"]))
    raise ValueError(f"Unsupported model: {config['model']}")


def main() -> None:
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    output_root = Path(args.output_root)
    exp_dir = output_root / config["experiment_name"]
    metrics_dir = exp_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(int(config["seed"]))
    bundle = load_bundle(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = build_batch(bundle, config, device)
    model = build_model(config, input_dim=bundle.node_features.shape[1]).to(device)

    labels = torch.as_tensor(bundle.labels.astype(np.float32), dtype=torch.float32, device=device)
    splits = bundle.splits.astype(str)
    train_mask = torch.as_tensor(splits == "train", dtype=torch.bool, device=device)
    val_mask = torch.as_tensor(splits == "val", dtype=torch.bool, device=device)
    test_mask = torch.as_tensor(splits == "test", dtype=torch.bool, device=device)
    y_train = bundle.labels[splits == "train"].astype(np.float32)
    pos_count = float(y_train.sum())
    neg_count = float(len(y_train) - pos_count)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg_count / max(pos_count, 1.0)], dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))

    best_state = None
    best_epoch = -1
    best_val_auc = -1.0
    patience_left = int(config["patience"])
    epoch_rows: list[dict[str, Any]] = []
    print(f"start experiment={config['experiment_name']} model={config['model']} device={device} features={len(bundle.feature_columns)} edges={len(batch.src)}", flush=True)

    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = criterion(logits[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(batch)).detach().cpu().numpy()
        val_metrics = safe_binary_metrics(bundle.labels[splits == "val"], probs[splits == "val"], threshold=0.5)
        improved = val_metrics["auc"] > best_val_auc + 1e-5
        if improved:
            best_val_auc = float(val_metrics["auc"])
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_left = int(config["patience"])
        else:
            patience_left -= 1
        epoch_rows.append(
            {
                "epoch": epoch,
                "train_loss": float(loss.detach().cpu().item()),
                "val_auc": float(val_metrics["auc"]),
                "val_ap": float(val_metrics["ap"]),
                "improved": bool(improved),
                "patience_left": int(patience_left),
            }
        )
        print(
            f"epoch={epoch} loss={float(loss.detach().cpu().item()):.6f} val_auc={val_metrics['auc']:.6f} val_ap={val_metrics['ap']:.6f} improved={improved} patience={patience_left}",
            flush=True,
        )
        if patience_left <= 0:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(batch)).detach().cpu().numpy()

    val_labels = bundle.labels[splits == "val"]
    val_probs = probs[splits == "val"]
    test_labels = bundle.labels[splits == "test"]
    test_probs = probs[splits == "test"]
    threshold = select_threshold_from_validation(val_labels, val_probs)
    val_metrics = safe_binary_metrics(val_labels, val_probs, threshold)
    test_metrics = safe_binary_metrics(test_labels, test_probs, threshold)
    split_stats = {
        "num_train": int((splits == "train").sum()),
        "num_val": int((splits == "val").sum()),
        "num_test": int((splits == "test").sum()),
        "num_fake_train": int(bundle.labels[splits == "train"].sum()),
        "num_fake_val": int(bundle.labels[splits == "val"].sum()),
        "num_fake_test": int(bundle.labels[splits == "test"].sum()),
    }

    result = {
        "experiment_name": config["experiment_name"],
        "model": config["model"],
        "graph_protocol": config["graph_protocol"],
        "relations": ",".join(config["relations"]),
        "relation_handling": config["relation_handling"],
        "feature_policy": config["feature_policy"],
        "feature_source": config["feature_source"],
        "feature_dim": int(bundle.node_features.shape[1]),
        "num_users": int(len(bundle.user_ids)),
        "num_edges": int(len(batch.src)),
        "hidden_dim": int(config["hidden_dim"]),
        "num_layers": int(config["num_layers"]),
        "heads": config.get("heads", "NA"),
        "num_bases": config.get("num_bases", "NA"),
        "lr": float(config["lr"]),
        "weight_decay": float(config["weight_decay"]),
        "dropout": float(config["dropout"]),
        "epochs": int(config["epochs"]),
        "patience": int(config["patience"]),
        "best_epoch": int(best_epoch),
        "threshold": float(threshold),
        "val_auc": float(val_metrics["auc"]),
        "val_ap": float(val_metrics["ap"]),
        "auc": float(test_metrics["auc"]),
        "ap": float(test_metrics["ap"]),
        "f1": float(test_metrics["f1"]),
        "precision": float(test_metrics["precision"]),
        "recall": float(test_metrics["recall"]),
        "accuracy": float(test_metrics["accuracy"]),
        "notes": config["notes"],
    }

    pd.DataFrame(epoch_rows).to_csv(metrics_dir / "epoch_metrics.csv", index=False)
    pd.DataFrame([result]).to_csv(metrics_dir / "model_results.csv", index=False)
    pd.DataFrame(
        {
            "user_id": np.asarray(bundle.user_ids)[test_mask.detach().cpu().numpy()],
            "label": test_labels.astype(int),
            "prob": test_probs.astype(np.float32),
            "pred": (test_probs >= threshold).astype(int),
            "split": "test",
        }
    ).to_csv(metrics_dir / "test_predictions.csv", index=False)
    bundle.edge_stats.to_csv(metrics_dir / "edge_stats.csv", index=False)
    (exp_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    (exp_dir / "feature_manifest.json").write_text(
        json.dumps(
            {
                "feature_columns": bundle.feature_columns,
                "excluded_columns": bundle.excluded_columns,
                "excluded_feature_tokens": list(EXCLUDED_FEATURE_TOKENS),
                "standardization": "fit mean/std on train split only",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (exp_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "config": config,
                "metrics": result,
                "split_stats": split_stats,
                "feature_columns": bundle.feature_columns,
                "excluded_columns": bundle.excluded_columns,
                "edge_stats": bundle.edge_stats.to_dict(orient="records"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"finished experiment={config['experiment_name']} auc={result['auc']:.6f} ap={result['ap']:.6f} f1={result['f1']:.6f}", flush=True)


if __name__ == "__main__":
    main()
