from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .baseline_models import (
    GATBaseline,
    GATCurrentTopK,
    GCNBaseline,
    GraphBatch,
    GraphSAGEBaseline,
    GraphSAGECurrentTopK,
    RGCNBaseline,
    RGCNCurrentTopK,
)
from .build_full_base_graph import FULL_BASE_RELATIONS, load_full_base_bundle, write_full_base_edge_stats
from .data_loader import EDGE_TYPES, load_protocol_bundle, write_edge_stats
from .metrics import epoch_frame, safe_binary_metrics, select_threshold_from_validation
from .utils import load_yaml, save_json, save_yaml, seed_everything, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a baseline model under the current top-k protocol.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def _build_batch(bundle: Any, relation_handling: str, device: torch.device) -> tuple[GraphBatch, int]:
    if relation_handling == "typed_multi_relation_graph":
        frame = bundle.relation_edges
        batch = GraphBatch(
            x=torch.as_tensor(bundle.node_features, dtype=torch.float32, device=device),
            src=torch.as_tensor(frame["src_index"].to_numpy(dtype=np.int64), dtype=torch.long, device=device),
            dst=torch.as_tensor(frame["dst_index"].to_numpy(dtype=np.int64), dtype=torch.long, device=device),
            relation_id=torch.as_tensor(frame["relation_id"].to_numpy(dtype=np.int64), dtype=torch.long, device=device),
            num_nodes=len(bundle.user_ids),
        )
        return batch, len(frame)

    frame = bundle.union_edges
    batch = GraphBatch(
        x=torch.as_tensor(bundle.node_features, dtype=torch.float32, device=device),
        src=torch.as_tensor(frame["src_index"].to_numpy(dtype=np.int64), dtype=torch.long, device=device),
        dst=torch.as_tensor(frame["dst_index"].to_numpy(dtype=np.int64), dtype=torch.long, device=device),
        relation_id=None,
        num_nodes=len(bundle.user_ids),
    )
    return batch, len(frame)


def _build_model(config: dict[str, Any], input_dim: int, num_relations: int) -> torch.nn.Module:
    model_name = str(config["model"]).lower()
    hidden_dim = int(config["hidden_dim"])
    num_layers = int(config.get("num_layers", 1))
    dropout = float(config["dropout"])
    if model_name == "gat":
        heads = int(config["heads"])
        if num_layers == 1:
            return GATCurrentTopK(input_dim=input_dim, hidden_dim=hidden_dim, heads=heads, dropout=dropout)
        return GATBaseline(input_dim=input_dim, hidden_dim=hidden_dim, heads=heads, num_layers=num_layers, dropout=dropout)
    if model_name == "graphsage":
        if num_layers == 1:
            return GraphSAGECurrentTopK(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
        return GraphSAGEBaseline(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    if model_name == "gcn":
        return GCNBaseline(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    if model_name == "rgcn":
        num_bases = int(config["num_bases"])
        if num_layers == 1:
            return RGCNCurrentTopK(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_relations=num_relations,
                num_bases=num_bases,
                dropout=dropout,
            )
        return RGCNBaseline(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_relations=num_relations,
            num_bases=num_bases,
            num_layers=num_layers,
            dropout=dropout,
        )
    raise ValueError(f"Unsupported model: {config['model']}")


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    output_root = Path(args.output_root)
    experiment_dir = output_root / config["experiment_name"]
    metrics_dir = experiment_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(experiment_dir / "train.log")

    graph_protocol = str(config.get("graph_protocol", "current_topk"))
    if graph_protocol == "full_base":
        feature_mode = str(config.get("full_base_feature_mode", "full_base_summary"))
        bundle = load_full_base_bundle(feature_mode=feature_mode)
        edge_stats = write_full_base_edge_stats(bundle, experiment_dir)
        if "edge_type" in edge_stats.columns and "relation" not in edge_stats.columns:
            edge_stats = edge_stats.rename(columns={"edge_type": "relation"})
            edge_stats.to_csv(metrics_dir / "edge_stats.csv", index=False)
        relation_names = FULL_BASE_RELATIONS
    else:
        bundle = load_protocol_bundle()
        edge_stats = write_edge_stats(bundle, experiment_dir)
        relation_names = EDGE_TYPES

    seed = int(config["seed"])
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch, num_edges = _build_batch(bundle, config["relation_handling"], device)
    model = _build_model(config, input_dim=bundle.node_features.shape[1], num_relations=len(relation_names)).to(device)
    logger.info("protocol=%s model=%s device=%s num_nodes=%s num_edges=%s", graph_protocol, config["model"], device, len(bundle.user_ids), num_edges)

    labels = torch.as_tensor(bundle.labels.astype(np.float32), dtype=torch.float32, device=device)
    train_mask = torch.as_tensor(bundle.splits == "train", dtype=torch.bool, device=device)
    val_mask = torch.as_tensor(bundle.splits == "val", dtype=torch.bool, device=device)
    test_mask = torch.as_tensor(bundle.splits == "test", dtype=torch.bool, device=device)

    y_train = bundle.labels[bundle.splits == "train"].astype(np.float32)
    pos_count = float(y_train.sum())
    neg_count = float(len(y_train) - pos_count)
    pos_weight = torch.tensor([neg_count / max(pos_count, 1.0)], dtype=torch.float32, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))

    best_state = None
    best_val_auc = -1.0
    best_epoch = -1
    patience_left = int(config["patience"])
    epoch_rows: list[dict[str, Any]] = []

    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        train_loss = criterion(logits[train_mask], labels[train_mask])
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(batch)
            val_probs = torch.sigmoid(logits[val_mask]).detach().cpu().numpy()
        val_metrics = safe_binary_metrics(bundle.labels[bundle.splits == "val"], val_probs, threshold=0.5)
        improved = val_metrics["auc"] > best_val_auc + 1e-5
        if improved:
            best_val_auc = float(val_metrics["auc"])
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = int(config["patience"])
        else:
            patience_left -= 1
        epoch_rows.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss.detach().cpu().item()),
                "val_auc": float(val_metrics["auc"]),
                "val_ap": float(val_metrics["ap"]),
                "improved": bool(improved),
                "patience_left": int(patience_left),
            }
        )
        logger.info(
            "epoch=%s train_loss=%.6f val_auc=%.6f val_ap=%.6f improved=%s patience_left=%s",
            epoch,
            float(train_loss.detach().cpu().item()),
            float(val_metrics["auc"]),
            float(val_metrics["ap"]),
            improved,
            patience_left,
        )
        if patience_left <= 0:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits = model(batch)
        probs = torch.sigmoid(logits).detach().cpu().numpy()

    val_labels = bundle.labels[bundle.splits == "val"]
    val_probs = probs[bundle.splits == "val"]
    test_labels = bundle.labels[bundle.splits == "test"]
    test_probs = probs[bundle.splits == "test"]
    test_threshold = select_threshold_from_validation(val_labels, val_probs)
    val_metrics = safe_binary_metrics(val_labels, val_probs, threshold=test_threshold)
    test_metrics = safe_binary_metrics(test_labels, test_probs, threshold=test_threshold)

    test_pred_frame = pd.DataFrame(
        {
            "user_id": np.asarray(bundle.user_ids)[bundle.splits == "test"],
            "label": test_labels.astype(int),
            "prob": test_probs.astype(np.float32),
            "score_or_prob": test_probs.astype(np.float32),
            "pred": (test_probs >= test_threshold).astype(int),
            "split": "test",
        }
    )
    test_pred_frame.to_csv(metrics_dir / "test_predictions.csv", index=False)
    epoch_frame(epoch_rows).to_csv(metrics_dir / "epoch_metrics.csv", index=False)

    split_stats = getattr(bundle, "split_stats", {})
    feature_dim = int(getattr(bundle, "feature_dim", bundle.node_features.shape[1]))
    blocked_label_columns = getattr(bundle, "blocked_label_columns", "UNKNOWN_FROM_D1")
    if isinstance(blocked_label_columns, list):
        blocked_label_columns = ",".join(blocked_label_columns)
    total_edges = int(edge_stats["num_edges"].sum()) if len(edge_stats) else int(num_edges)
    result_row = {
        "experiment_name": config["experiment_name"],
        "model": config["model"],
        "graph_protocol": config["graph_protocol"],
        "relations": config.get("relations", config.get("edge_set", "UNKNOWN")),
        "relation_handling": config["relation_handling"],
        "feature_source": config["feature_source"],
        "feature_dim": feature_dim,
        "blocked_label_columns": blocked_label_columns,
        "num_users": len(bundle.user_ids),
        "num_train": int(split_stats.get("num_train", int((bundle.splits == "train").sum()))),
        "num_val": int(split_stats.get("num_val", int((bundle.splits == "val").sum()))),
        "num_test": int(split_stats.get("num_test", int((bundle.splits == "test").sum()))),
        "num_fake_train": int(split_stats.get("num_fake_train", int(bundle.labels[bundle.splits == "train"].sum()))),
        "num_real_train": int(split_stats.get("num_real_train", int((bundle.splits == "train").sum() - bundle.labels[bundle.splits == "train"].sum()))),
        "num_fake_val": int(split_stats.get("num_fake_val", int(bundle.labels[bundle.splits == "val"].sum()))),
        "num_real_val": int(split_stats.get("num_real_val", int((bundle.splits == "val").sum() - bundle.labels[bundle.splits == "val"].sum()))),
        "num_fake_test": int(split_stats.get("num_fake_test", int(bundle.labels[bundle.splits == "test"].sum()))),
        "num_real_test": int(split_stats.get("num_real_test", int((bundle.splits == "test").sum() - bundle.labels[bundle.splits == "test"].sum()))),
        "num_edges": total_edges,
        "hidden_dim": int(config["hidden_dim"]),
        "num_layers": int(config["num_layers"]),
        "heads": config.get("heads", "UNKNOWN_FROM_D1"),
        "num_bases": config.get("num_bases", "UNKNOWN_FROM_D1"),
        "use_neighbor_sampling": bool(config.get("use_neighbor_sampling", False)),
        "optimizer": config["optimizer"],
        "lr": float(config["lr"]),
        "weight_decay": float(config["weight_decay"]),
        "dropout": float(config["dropout"]),
        "epochs": int(config["epochs"]),
        "patience": int(config["patience"]),
        "early_stopping_metric": config["early_stopping_metric"],
        "AUC": float(test_metrics["auc"]),
        "AP": float(test_metrics["ap"]),
        "F1": float(test_metrics["f1"]),
        "Recall": float(test_metrics["recall"]),
        "Precision": float(test_metrics["precision"]),
        "best_epoch": int(best_epoch),
        "test_threshold": float(test_threshold),
        "output_dir": str(experiment_dir),
        "notes": config["notes"],
    }
    pd.DataFrame([result_row]).to_csv(metrics_dir / "model_results.csv", index=False)

    save_yaml(experiment_dir / "config.yaml", config)
    save_json(
        experiment_dir / "run_summary.json",
        {
            **config,
            "best_epoch": int(best_epoch),
            "test_threshold": float(test_threshold),
            "metrics": result_row,
            "d1_alignment_notes": bundle.notes,
            "split_stats": split_stats,
            "feature_dim": feature_dim,
            "blocked_label_columns": getattr(bundle, "blocked_label_columns", []),
            "unknown_from_d1": config.get("unknown_from_d1", []),
        },
    )
    logger.info("finished experiment=%s auc=%.6f ap=%.6f f1=%.6f", config["experiment_name"], result_row["AUC"], result_row["AP"], result_row["F1"])


if __name__ == "__main__":
    main()
