from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .baseline_models import GATCurrentTopK, GraphBatch, GraphSAGECurrentTopK, RGCNCurrentTopK
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
    dropout = float(config["dropout"])
    if model_name == "gat":
        return GATCurrentTopK(input_dim=input_dim, hidden_dim=hidden_dim, heads=int(config["heads"]), dropout=dropout)
    if model_name == "graphsage":
        return GraphSAGECurrentTopK(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
    if model_name == "rgcn":
        return RGCNCurrentTopK(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_relations=num_relations,
            num_bases=int(config["num_bases"]),
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

    bundle = load_protocol_bundle()
    write_edge_stats(bundle, experiment_dir)

    seed = int(config["seed"])
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch, num_edges = _build_batch(bundle, config["relation_handling"], device)
    model = _build_model(config, input_dim=bundle.node_features.shape[1], num_relations=len(EDGE_TYPES)).to(device)

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
            "pred": (test_probs >= test_threshold).astype(int),
            "split": "test",
        }
    )
    test_pred_frame.to_csv(metrics_dir / "test_predictions.csv", index=False)
    epoch_frame(epoch_rows).to_csv(metrics_dir / "epoch_metrics.csv", index=False)

    result_row = {
        "experiment_name": config["experiment_name"],
        "model": config["model"],
        "graph_protocol": config["graph_protocol"],
        "edge_set": config["edge_set"],
        "relation_handling": config["relation_handling"],
        "feature_source": config["feature_source"],
        "num_users": len(bundle.user_ids),
        "num_edges": int(num_edges),
        "hidden_dim": int(config["hidden_dim"]),
        "num_layers": int(config["num_layers"]),
        "heads": config.get("heads", "UNKNOWN_FROM_D1"),
        "num_bases": config.get("num_bases", "UNKNOWN_FROM_D1"),
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
            "unknown_from_d1": config.get("unknown_from_d1", []),
        },
    )
    logger.info("finished experiment=%s auc=%.6f ap=%.6f f1=%.6f", config["experiment_name"], result_row["AUC"], result_row["AP"], result_row["F1"])


if __name__ == "__main__":
    main()
