from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from graph.graph_pipeline import build_self_feature_matrix, compute_edge_stats
from graph.relation_model import (
    EDGE_SET_DEFINITIONS,
    _build_edge_pack,
    _fit_feature_only_baseline,
    _import_torch_modules,
    _safe_binary_metrics,
    _select_threshold_from_validation,
    _standardize_feature_matrix,
)
from graph.routes.routeG_egatpp.src.egatpp_model import build_egatpp_components
from graph.scripts.route_runner import _load_base_artifacts

MAIN_PROJECT_ROOT = Path("/home/xyz/HuChao (2)/Bert-TextClassification")
BASE_PROTOCOL_DIR = MAIN_PROJECT_ROOT / "graph" / "outputs" / "yelpzip_balanced_current_graph_no_reweight_20260502_160620"
REFERENCE_D1_DIR = MAIN_PROJECT_ROOT / "graph" / "outputs" / "routeD_tns_guided_logic_egat_20260504_200855" / "D1_EGAT_Base_LogicAE_CB"
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "routeG_egatpp.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--config_path", default=str(DEFAULT_CONFIG))
    parser.add_argument("--smoke_mode", action="store_true")
    return parser.parse_args()


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_assets():
    artifacts = _load_base_artifacts(BASE_PROTOCOL_DIR)
    d1_summary = json.loads((REFERENCE_D1_DIR / "run_summary.json").read_text(encoding="utf-8"))
    user_df = artifacts["user_df"].copy()
    self_features = build_self_feature_matrix(user_df, artifacts["user_abnormal_vectors"])
    edge_frames = {
        name: pd.read_csv(REFERENCE_D1_DIR / "edges" / f"{name}_edges.csv")
        for name in EDGE_SET_DEFINITIONS["Base_LogicAE_CB"]
    }
    return {
        "user_df": user_df,
        "self_features": self_features,
        "edge_frames": edge_frames,
        "d1_best": d1_summary["best_graph_model"],
    }


def _train_egatpp_variant(
    *,
    user_df: pd.DataFrame,
    self_features: np.ndarray,
    edge_frames: dict[str, pd.DataFrame],
    output_dir: Path,
    seed: int,
    hidden_dim: int,
    num_layers: int,
    heads: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    gatv2: bool,
    edge_gate: bool,
) -> dict[str, Any]:
    torch, _, _ = _import_torch_modules()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ordered_users = user_df.drop_duplicates(subset=["user_id"]).reset_index(drop=True)
    user_ids = ordered_users["user_id"].astype(str).tolist()
    user_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
    labels = ordered_users["user_label"].to_numpy(dtype=np.int64)
    splits = ordered_users["split"].astype(str).to_numpy()
    train_mask = splits == "train"
    val_mask = splits == "val"
    test_mask = splits == "test"

    feature_input = _standardize_feature_matrix(self_features.astype(np.float32))
    edge_packs = {
        relation: _build_edge_pack(relation, edge_frames.get(relation, pd.DataFrame()), user_index, relation_topk=None)
        for relation in EDGE_SET_DEFINITIONS["Base_LogicAE_CB"]
    }
    edge_dim_map = {relation: max(1, pack.edge_features.shape[1]) for relation, pack in edge_packs.items()}
    EGATPPClassifier = build_egatpp_components(torch)
    model = EGATPPClassifier(
        input_dim=feature_input.shape[1],
        relations=sorted(edge_packs.keys()),
        edge_dim_map=edge_dim_map,
        hidden_dim=hidden_dim,
        heads=heads,
        dropout=dropout,
        num_layers=num_layers,
        gatv2=gatv2,
        edge_gate=edge_gate,
    ).to(device)

    feature_tensor = torch.as_tensor(feature_input, dtype=torch.float32, device=device)
    y_train = labels[train_mask].astype(np.float32)
    pos_count = float(y_train.sum())
    neg_count = float(len(y_train) - pos_count)
    pos_weight = torch.tensor([neg_count / max(pos_count, 1.0)], dtype=torch.float32, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val_auc = -1.0
    best_epoch = -1
    best_val_probs = None
    bad_epochs = 0
    epoch_rows = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(feature_tensor, edge_packs)
        loss = criterion(logits[train_mask], torch.as_tensor(labels[train_mask], dtype=torch.float32, device=device))
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(feature_tensor, edge_packs)).detach().cpu().numpy()
        threshold = _select_threshold_from_validation(labels[val_mask], probs[val_mask])
        val_metrics = _safe_binary_metrics(labels[val_mask], probs[val_mask], threshold)
        epoch_rows.append(
            {
                "epoch": epoch,
                "loss": float(loss.detach().cpu().item()),
                "val_auc": val_metrics["auc"],
                "val_ap": val_metrics["ap"],
                "val_f1": val_metrics["f1"],
                "val_recall": val_metrics["recall"],
                "val_precision": val_metrics["precision"],
                "threshold": threshold,
            }
        )
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_epoch = epoch
            best_val_probs = probs.copy()
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("EGAT++ training failed to produce a best state.")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_probs = torch.sigmoid(model(feature_tensor, edge_packs)).detach().cpu().numpy()
    threshold = _select_threshold_from_validation(labels[val_mask], best_val_probs[val_mask])
    test_metrics = _safe_binary_metrics(labels[test_mask], test_probs[test_mask], threshold)
    val_metrics = _safe_binary_metrics(labels[val_mask], best_val_probs[val_mask], threshold)

    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(epoch_rows).to_csv(metrics_dir / "epoch_metrics.csv", index=False)
    pd.DataFrame(
        {
            "user_id": ordered_users.loc[test_mask, "user_id"].astype(str).tolist(),
            "label": labels[test_mask].tolist(),
            "prob": test_probs[test_mask].tolist(),
            "pred": (test_probs[test_mask] >= threshold).astype(int).tolist(),
            "split": ["test"] * int(test_mask.sum()),
        }
    ).to_csv(metrics_dir / "test_predictions.csv", index=False)
    return {
        "threshold": float(threshold),
        "val_auc": float(val_metrics["auc"]),
        "val_ap": float(val_metrics["ap"]),
        "auc": float(test_metrics["auc"]),
        "ap": float(test_metrics["ap"]),
        "recall": float(test_metrics["recall"]),
        "precision": float(test_metrics["precision"]),
        "f1": float(test_metrics["f1"]),
        "accuracy": float(test_metrics["accuracy"]),
        "best_epoch": int(best_epoch),
    }


def _run_single(
    *,
    exp_name: str,
    output_dir: Path,
    assets: dict,
    cfg: dict,
    variant: str,
    smoke_mode: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    compute_edge_stats(edge_frames=assets["edge_frames"], user_df=assets["user_df"], output_dir=output_dir)
    if variant == "g0":
        from graph.relation_model import run_relation_aggregation_experiments

        df = run_relation_aggregation_experiments(
            user_df=assets["user_df"],
            self_features=assets["self_features"],
            edge_frames=assets["edge_frames"],
            output_dir=output_dir / "metrics",
            review_encoder_name="llm_masked_logic",
            model_kind="edge_aware_gat",
            seed=int(cfg["seed"]),
            backbone="current_egat",
            relation_model="edge_aware_gat",
            use_abnormal_edge_weight=False,
            use_abnormal_gate=False,
            use_abnormal_value_gate=False,
            use_abnormal_attention_bias=False,
            abnormal_score_source="auto",
            abnormal_edge_lambda=1.0,
            abnormal_edge_eta=0.5,
            abnormal_gate_eta=0.5,
            abnormal_pair_mode="both_high",
            abnormal_gate_learnable=False,
            abnormal_attention_gamma=1.0,
            selected_edge_set="Base_LogicAE_CB",
            relation_topk=None,
            use_node_gat=False,
            max_epochs_override=1 if smoke_mode else None,
            patience_override=1 if smoke_mode else None,
        )
        row = df.loc[df["edge_set"] == "Base_LogicAE_CB"].iloc[0].to_dict()
        row["best_epoch"] = row.get("best_epoch", None)
    else:
        row = _train_egatpp_variant(
            user_df=assets["user_df"],
            self_features=assets["self_features"],
            edge_frames=assets["edge_frames"],
            output_dir=output_dir,
            seed=int(cfg["seed"]),
            hidden_dim=int(cfg["hidden_dim"]),
            num_layers=int(cfg["num_layers"]),
            heads=int(cfg["heads"]),
            dropout=float(cfg["dropout"]),
            lr=float(cfg["lr"]),
            weight_decay=float(cfg["weight_decay"]),
            max_epochs=1 if smoke_mode else int(cfg["max_epochs"]),
            patience=1 if smoke_mode else int(cfg["patience"]),
            gatv2=(variant == "g2"),
            edge_gate=(variant == "g2"),
        )
        metrics_dir = output_dir / "metrics"
        pd.DataFrame(
            [
                {
                    "review_encoder": "llm_masked_logic",
                    "model_name": "egatpp_full" if variant == "g2" else "egatpp_lite",
                    "edge_set": "Base_LogicAE_CB",
                    "threshold": row["threshold"],
                    "val_auc": row["val_auc"],
                    "val_ap": row["val_ap"],
                    "auc": row["auc"],
                    "ap": row["ap"],
                    "recall": row["recall"],
                    "precision": row["precision"],
                    "f1": row["f1"],
                    "accuracy": row["accuracy"],
                    "num_train_users": int((assets["user_df"]["split"] == "train").sum()),
                    "num_val_users": int((assets["user_df"]["split"] == "val").sum()),
                    "num_test_users": int((assets["user_df"]["split"] == "test").sum()),
                    "num_fake_train": int(((assets["user_df"]["split"] == "train") & (assets["user_df"]["user_label"] == 1)).sum()),
                    "num_fake_val": int(((assets["user_df"]["split"] == "val") & (assets["user_df"]["user_label"] == 1)).sum()),
                    "num_fake_test": int(((assets["user_df"]["split"] == "test") & (assets["user_df"]["user_label"] == 1)).sum()),
                    "backbone": "egatpp_full" if variant == "g2" else "egatpp_lite",
                    "relation_model": "edge_aware_gat",
                }
            ]
        ).to_csv(metrics_dir / "model_results.csv", index=False)
    _save_json(
        output_dir / "run_summary.json",
        {
            "experiment_name": exp_name,
            "variant": variant,
            "best_graph_model": row,
            "base_source": "D1_EGAT_Base_LogicAE_CB",
            "implementation": "routeG_egatpp",
        },
    )
    _save_json(output_dir / "config.json", cfg | {"experiment_name": exp_name, "variant": variant})
    return row


def main() -> None:
    args = parse_args()
    cfg = json.loads(Path(args.config_path).read_text(encoding="utf-8"))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    assets = _load_assets()

    rows = [
        {
            "experiment_name": "D1_EGAT_Base_LogicAE_CB",
            "variant": "reference_only",
            "AUC": assets["d1_best"]["auc"],
            "AP": assets["d1_best"]["ap"],
            "F1": assets["d1_best"]["f1"],
            "Recall": assets["d1_best"]["recall"],
            "Precision": assets["d1_best"]["precision"],
            "best_epoch": assets["d1_best"].get("best_epoch"),
            "threshold": assets["d1_best"]["threshold"],
            "notes": "reference row",
        }
    ]
    for exp_name, variant in [
        ("G0_D1_EGAT", "g0"),
        ("G1_EGATPP_Lite", "g1"),
        ("G2_EGATPP_Full", "g2"),
    ]:
        row = _run_single(
            exp_name=exp_name,
            output_dir=output_root / exp_name,
            assets=assets,
            cfg=cfg,
            variant=variant,
            smoke_mode=args.smoke_mode,
        )
        rows.append(
            {
                "experiment_name": exp_name,
                "variant": variant,
                "AUC": row["auc"],
                "AP": row["ap"],
                "F1": row["f1"],
                "Recall": row["recall"],
                "Precision": row["precision"],
                "best_epoch": row.get("best_epoch"),
                "threshold": row["threshold"],
                "notes": "smoke" if args.smoke_mode else "",
            }
        )
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_root / "routeG_summary.csv", index=False)
    (output_root / "routeG_summary.md").write_text(summary_df.to_csv(index=False), encoding="utf-8")


if __name__ == "__main__":
    main()
