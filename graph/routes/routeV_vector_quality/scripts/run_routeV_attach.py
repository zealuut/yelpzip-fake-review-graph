"""RouteV exploratory attach-training from V1a.

This runner is intentionally route-local. It reuses a V1a checkpoint only for
directional exploration, trains a frozen graph surrogate on fixed V1a
Base_LogicAE_CB topology, then lets graph node BCE gradients update the
review encoder's abnormal/vector path.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []

PROJECT_ROOT = Path(__file__).resolve().parents[4]
ROUTE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROUTE_ROOT / "configs" / "routeV_attach_from_v1a.json"

sys.path.insert(0, str(PROJECT_ROOT))

from graph.graph_pipeline import build_self_feature_matrix  # noqa: E402
from graph.llm_utils import numeric_feature_columns  # noqa: E402
from graph.relation_model import (  # noqa: E402
    EDGE_SET_DEFINITIONS,
    _build_edge_pack,
    _fit_graph_backbone_model,
    _predict_graph_backbone_probs,
    _safe_binary_metrics,
    _select_threshold_from_validation,
)
from graph.review_training import build_review_model, build_tokenizer, compute_binary_metrics  # noqa: E402
from graph.routes.routeV_vector_quality.scripts.run_routeV_queue import (  # noqa: E402
    RouteVEncodingArtifacts,
    _align_user_artifacts_to_prepared_labels,
    _build_routev_dataloaders,
    _encode_reviews_routev,
    _json_ready,
    _load_json,
    _save_json,
)
from graph.routes.routeV_vector_quality.src.vector_reg_loss import UserVectorSeparabilityLoss  # noqa: E402


@dataclass
class AttachContext:
    source_dir: Path
    source_user_df: pd.DataFrame
    source_review_df: pd.DataFrame
    llm_feature_df: pd.DataFrame
    abnormal_masks: np.ndarray
    dataloaders: dict[str, DataLoader]
    ordered_reviews: pd.DataFrame
    user_abnormal_vectors: np.ndarray
    user_text_vectors: np.ndarray
    self_features: np.ndarray
    edge_frames: dict[str, pd.DataFrame]
    selected_reviews_by_user: dict[int, list[int]]
    selected_review_ids: set[int]
    review_node_to_position: dict[int, int]


class SelectedUserDataset(Dataset):
    def __init__(self, user_indices: np.ndarray) -> None:
        self.user_indices = torch.tensor(user_indices, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.user_indices.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.user_indices[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RouteV exploratory attach-training from V1a")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--config_path", default=str(DEFAULT_CONFIG))
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--smoke_max_users", type=int, default=256)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved.resolve()


def _safe_copy_file(source: Path, destination: Path) -> None:
    if source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _standardize_stats(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=np.float32)
    mean = matrix.mean(axis=0, keepdims=True).astype(np.float32)
    std = matrix.std(axis=0, keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def _standardize_tensor(features: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (features - mean) / std


def _load_context(config: dict[str, Any], shared: dict[str, Any], *, smoke_test: bool, smoke_max_users: int) -> AttachContext:
    source_dir = _resolve_path(config["source_variant_dir"])
    if not source_dir.exists():
        raise FileNotFoundError(f"Missing source V1a variant directory: {source_dir}")

    source_user_df = pd.read_csv(source_dir / "logic_vectors" / "user_summary.csv")
    source_user_df["user_id"] = source_user_df["user_id"].astype(str)
    source_review_df = pd.read_csv(source_dir / "prepared_data" / "reviews_canonical.csv")
    prepared_user_df = pd.read_csv(source_dir / "prepared_data" / "users_canonical.csv")
    source_review_output = pd.read_csv(source_dir / "review_encoder" / "review_output.csv")
    review_scores = pd.read_csv(source_dir / "review_scores_enriched.csv")
    llm_feature_df = pd.read_csv(source_dir / "llm_mask" / "llm_review_features.csv")
    abnormal_masks = np.load(source_dir / "llm_mask" / "abnormal_token_masks.npy")
    user_abnormal_vectors = np.load(source_dir / "logic_vectors" / "user_abnormal_vectors.npy").astype(np.float32)
    user_text_vectors = np.load(source_dir / "logic_vectors" / "user_text_vectors.npy").astype(np.float32)

    prepared_user_df["user_id"] = prepared_user_df["user_id"].astype(str)
    user_label_map = prepared_user_df.set_index("user_id")["user_label"].astype(int).to_dict()
    split_map = prepared_user_df.set_index("user_id")["split"].astype(str).to_dict()
    source_review_df["user_id"] = source_review_df["user_id"].astype(str)
    source_review_df["user_label"] = source_review_df["user_id"].map(user_label_map).astype(int)
    source_review_df["split"] = source_review_df["user_id"].map(split_map).astype(str)
    source_user_df = _align_user_artifacts_to_prepared_labels(source_user_df, prepared_user_df)

    if smoke_test:
        per_split = max(8, int(smoke_max_users) // 3)
        keep_user_parts = []
        for split_name in ["train", "val", "test"]:
            split_users = source_user_df[source_user_df["split"].astype(str).eq(split_name)].sort_values("user_id").head(per_split)
            keep_user_parts.append(split_users)
        keep_frame = pd.concat(keep_user_parts, ignore_index=True).head(int(smoke_max_users))
        keep_users = set(keep_frame["user_id"].astype(str))
        user_keep_mask = source_user_df["user_id"].isin(keep_users).to_numpy()
        user_positions = np.where(user_keep_mask)[0].astype(np.int64)
        source_user_df = source_user_df.loc[user_keep_mask].reset_index(drop=True)
        user_abnormal_vectors = user_abnormal_vectors[user_positions]
        user_text_vectors = user_text_vectors[user_positions]
        ordered_review_ids = source_review_df.sort_values("review_node_id")["review_node_id"].astype(int).tolist()
        review_position_map = {review_id: pos for pos, review_id in enumerate(ordered_review_ids)}
        kept_ordered_review_ids = sorted(source_review_df.loc[source_review_df["user_id"].isin(keep_users), "review_node_id"].astype(int).tolist())
        mask_positions = np.asarray([review_position_map[review_id] for review_id in kept_ordered_review_ids], dtype=np.int64)
        source_review_df = source_review_df[source_review_df["user_id"].isin(keep_users)].reset_index(drop=True)
        source_review_output = source_review_output[source_review_output["user_id"].astype(str).isin(keep_users)].reset_index(drop=True)
        review_scores = review_scores[review_scores["user_id"].astype(str).isin(keep_users)].reset_index(drop=True)
        keep_review_ids = set(source_review_df["review_node_id"].astype(int))
        llm_feature_df = llm_feature_df[llm_feature_df["review_node_id"].astype(int).isin(keep_review_ids)].reset_index(drop=True)
        abnormal_masks = abnormal_masks[mask_positions]

    tokenizer = build_tokenizer(
        review_encoder=shared.get("review_encoder", "llm_masked_logic"),
        primary_model_name_or_path=shared["primary_model_name_or_path"],
        max_seq_length=int(shared.get("max_seq_length", 256)),
    )
    dataloaders, ordered_reviews = _build_routev_dataloaders(
        review_df=source_review_df,
        llm_feature_df=llm_feature_df,
        abnormal_masks=abnormal_masks,
        tokenizer=tokenizer,
        max_seq_length=int(shared.get("max_seq_length", 256)),
        batch_size=int(shared.get("batch_size", 16)),
    )

    edge_frames: dict[str, pd.DataFrame] = {}
    for relation in sorted({rel for rels in EDGE_SET_DEFINITIONS.values() for rel in rels}):
        path = source_dir / "edges" / f"{relation}_edges.csv"
        edge_frames[relation] = pd.read_csv(path) if path.exists() else pd.DataFrame()
        if smoke_test and not edge_frames[relation].empty:
            keep_users = set(source_user_df["user_id"].astype(str))
            frame = edge_frames[relation].copy()
            frame["src_user_id"] = frame["src_user_id"].astype(str)
            frame["dst_user_id"] = frame["dst_user_id"].astype(str)
            edge_frames[relation] = frame[frame["src_user_id"].isin(keep_users) & frame["dst_user_id"].isin(keep_users)].reset_index(drop=True)

    score_split = "split"
    if score_split not in review_scores.columns:
        score_split = "split_x" if "split_x" in review_scores.columns else "split_y"
    review_scores = review_scores.copy()
    review_scores["user_id"] = review_scores["user_id"].astype(str)
    selected_reviews_by_user: dict[int, list[int]] = {}
    user_id_to_index = {uid: idx for idx, uid in enumerate(source_user_df["user_id"].astype(str).tolist())}
    for user_id, group in review_scores.groupby("user_id"):
        if user_id not in user_id_to_index:
            continue
        ranked = group.sort_values(["evidence_score", "review_node_id"], ascending=[False, True], kind="mergesort")
        selected_reviews_by_user[user_id_to_index[user_id]] = ranked.head(int(shared.get("top_m", 3)))["review_node_id"].astype(int).tolist()
    selected_review_ids = {rid for values in selected_reviews_by_user.values() for rid in values}
    review_node_to_position = {
        int(review_id): pos for pos, review_id in enumerate(ordered_reviews["review_node_id"].astype(int).tolist())
    }

    return AttachContext(
        source_dir=source_dir,
        source_user_df=source_user_df,
        source_review_df=source_review_df,
        llm_feature_df=llm_feature_df,
        abnormal_masks=abnormal_masks,
        dataloaders=dataloaders,
        ordered_reviews=ordered_reviews,
        user_abnormal_vectors=user_abnormal_vectors,
        user_text_vectors=user_text_vectors,
        self_features=build_self_feature_matrix(source_user_df, user_abnormal_vectors),
        edge_frames=edge_frames,
        selected_reviews_by_user=selected_reviews_by_user,
        selected_review_ids=selected_review_ids,
        review_node_to_position=review_node_to_position,
    )


def _build_model(shared: dict[str, Any], source_checkpoint: Path, device: torch.device) -> nn.Module:
    model = build_review_model(
        review_encoder=shared.get("review_encoder", "llm_masked_logic"),
        primary_model_name_or_path=shared["primary_model_name_or_path"],
        numeric_feature_dim=len(numeric_feature_columns()),
        vector_dim=int(shared.get("vector_dim", 256)),
        secondary_model_name_or_path=shared.get("secondary_model_name_or_path"),
        freeze_primary=bool(shared.get("freeze_primary", False)),
        freeze_secondary=bool(shared.get("freeze_secondary", False)),
        abnormal_aux_enabled=False,
        disable_cross_attention=bool(shared.get("disable_cross_attention", False)),
        disable_logic_bilstm=bool(shared.get("disable_logic_bilstm", False)),
        logic_pooling=shared.get("logic_pooling", "attention"),
        gate_mode=shared.get("gate_mode", "learned"),
    )
    state = torch.load(source_checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    return model


def _set_attach_trainable(model: nn.Module, patterns: list[str] | None = None) -> list[str]:
    patterns = patterns or [
        "logic_bilstm",
        "logic_proj",
        "logic_pool",
        "cross_attn",
        "gate_mlp",
        "abnormal_vector_mlp",
        "review_classifier",
        "vector_norm",
    ]
    trainable: list[str] = []
    for _, parameter in model.named_parameters():
        parameter.requires_grad = False
    for name, parameter in model.named_parameters():
        if any(pattern in name for pattern in patterns):
            parameter.requires_grad = True
            trainable.append(name)
    return trainable


def _train_graph_surrogate(
    context: AttachContext,
    shared: dict[str, Any],
    output_dir: Path,
    device: torch.device,
    *,
    smoke_test: bool,
) -> dict[str, Any]:
    ordered_users = context.source_user_df.drop_duplicates(subset=["user_id"]).reset_index(drop=True)
    user_index = {user_id: idx for idx, user_id in enumerate(ordered_users["user_id"].astype(str).tolist())}
    labels = ordered_users["user_label"].to_numpy(dtype=np.int64)
    splits = ordered_users["split"].astype(str).to_numpy()
    relations = EDGE_SET_DEFINITIONS[str(shared.get("edge_set", "Base_LogicAE_CB"))]
    edge_packs = {
        relation: _build_edge_pack(relation, context.edge_frames.get(relation, pd.DataFrame()), user_index)
        for relation in relations
    }
    max_epochs = 6 if smoke_test else int(shared.get("surrogate_max_epochs", 100))
    patience = 2 if smoke_test else int(shared.get("surrogate_patience", 16))
    graph_model, standardized_features, _ = _fit_graph_backbone_model(
        node_features=context.self_features,
        edge_packs=edge_packs,
        labels=labels,
        train_mask=splits == "train",
        val_mask=splits == "val",
        seed=int(shared.get("seed", 42)),
        backbone=shared.get("model_backbone", "current_egat"),
        use_node_gat=bool(shared.get("use_node_gat", False)),
        max_epochs_override=max_epochs,
        patience_override=patience,
    )
    graph_model.to(device)
    for parameter in graph_model.parameters():
        parameter.requires_grad = False
    mean, std = _standardize_stats(context.self_features)
    val_probs = _predict_graph_backbone_probs(
        graph_model,
        standardized_features,
        edge_packs,
        abnormal_biases=None,
        abnormal_value_gates=None,
        use_node_gat=bool(shared.get("use_node_gat", False)),
        mask=splits == "val",
        torch=torch,
        device=device,
    )
    threshold = _select_threshold_from_validation(labels[splits == "val"], val_probs)
    test_probs = _predict_graph_backbone_probs(
        graph_model,
        standardized_features,
        edge_packs,
        abnormal_biases=None,
        abnormal_value_gates=None,
        use_node_gat=bool(shared.get("use_node_gat", False)),
        mask=splits == "test",
        torch=torch,
        device=device,
    )
    metrics = _safe_binary_metrics(labels[splits == "test"], test_probs, threshold=threshold)
    payload = {
        "status": "ok",
        "edge_set": shared.get("edge_set", "Base_LogicAE_CB"),
        "model_name": "frozen_surrogate_current_egat_edge_aware_gat",
        "threshold": threshold,
        "metrics": metrics,
        "max_epochs": max_epochs,
        "patience": patience,
    }
    _save_json(output_dir / "surrogate_graph_metrics.json", payload)
    torch.save(graph_model.state_dict(), output_dir / "frozen_graph_surrogate.pt")
    return {
        "model": graph_model,
        "edge_packs": edge_packs,
        "labels": labels,
        "splits": splits,
        "mean": mean,
        "std": std,
        "metrics": metrics,
        "threshold": threshold,
    }


def _selected_review_batch(
    context: AttachContext,
    user_indices: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    review_positions: list[int] = []
    repeated_users: list[int] = []
    for user_idx in user_indices.detach().cpu().numpy().astype(int).tolist():
        for review_id in context.selected_reviews_by_user.get(user_idx, []):
            if review_id in context.review_node_to_position:
                review_positions.append(context.review_node_to_position[review_id])
                repeated_users.append(user_idx)
    return np.asarray(review_positions, dtype=np.int64), np.asarray(repeated_users, dtype=np.int64)


def _forward_selected_reviews(
    model: nn.Module,
    all_loader: DataLoader,
    selected_positions: np.ndarray,
    device: torch.device,
) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor]:
    dataset = all_loader.dataset
    tensors = [dataset[int(pos)] for pos in selected_positions.tolist()]
    input_ids = torch.stack([item["input_ids"] for item in tensors]).to(device)
    attention_mask = torch.stack([item["attention_mask"] for item in tensors]).to(device)
    abnormal_mask = torch.stack([item["abnormal_mask"] for item in tensors]).to(device)
    numeric_features = torch.stack([item["numeric_features"] for item in tensors]).to(device)
    labels = torch.stack([item["label"] for item in tensors]).to(device)
    user_labels = torch.stack([item["user_label"] for item in tensors]).to(device)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        abnormal_token_mask=abnormal_mask,
        numeric_features=numeric_features,
    )
    return outputs, labels, user_labels, torch.stack([item["user_id_idx"] for item in tensors]).to(device)


def _aggregate_user_vectors(
    review_vectors: torch.Tensor,
    repeated_users: np.ndarray,
    batch_user_indices: torch.Tensor,
) -> torch.Tensor:
    local_map = {int(user_idx): pos for pos, user_idx in enumerate(batch_user_indices.detach().cpu().numpy().astype(int).tolist())}
    local_indices = torch.tensor([local_map[int(user_idx)] for user_idx in repeated_users.tolist()], dtype=torch.long, device=review_vectors.device)
    out = torch.zeros((len(local_map), review_vectors.shape[1]), dtype=review_vectors.dtype, device=review_vectors.device)
    counts = torch.zeros((len(local_map), 1), dtype=review_vectors.dtype, device=review_vectors.device)
    out.index_add_(0, local_indices, review_vectors)
    counts.index_add_(0, local_indices, torch.ones((len(repeated_users), 1), dtype=review_vectors.dtype, device=review_vectors.device))
    return out / counts.clamp_min(1.0)


def _attach_train_variant(
    model: nn.Module,
    context: AttachContext,
    graph_state: dict[str, Any],
    shared: dict[str, Any],
    variant: dict[str, Any],
    exp_dir: Path,
    device: torch.device,
    *,
    smoke_test: bool,
) -> list[dict[str, Any]]:
    trainable_names = _set_attach_trainable(model, variant.get("trainable_patterns"))
    _save_json(exp_dir / "trainable_parameters.json", trainable_names)
    model.train()

    graph_model = graph_state["model"]
    graph_model.eval()
    labels_np = graph_state["labels"]
    splits = graph_state["splits"]
    train_user_indices = np.where(splits == "train")[0].astype(np.int64)
    if smoke_test:
        train_user_indices = train_user_indices[: min(len(train_user_indices), int(shared.get("smoke_max_train_users", 96)))]
    user_loader = DataLoader(
        SelectedUserDataset(train_user_indices),
        batch_size=int(variant.get("attach_user_batch_size", shared.get("attach_user_batch_size", 64))),
        shuffle=True,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(variant.get("attach_learning_rate", shared.get("attach_learning_rate", 1e-5))),
        weight_decay=float(variant.get("attach_weight_decay", 1e-4)),
    )
    labels_tensor = torch.as_tensor(labels_np, dtype=torch.float32, device=device)
    y_train = labels_np[splits == "train"].astype(np.float32)
    pos_count = float(y_train.sum())
    neg_count = float(len(y_train) - pos_count)
    graph_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg_count / max(pos_count, 1.0)], dtype=torch.float32, device=device))
    review_criterion = nn.BCEWithLogitsLoss()
    supcon = UserVectorSeparabilityLoss(temperature=float(variant.get("supcon_temperature", shared.get("supcon_temperature", 0.1))))

    base_features = torch.as_tensor(context.self_features, dtype=torch.float32, device=device)
    base_abnormal = torch.as_tensor(context.user_abnormal_vectors, dtype=torch.float32, device=device)
    mean = torch.as_tensor(graph_state["mean"], dtype=torch.float32, device=device)
    std = torch.as_tensor(graph_state["std"], dtype=torch.float32, device=device)
    abnormal_start = base_features.shape[1] - base_abnormal.shape[1]
    beta = float(variant.get("graph_beta", 0.0))
    review_weight = float(variant.get("review_weight", shared.get("review_weight", 0.1)))
    supcon_weight = float(variant.get("supcon_weight", shared.get("supcon_weight", 0.1)))
    delta_l2_weight = float(variant.get("delta_l2_weight", shared.get("delta_l2_weight", 0.01)))
    attach_epochs = 1 if smoke_test and int(variant.get("attach_epochs", shared.get("attach_epochs", 2))) > 0 else int(variant.get("attach_epochs", shared.get("attach_epochs", 2)))
    history: list[dict[str, Any]] = []

    if attach_epochs <= 0 or (beta == 0.0 and review_weight == 0.0 and supcon_weight == 0.0 and delta_l2_weight == 0.0):
        return history

    for epoch in range(attach_epochs):
        epoch_losses: list[float] = []
        epoch_graph: list[float] = []
        epoch_review: list[float] = []
        epoch_supcon: list[float] = []
        for batch_user_indices in tqdm(user_loader, desc=f"Attach {variant['name']} epoch {epoch + 1}", leave=False):
            batch_user_indices = batch_user_indices.to(device)
            selected_positions, repeated_users = _selected_review_batch(context, batch_user_indices, device)
            if len(selected_positions) == 0:
                continue
            outputs, review_labels, user_labels_by_review, _ = _forward_selected_reviews(
                model=model,
                all_loader=context.dataloaders["all"],
                selected_positions=selected_positions,
                device=device,
            )
            user_vectors = _aggregate_user_vectors(outputs.review_vector, repeated_users, batch_user_indices)

            node_features = base_features.clone()
            node_features[batch_user_indices, abnormal_start:] = user_vectors
            graph_input = _standardize_tensor(node_features, mean, std)
            logits = graph_model(graph_input, graph_state["edge_packs"], torch=torch)
            graph_loss = graph_criterion(logits[batch_user_indices], labels_tensor[batch_user_indices])
            review_loss = review_criterion(outputs.review_logit, review_labels)
            batch_user_labels = labels_tensor[batch_user_indices]
            supcon_loss = supcon(user_vectors, torch.arange(len(batch_user_indices), device=device), batch_user_labels)
            delta_l2 = (user_vectors - base_abnormal[batch_user_indices]).pow(2).mean()
            loss = beta * graph_loss + review_weight * review_loss + supcon_weight * supcon_loss + delta_l2_weight * delta_l2

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
            epoch_graph.append(float(graph_loss.detach().cpu()))
            epoch_review.append(float(review_loss.detach().cpu()))
            epoch_supcon.append(float(supcon_loss.detach().cpu()))

        row = {
            "epoch": epoch + 1,
            "loss": float(np.mean(epoch_losses) if epoch_losses else 0.0),
            "graph_loss": float(np.mean(epoch_graph) if epoch_graph else 0.0),
            "review_loss": float(np.mean(epoch_review) if epoch_review else 0.0),
            "supcon_loss": float(np.mean(epoch_supcon) if epoch_supcon else 0.0),
            "graph_beta": beta,
            "review_weight": review_weight,
            "supcon_weight": supcon_weight,
            "delta_l2_weight": delta_l2_weight,
        }
        history.append(row)
        print(
            f"    attach epoch {epoch + 1}: loss={row['loss']:.6f} graph={row['graph_loss']:.6f} "
            f"review={row['review_loss']:.6f} supcon={row['supcon_loss']:.6f}",
            flush=True,
        )

    return history


def _encode_and_evaluate(
    model: nn.Module,
    context: AttachContext,
    graph_state: dict[str, Any],
    shared: dict[str, Any],
    variant: dict[str, Any],
    exp_dir: Path,
    device: torch.device,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    review_encoder_dir = exp_dir / "review_encoder"
    review_encoder_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = review_encoder_dir / "best_review_encoder.pt"
    torch.save(model.state_dict(), checkpoint_path)
    metrics_path = review_encoder_dir / "review_encoder_metrics.json"
    _save_json(metrics_path, {"checkpoint_path": str(checkpoint_path), "attach_history": history})
    artifacts: RouteVEncodingArtifacts = _encode_reviews_routev(
        model=model,
        dataloader=context.dataloaders["all"],
        review_df=context.source_review_df,
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
        device=device,
        use_graph_vector=False,
    )
    artifacts.review_output_df.to_csv(review_encoder_dir / "review_output.csv", index=False)
    np.save(review_encoder_dir / "selected_review_vectors.npy", artifacts.review_vectors)
    np.save(review_encoder_dir / "selected_text_vectors.npy", artifacts.text_vectors)

    review_vector_by_node = {
        int(review_id): artifacts.review_vectors[pos]
        for pos, review_id in enumerate(artifacts.review_output_df["review_node_id"].astype(int).tolist())
    }
    user_vectors: list[np.ndarray] = []
    for user_idx in range(len(context.source_user_df)):
        selected = context.selected_reviews_by_user.get(user_idx, [])
        vectors = [review_vector_by_node[review_id] for review_id in selected if review_id in review_vector_by_node]
        if vectors:
            user_vectors.append(np.asarray(vectors, dtype=np.float32).mean(axis=0))
        else:
            user_vectors.append(context.user_abnormal_vectors[user_idx])
    user_abnormal_vectors = np.asarray(user_vectors, dtype=np.float32)
    user_df = context.source_user_df.copy()
    self_features = build_self_feature_matrix(user_df, user_abnormal_vectors)
    logic_dir = exp_dir / "logic_vectors"
    logic_dir.mkdir(parents=True, exist_ok=True)
    np.save(logic_dir / "review_abnormal_vectors.npy", artifacts.review_vectors)
    np.save(logic_dir / "review_text_vectors.npy", artifacts.text_vectors)
    np.save(logic_dir / "user_abnormal_vectors_initial.npy", context.user_abnormal_vectors)
    np.save(logic_dir / "user_abnormal_vectors.npy", user_abnormal_vectors)
    np.save(logic_dir / "user_text_vectors.npy", context.user_text_vectors)
    user_df.to_csv(logic_dir / "user_summary.csv", index=False)
    user_df.to_csv(exp_dir / "user_scores_enriched.csv", index=False)

    labels = graph_state["labels"]
    splits = graph_state["splits"]
    mean = graph_state["mean"]
    std = graph_state["std"]
    standardized_features = ((self_features.astype(np.float32) - mean) / std).astype(np.float32)
    val_probs = _predict_graph_backbone_probs(
        graph_state["model"],
        standardized_features,
        graph_state["edge_packs"],
        abnormal_biases=None,
        abnormal_value_gates=None,
        use_node_gat=bool(shared.get("use_node_gat", False)),
        mask=splits == "val",
        torch=torch,
        device=device,
    )
    threshold = _select_threshold_from_validation(labels[splits == "val"], val_probs)
    test_probs = _predict_graph_backbone_probs(
        graph_state["model"],
        standardized_features,
        graph_state["edge_packs"],
        abnormal_biases=None,
        abnormal_value_gates=None,
        use_node_gat=bool(shared.get("use_node_gat", False)),
        mask=splits == "test",
        torch=torch,
        device=device,
    )
    val_metrics = _safe_binary_metrics(labels[splits == "val"], val_probs, threshold)
    test_metrics = _safe_binary_metrics(labels[splits == "test"], test_probs, threshold)
    metrics_dir = exp_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    model_results = pd.DataFrame(
        [
            {
                "review_encoder": shared.get("review_encoder", "llm_masked_logic"),
                "model_name": "frozen_surrogate_current_egat_edge_aware_gat",
                "edge_set": shared.get("edge_set", "Base_LogicAE_CB"),
                "threshold": threshold,
                "val_auc": val_metrics["auc"],
                "val_ap": val_metrics["ap"],
                "auc": test_metrics["auc"],
                "ap": test_metrics["ap"],
                "recall": test_metrics["recall"],
                "precision": test_metrics["precision"],
                "f1": test_metrics["f1"],
                "accuracy": test_metrics["accuracy"],
                "num_train_users": int((splits == "train").sum()),
                "num_val_users": int((splits == "val").sum()),
                "num_test_users": int((splits == "test").sum()),
                "num_fake_train": int(labels[splits == "train"].sum()),
                "num_fake_val": int(labels[splits == "val"].sum()),
                "num_fake_test": int(labels[splits == "test"].sum()),
                "backbone": shared.get("model_backbone", "current_egat"),
                "relation_model": shared.get("relation_model", "edge_aware_gat"),
                "fixed_topology": True,
                "fixed_top_m_selection": True,
            }
        ]
    )
    model_results.to_csv(metrics_dir / "model_results.csv", index=False)
    return model_results.iloc[0].to_dict()


def _artifact_reuse_manifest(config: dict[str, Any], context: AttachContext) -> dict[str, Any]:
    return {
        "experiment_type": "exploratory_only_checkpoint_reuse",
        "source_variant_dir": str(context.source_dir),
        "policy": config.get("paper_facing_policy"),
        "items": [
            {
                "path": "source review_encoder/best_review_encoder.pt",
                "class": "review_checkpoint",
                "reuse_mode": "artifact_anchored_only",
                "reason": "Exploratory direction check starts from V1a. Formal attach variants must fresh-train their own checkpoint.",
            },
            {
                "path": "source edges/Base_LogicAE_CB",
                "class": "fixed_graph_topology",
                "reuse_mode": "artifact_anchored_only",
                "reason": "Frozen-graph attach isolates whether graph node BCE can improve the abnormal-vector path without topology changes.",
            },
            {
                "path": "source top-m review selection",
                "class": "fixed_user_vector_pooling",
                "reuse_mode": "artifact_anchored_only",
                "reason": "Top-m selection is fixed from V1a evidence scores for the first attach diagnostic.",
            },
            {
                "path": "logic_vectors/user_abnormal_vectors.npy",
                "class": "learned_user_abnormal_vectors",
                "reuse_mode": "regenerated",
                "reason": "Recomputed from the attach-updated review encoder after training.",
            },
        ],
    }


def _run_variant(
    output_root: Path,
    config: dict[str, Any],
    context: AttachContext,
    graph_state: dict[str, Any],
    variant: dict[str, Any],
    *,
    smoke_test: bool,
) -> dict[str, Any]:
    shared = dict(config["base_protocol_args"])
    if smoke_test:
        shared["batch_size"] = min(int(shared.get("batch_size", 16)), 8)
    device = resolve_device(shared.get("device", "auto"))
    name = variant["name"]
    exp_dir = output_root / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    source_checkpoint = context.source_dir / "review_encoder" / "best_review_encoder.pt"
    if not source_checkpoint.exists():
        source_summary = _load_json(context.source_dir / "run_summary.json")
        source_checkpoint = Path(source_summary["selected_checkpoint"])
    model = _build_model(shared, source_checkpoint, device)
    print(f"    loaded source checkpoint: {source_checkpoint}", flush=True)
    history = _attach_train_variant(
        model=model,
        context=context,
        graph_state=graph_state,
        shared=shared,
        variant=variant,
        exp_dir=exp_dir,
        device=device,
        smoke_test=smoke_test,
    )
    best_graph_model = _encode_and_evaluate(
        model=model,
        context=context,
        graph_state=graph_state,
        shared=shared,
        variant=variant,
        exp_dir=exp_dir,
        device=device,
        history=history,
    )
    run_config = {
        **shared,
        "variant": variant,
        "experiment_type": "exploratory_only_checkpoint_reuse",
        "source_variant_dir": str(context.source_dir),
        "paper_facing_policy": config.get("paper_facing_policy"),
        "surrogate_graph": config.get("surrogate_graph"),
        "artifact_reuse": _artifact_reuse_manifest(config, context),
        "fixed_topology": True,
        "fixed_top_m_selection": True,
        "smoke_test": bool(smoke_test),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    summary = {
        "status": "ok",
        "variant": name,
        "output_dir": str(exp_dir),
        "experiment_type": "exploratory_only_checkpoint_reuse",
        "checkpoint_reuse": "exploratory_only",
        "paper_facing_policy": config.get("paper_facing_policy"),
        "source_variant_dir": str(context.source_dir),
        "attach_history": history,
        "surrogate_graph_metrics": graph_state["metrics"],
        "best_graph_model": best_graph_model,
        "artifact_reuse": run_config["artifact_reuse"],
    }
    _save_json(exp_dir / "run_config.json", run_config)
    _save_json(exp_dir / "routeV_attach_variant_config.json", {"variant": variant, "shared": shared})
    _save_json(exp_dir / "run_summary.json", summary)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def _graph_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    row = dict(summary.get("best_graph_model") or {})
    return {
        "graph_auc": float(row.get("auc", 0.0)),
        "graph_ap": float(row.get("ap", 0.0)),
        "graph_f1": float(row.get("f1", 0.0)),
        "graph_recall": float(row.get("recall", 0.0)),
        "graph_precision": float(row.get("precision", 0.0)),
        "edge_set": str(row.get("edge_set", "")),
        "model_name": str(row.get("model_name", "")),
    }


def main() -> None:
    args = parse_args()
    config = _load_json(Path(args.config_path))
    shared = dict(config["base_protocol_args"])
    if args.smoke_test:
        shared["batch_size"] = min(int(shared.get("batch_size", 16)), 8)
        shared["surrogate_max_epochs"] = 6
        shared["surrogate_patience"] = 2
    seed_everything(int(shared.get("seed", 42)))
    device = resolve_device(shared.get("device", "auto"))
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    status_file = PROJECT_ROOT / "graph" / "logs" / "status" / f"routeV_attach_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    status_file.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 72, flush=True)
    print("RouteV Attach From V1a", flush=True)
    print(f"Output: {output_root}", flush=True)
    print(f"Config: {args.config_path}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Smoke: {args.smoke_test}", flush=True)
    print("=" * 72, flush=True)
    _save_json(
        status_file,
        {
            "status": "starting",
            "output_root": str(output_root),
            "config_path": str(args.config_path),
            "experiment_type": "exploratory_only_checkpoint_reuse",
            "started_at": datetime.now().isoformat(timespec="seconds"),
        },
    )

    context = _load_context(config, shared, smoke_test=bool(args.smoke_test), smoke_max_users=int(args.smoke_max_users))
    surrogate_dir = output_root / "_frozen_graph_surrogate"
    surrogate_dir.mkdir(parents=True, exist_ok=True)
    _save_json(status_file, {"status": "training_frozen_surrogate", "output_root": str(output_root)})
    graph_state = _train_graph_surrogate(context, shared, surrogate_dir, device, smoke_test=bool(args.smoke_test))
    print(
        f"Frozen surrogate test AUC={graph_state['metrics']['auc']:.6f} "
        f"AP={graph_state['metrics']['ap']:.6f} F1={graph_state['metrics']['f1']:.6f}",
        flush=True,
    )

    available = config["variants"]
    requested = set(args.variants or [variant["name"] for variant in available])
    selected = [variant for variant in available if variant["name"] in requested]
    if not selected:
        raise ValueError(f"No requested RouteV attach variants matched: {sorted(requested)}")

    results: list[dict[str, Any]] = []
    for variant in selected:
        name = variant["name"]
        exp_dir = output_root / name
        print(f"\n[RouteV Attach] Running {name}", flush=True)
        _save_json(
            status_file,
            {
                "status": "running_variant",
                "active_variant": name,
                "output_root": str(output_root),
                "results": results,
            },
        )
        if args.resume and (exp_dir / "run_summary.json").exists():
            summary = _load_json(exp_dir / "run_summary.json")
        else:
            try:
                summary = _run_variant(
                    output_root=output_root,
                    config=config,
                    context=context,
                    graph_state=graph_state,
                    variant=variant,
                    smoke_test=bool(args.smoke_test),
                )
            except Exception as exc:
                failure = {"name": name, "status": "failed", "error": repr(exc)}
                results.append(failure)
                _save_json(output_root / "routeV_attach_summary.json", {"status": "failed", "results": results})
                _save_json(status_file, {"status": "failed", "failed_variant": name, "results": results})
                raise
        result = {"name": name, "status": summary.get("status", "ok"), **_graph_metrics(summary)}
        results.append(result)
        _save_json(
            status_file,
            {
                "status": "running",
                "last_completed_variant": name,
                "output_root": str(output_root),
                "results": results,
            },
        )
        print(
            f"    {name}: AUC={result['graph_auc']:.6f} AP={result['graph_ap']:.6f} "
            f"F1={result['graph_f1']:.6f}",
            flush=True,
        )

    final_summary = {
        "status": "complete",
        "output_root": str(output_root),
        "config_path": str(args.config_path),
        "experiment_type": "exploratory_only_checkpoint_reuse",
        "paper_facing_policy": config.get("paper_facing_policy"),
        "source_variant_dir": str(context.source_dir),
        "surrogate_graph_metrics": graph_state["metrics"],
        "acceptance_reference": config.get("acceptance_reference"),
        "results": results,
    }
    _save_json(output_root / "routeV_attach_summary.json", final_summary)
    _save_json(status_file, final_summary)
    print("\nRouteV attach summary:", flush=True)
    for result in results:
        print(
            f"  {result['name']}: {result['status']} "
            f"AUC={result.get('graph_auc', 'NA')} AP={result.get('graph_ap', 'NA')} F1={result.get('graph_f1', 'NA')}",
            flush=True,
        )
    print(f"Summary: {output_root / 'routeV_attach_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
