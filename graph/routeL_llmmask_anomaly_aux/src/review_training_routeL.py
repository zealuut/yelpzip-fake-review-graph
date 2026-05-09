from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from graph.llm_utils import numeric_feature_columns
from graph.review_training import build_review_dataloaders, build_tokenizer
from graph.review_training import compute_binary_metrics

from .review_models_routeL import RouteLLMMaskedLogicEncoder, RouteLReviewEncoderOutput


def load_routeL_review_frames(base_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    base_dir = Path(base_dir)
    review_df = pd.read_csv(base_dir / "prepared_data/reviews_canonical.csv")
    llm_feature_df = pd.read_csv(base_dir / "llm_mask/llm_review_features.csv")
    abnormal_masks = np.load(base_dir / "llm_mask/abnormal_token_masks.npy")
    return review_df, llm_feature_df, abnormal_masks


def build_routeL_dataloaders(
    base_dir: str | Path,
    primary_model_name_or_path: str,
    max_seq_length: int,
    batch_size: int,
) -> dict[str, Any]:
    review_df, llm_feature_df, abnormal_masks = load_routeL_review_frames(base_dir)
    tokenizer = build_tokenizer("llm_masked_logic", primary_model_name_or_path, max_seq_length=max_seq_length)
    return build_review_dataloaders(
        review_df=review_df,
        llm_feature_df=llm_feature_df,
        abnormal_masks=abnormal_masks,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
    )


def build_routeL_model(
    primary_model_name_or_path: str,
    numeric_feature_dim: int,
    vector_dim: int,
    secondary_model_name_or_path: str | None,
    freeze_primary: bool,
    freeze_secondary: bool,
    fusion_mode: str,
    use_anomaly_aux_loss: bool,
    anomaly_warmup_ratio: float,
    lambda_aux: float,
) -> RouteLLMMaskedLogicEncoder:
    del use_anomaly_aux_loss, anomaly_warmup_ratio, lambda_aux
    return RouteLLMMaskedLogicEncoder(
        primary_model_name_or_path=primary_model_name_or_path,
        numeric_feature_dim=numeric_feature_dim,
        vector_dim=vector_dim,
        secondary_model_name_or_path=secondary_model_name_or_path,
        freeze_primary=freeze_primary,
        freeze_secondary=freeze_secondary,
        fusion_mode=fusion_mode,
    )


def compute_routeL_losses(
    outputs: RouteLReviewEncoderOutput,
    labels: torch.Tensor,
    pos_weight_value: float,
    lambda_aux: float,
) -> dict[str, torch.Tensor]:
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, device=labels.device))
    main_loss = criterion(outputs.review_logit, labels)
    aux_loss = criterion(outputs.aux_logit, labels)
    total = main_loss + float(lambda_aux) * aux_loss
    return {"main_loss": main_loss, "aux_loss": aux_loss, "total_loss": total}


def compute_pos_weight(train_loader: Any, device: torch.device) -> float:
    label_batches = [batch["label"].numpy() for batch in train_loader]
    if not label_batches:
        return 1.0
    labels = np.concatenate(label_batches)
    pos = float(labels.sum())
    neg = float(len(labels) - pos)
    return neg / max(pos, 1.0)


def _run_epoch(
    model: RouteLLMMaskedLogicEncoder,
    dataloader: Any,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    pos_weight_value: float,
    lambda_aux: float,
    warmup_active: bool,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    losses: list[float] = []
    all_labels: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    all_aux_probs: list[np.ndarray] = []
    training = optimizer is not None
    model.train(training)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, device=device))

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        abnormal_mask = batch["abnormal_mask"].to(device)
        numeric_features = batch["numeric_features"].to(device)
        labels = batch["label"].to(device)

        with torch.set_grad_enabled(training):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                abnormal_token_mask=abnormal_mask,
                numeric_features=numeric_features,
                warmup_active=warmup_active,
            )
            main_loss = criterion(outputs.review_logit, labels)
            aux_loss = criterion(outputs.aux_logit, labels)
            loss = main_loss + float(lambda_aux) * aux_loss
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        losses.append(float(loss.detach().cpu()))
        all_labels.append(labels.detach().cpu().numpy())
        all_probs.append(torch.sigmoid(outputs.review_logit).detach().cpu().numpy())
        all_aux_probs.append(torch.sigmoid(outputs.aux_logit).detach().cpu().numpy())

    return (
        float(np.mean(losses) if losses else 0.0),
        np.concatenate(all_labels) if all_labels else np.asarray([], dtype=np.float32),
        np.concatenate(all_probs) if all_probs else np.asarray([], dtype=np.float32),
        np.concatenate(all_aux_probs) if all_aux_probs else np.asarray([], dtype=np.float32),
    )


def train_routeL_review_encoder(
    model: RouteLLMMaskedLogicEncoder,
    dataloaders: dict[str, Any],
    output_dir: str | Path,
    device: torch.device,
    learning_rate: float,
    num_epochs: int,
    patience: int,
    lambda_aux: float,
    fusion_mode: str,
    anomaly_warmup_ratio: float,
) -> tuple[Path, Path, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_label_batches = [batch["label"].numpy() for batch in dataloaders["train"]]
    if not train_label_batches:
        raise ValueError("Train split is empty after user-based splitting; cannot train review encoder.")
    train_labels = np.concatenate(train_label_batches)
    positive_count = float(train_labels.sum())
    negative_count = float(len(train_labels) - positive_count)
    pos_weight_value = negative_count / max(positive_count, 1.0)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
    best_state_path = output_dir / "best_review_encoder.pt"
    metrics_path = output_dir / "review_encoder_metrics.csv"
    best_val_auc = float("-inf")
    bad_epochs = 0
    epoch_rows: list[dict[str, Any]] = []
    model.to(device)

    for epoch_index in range(int(num_epochs)):
        warmup_active = bool(fusion_mode == "late" and epoch_index < max(int(np.ceil(num_epochs * anomaly_warmup_ratio)), 1))
        train_loss, train_y, train_prob, train_aux_prob = _run_epoch(
            model=model,
            dataloader=dataloaders["train"],
            device=device,
            optimizer=optimizer,
            pos_weight_value=pos_weight_value,
            lambda_aux=lambda_aux,
            warmup_active=warmup_active,
        )
        val_loss, val_y, val_prob, val_aux_prob = _run_epoch(
            model=model,
            dataloader=dataloaders["val"],
            device=device,
            optimizer=None,
            pos_weight_value=pos_weight_value,
            lambda_aux=lambda_aux,
            warmup_active=warmup_active,
        )
        train_metrics = compute_binary_metrics(train_y, train_prob)
        val_metrics = compute_binary_metrics(val_y, val_prob)
        epoch_row = {
            "epoch": epoch_index + 1,
            "fusion_mode": fusion_mode,
            "warmup_active": bool(warmup_active),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_auc": train_metrics["auc"],
            "train_ap": train_metrics["ap"],
            "train_recall": train_metrics["recall"],
            "train_precision": train_metrics["precision"],
            "train_f1": train_metrics["f1"],
            "val_auc": val_metrics["auc"],
            "val_ap": val_metrics["ap"],
            "val_recall": val_metrics["recall"],
            "val_precision": val_metrics["precision"],
            "val_f1": val_metrics["f1"],
            "train_aux_prob_mean": float(np.mean(train_aux_prob)) if len(train_aux_prob) else 0.0,
            "val_aux_prob_mean": float(np.mean(val_aux_prob)) if len(val_aux_prob) else 0.0,
        }
        epoch_rows.append(epoch_row)
        if val_metrics["auc"] >= best_val_auc:
            best_val_auc = val_metrics["auc"]
            bad_epochs = 0
            torch.save(model.state_dict(), best_state_path)
        else:
            bad_epochs += 1
            if bad_epochs >= int(patience):
                break

    metrics_df = pd.DataFrame(epoch_rows)
    metrics_df.to_csv(metrics_path, index=False)
    return best_state_path, metrics_path, metrics_df


def encode_routeL_all_reviews(
    model: RouteLLMMaskedLogicEncoder,
    dataloader: Any,
    review_df: pd.DataFrame,
    checkpoint_path: str | Path,
    metrics_path: str | Path,
    device: torch.device,
    fusion_mode: str,
    anomaly_warmup_ratio: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    del metrics_path
    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
    model.to(device)
    model.eval()
    rows: list[dict[str, Any]] = []
    review_vectors: list[np.ndarray] = []
    text_vectors: list[np.ndarray] = []
    review_lookup = review_df.set_index("review_node_id")
    with torch.no_grad():
        for batch in dataloader:
            review_ids = batch["review_id"].cpu().numpy()
            warmup_active = bool(fusion_mode == "late" and anomaly_warmup_ratio > 0.0)
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                abnormal_token_mask=batch["abnormal_mask"].to(device),
                numeric_features=batch["numeric_features"].to(device),
                warmup_active=warmup_active,
            )
            probs = torch.sigmoid(outputs.review_logit).cpu().numpy()
            aux_probs = torch.sigmoid(outputs.aux_logit).cpu().numpy()
            gates = outputs.gate.detach().cpu().numpy()
            review_vec = outputs.review_vector.detach().cpu().numpy()
            text_vec = outputs.text_vector.detach().cpu().numpy()
            for index, review_id in enumerate(review_ids):
                row = review_lookup.loc[int(review_id)]
                rows.append(
                    {
                        "review_node_id": int(review_id),
                        "user_id": row["user_id"],
                        "product_id": row["product_id"],
                        "rating": float(row["rating"]),
                        "review_date": row["review_date"],
                        "review_label": int(row["review_label"]),
                        "split": row["split"],
                        "review_prob": float(probs[index]),
                        "aux_prob": float(aux_probs[index]),
                        "gate": float(gates[index]),
                    }
                )
                review_vectors.append(review_vec[index])
                text_vectors.append(text_vec[index])
    review_output_df = pd.DataFrame(rows).sort_values("review_node_id").reset_index(drop=True)
    return review_output_df, np.asarray(review_vectors, dtype=np.float32), np.asarray(text_vectors, dtype=np.float32)
