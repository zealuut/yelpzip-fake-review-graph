from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - local fallback
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []

try:
    from transformers import AutoTokenizer
except Exception:  # pragma: no cover - resolved on the server/runtime
    AutoTokenizer = None

from .llm_utils import numeric_feature_columns
from .review_models import LLMMaskedLogicEncoder, MockReviewEncoder, ReviewEncoderOutput


TOKEN_RE = re.compile(r"\S+")


@dataclass
class ReviewEncodingArtifacts:
    review_output_df: pd.DataFrame
    review_vectors: np.ndarray
    text_vectors: np.ndarray
    checkpoint_path: Path
    metrics_path: Path


class SimpleWhitespaceTokenizer:
    def __init__(self, max_length: int = 256) -> None:
        self.max_length = max_length
        self.name_or_path = "simple-whitespace-tokenizer"

    def _encode_one(self, text: str, max_length: int, return_offsets_mapping: bool) -> dict[str, Any]:
        text = str(text or "")
        input_ids = [101]
        offsets = [(0, 0)]
        for match in TOKEN_RE.finditer(text):
            token = match.group(0).lower()
            token_id = abs(hash(token)) % 4096 + 1000
            input_ids.append(token_id)
            offsets.append((match.start(), match.end()))
            if len(input_ids) >= max_length - 1:
                break
        input_ids.append(102)
        offsets.append((0, 0))

        attention_mask = [1] * len(input_ids)
        while len(input_ids) < max_length:
            input_ids.append(0)
            attention_mask.append(0)
            offsets.append((0, 0))

        payload = {
            "input_ids": input_ids[:max_length],
            "attention_mask": attention_mask[:max_length],
        }
        if return_offsets_mapping:
            payload["offset_mapping"] = offsets[:max_length]
        return payload

    def __call__(
        self,
        texts: str | list[str],
        padding: str = "max_length",
        truncation: bool = True,
        max_length: int | None = None,
        return_offsets_mapping: bool = False,
        return_tensors: str | None = None,
    ) -> dict[str, Any]:
        max_length = max_length or self.max_length
        is_single = isinstance(texts, str)
        text_list = [texts] if is_single else list(texts)
        encoded = [
            self._encode_one(text, max_length=max_length, return_offsets_mapping=return_offsets_mapping)
            for text in text_list
        ]

        keys = encoded[0].keys()
        batch = {key: [item[key] for item in encoded] for key in keys}
        if is_single and return_tensors != "pt":
            return {key: values[0] for key, values in batch.items()}
        if return_tensors == "pt":
            tensor_batch: dict[str, Any] = {}
            for key, values in batch.items():
                if key == "offset_mapping":
                    tensor_batch[key] = values
                else:
                    tensor_batch[key] = torch.tensor(values, dtype=torch.long)
            return tensor_batch
        return batch


class ReviewDataset(Dataset):
    def __init__(
        self,
        review_ids: np.ndarray,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        abnormal_mask: torch.Tensor,
        numeric_features: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        self.review_ids = torch.tensor(review_ids, dtype=torch.long)
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.abnormal_mask = abnormal_mask
        self.numeric_features = numeric_features
        self.labels = labels

    def __len__(self) -> int:
        return int(self.review_ids.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "review_id": self.review_ids[index],
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "abnormal_mask": self.abnormal_mask[index],
            "numeric_features": self.numeric_features[index],
            "label": self.labels[index],
        }


def build_tokenizer(review_encoder: str, primary_model_name_or_path: str, max_seq_length: int) -> Any:
    if review_encoder == "mock":
        return SimpleWhitespaceTokenizer(max_length=max_seq_length)
    if AutoTokenizer is None:
        raise ImportError("transformers is required for the non-mock tokenizer path")
    return AutoTokenizer.from_pretrained(primary_model_name_or_path, use_fast=True)


def build_review_model(
    review_encoder: str,
    primary_model_name_or_path: str,
    numeric_feature_dim: int,
    vector_dim: int,
    secondary_model_name_or_path: str | None = None,
    freeze_primary: bool = False,
    freeze_secondary: bool = False,
) -> nn.Module:
    if review_encoder == "mock":
        return MockReviewEncoder(numeric_feature_dim=numeric_feature_dim, vector_dim=vector_dim)
    if review_encoder != "llm_masked_logic":
        raise ValueError(f"Unsupported review encoder: {review_encoder}")
    return LLMMaskedLogicEncoder(
        primary_model_name_or_path=primary_model_name_or_path,
        numeric_feature_dim=numeric_feature_dim,
        vector_dim=vector_dim,
        secondary_model_name_or_path=secondary_model_name_or_path,
        freeze_primary=freeze_primary,
        freeze_secondary=freeze_secondary,
    )


def build_review_dataloaders(
    review_df: pd.DataFrame,
    llm_feature_df: pd.DataFrame,
    abnormal_masks: np.ndarray,
    tokenizer: Any,
    max_seq_length: int,
    batch_size: int,
) -> dict[str, DataLoader]:
    ordered_reviews = review_df.sort_values("review_node_id").reset_index(drop=True)
    ordered_features = llm_feature_df.sort_values("review_node_id").reset_index(drop=True)

    if ordered_reviews["review_node_id"].tolist() != ordered_features["review_node_id"].tolist():
        raise ValueError("Review frame and LLM feature frame are misaligned on review_node_id.")

    encoded = tokenizer(
        ordered_reviews["review_text"].tolist(),
        padding="max_length",
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    abnormal_mask = torch.tensor(abnormal_masks, dtype=torch.float32)
    numeric_tensor = torch.tensor(ordered_features[numeric_feature_columns()].to_numpy(dtype=np.float32), dtype=torch.float32)
    label_tensor = torch.tensor(ordered_reviews["review_label"].to_numpy(dtype=np.float32), dtype=torch.float32)

    dataloaders: dict[str, DataLoader] = {}
    for split_name in ["train", "val", "test"]:
        index_mask = ordered_reviews["split"].eq(split_name).to_numpy()
        tensor_mask = torch.tensor(index_mask, dtype=torch.bool)
        dataset = ReviewDataset(
            review_ids=ordered_reviews.loc[index_mask, "review_node_id"].to_numpy(dtype=np.int64),
            input_ids=input_ids[tensor_mask],
            attention_mask=attention_mask[tensor_mask],
            abnormal_mask=abnormal_mask[tensor_mask],
            numeric_features=numeric_tensor[tensor_mask],
            labels=label_tensor[tensor_mask],
        )
        dataloaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
        )

    dataloaders["all"] = DataLoader(
        ReviewDataset(
            review_ids=ordered_reviews["review_node_id"].to_numpy(dtype=np.int64),
            input_ids=input_ids,
            attention_mask=attention_mask,
            abnormal_mask=abnormal_mask,
            numeric_features=numeric_tensor,
            labels=label_tensor,
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    return dataloaders


def compute_binary_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    preds = (probs >= threshold).astype(int)
    return {
        "auc": float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else 0.0,
        "ap": float(average_precision_score(labels, probs)) if len(np.unique(labels)) > 1 else 0.0,
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "accuracy": float(accuracy_score(labels, preds)),
    }


def _run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.Module,
) -> tuple[float, np.ndarray, np.ndarray]:
    losses: list[float] = []
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    training = optimizer is not None
    model.train(training)

    iterator = tqdm(dataloader, desc="Train" if training else "Eval", leave=False)
    for batch in iterator:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        abnormal_mask = batch["abnormal_mask"].to(device)
        numeric_features = batch["numeric_features"].to(device)
        labels = batch["label"].to(device)

        with torch.set_grad_enabled(training):
            outputs: ReviewEncoderOutput = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                abnormal_token_mask=abnormal_mask,
                numeric_features=numeric_features,
            )
            loss = criterion(outputs.review_logit, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        losses.append(float(loss.detach().cpu()))
        probs = torch.sigmoid(outputs.review_logit).detach().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.detach().cpu().numpy())

    return (
        float(np.mean(losses) if losses else 0.0),
        np.concatenate(all_labels) if all_labels else np.asarray([], dtype=np.float32),
        np.concatenate(all_probs) if all_probs else np.asarray([], dtype=np.float32),
    )


def train_review_encoder(
    model: nn.Module,
    dataloaders: dict[str, DataLoader],
    output_dir: str | Path,
    device: torch.device,
    learning_rate: float,
    num_epochs: int,
    patience: int,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_label_batches = [batch["label"].numpy() for batch in dataloaders["train"]]
    if not train_label_batches:
        raise ValueError("Train split is empty after user-based splitting; cannot train review encoder.")
    train_labels = np.concatenate(train_label_batches)
    positive_count = float(train_labels.sum())
    negative_count = float(len(train_labels) - positive_count)
    pos_weight_value = negative_count / max(positive_count, 1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, device=device))
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
    )

    best_state_path = output_dir / "best_review_encoder.pt"
    metrics_path = output_dir / "review_encoder_metrics.json"
    best_val_auc = float("-inf")
    best_payload: dict[str, Any] = {}
    bad_epochs = 0

    model.to(device)
    for epoch_index in range(num_epochs):
        train_loss, train_y, train_prob = _run_epoch(
            model=model,
            dataloader=dataloaders["train"],
            device=device,
            optimizer=optimizer,
            criterion=criterion,
        )
        val_loss, val_y, val_prob = _run_epoch(
            model=model,
            dataloader=dataloaders["val"],
            device=device,
            optimizer=None,
            criterion=criterion,
        )
        train_metrics = compute_binary_metrics(train_y, train_prob)
        val_metrics = compute_binary_metrics(val_y, val_prob)
        epoch_payload = {
            "epoch": epoch_index + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        if val_metrics["auc"] >= best_val_auc:
            best_val_auc = val_metrics["auc"]
            bad_epochs = 0
            torch.save(model.state_dict(), best_state_path)
            best_payload = epoch_payload
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    metrics_path.write_text(
        pd.Series(best_payload).to_json(force_ascii=False, indent=2),
        encoding="utf-8",
    )
    return best_state_path, metrics_path


def encode_all_reviews(
    model: nn.Module,
    dataloader: DataLoader,
    review_df: pd.DataFrame,
    checkpoint_path: str | Path,
    metrics_path: str | Path,
    device: torch.device,
) -> ReviewEncodingArtifacts:
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    rows: list[dict[str, Any]] = []
    review_vectors: list[np.ndarray] = []
    text_vectors: list[np.ndarray] = []
    review_lookup = review_df.set_index("review_node_id")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Encoding reviews", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            abnormal_mask = batch["abnormal_mask"].to(device)
            numeric_features = batch["numeric_features"].to(device)
            review_ids = batch["review_id"].cpu().numpy()

            outputs: ReviewEncoderOutput = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                abnormal_token_mask=abnormal_mask,
                numeric_features=numeric_features,
            )
            probs = torch.sigmoid(outputs.review_logit).cpu().numpy()
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
                        "p_fake_review": float(probs[index]),
                        "review_gate": float(gates[index]),
                    }
                )
                review_vectors.append(review_vec[index])
                text_vectors.append(text_vec[index])

    return ReviewEncodingArtifacts(
        review_output_df=pd.DataFrame(rows).sort_values("review_node_id").reset_index(drop=True),
        review_vectors=np.asarray(review_vectors, dtype=np.float32),
        text_vectors=np.asarray(text_vectors, dtype=np.float32),
        checkpoint_path=Path(checkpoint_path),
        metrics_path=Path(metrics_path),
    )
