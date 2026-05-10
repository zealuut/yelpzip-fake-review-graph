from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib
import json
import re

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_distances
from torch import nn

from graph.review_training import (
    build_review_dataloaders,
    build_tokenizer,
    compute_binary_metrics,
)

from .review_models_routeL_text import RouteLTextEvidenceEncoder, RouteLTextEvidenceOutput


_WORD_RE = re.compile(r"[A-Za-z']+")
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")
_SUPERLATIVE_RE = re.compile(r".*(est|most)$", re.IGNORECASE)
_POSITIVE_WORDS = {
    "amazing", "awesome", "best", "excellent", "fantastic", "great", "love", "perfect", "wonderful",
}
_NEGATIVE_WORDS = {
    "awful", "bad", "disappointing", "hate", "horrible", "poor", "terrible", "worst",
}
_FIRST_PERSON = {"i", "me", "my", "mine", "we", "us", "our", "ours"}


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


def _simple_sentences(text: str) -> list[str]:
    parts = [part.strip() for part in _SENTENCE_SPLIT_RE.split(str(text or "")) if part.strip()]
    return parts if parts else [str(text or "").strip()]


def _simple_tokens(text: str) -> list[str]:
    return _WORD_RE.findall(str(text or "").lower())


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def build_psycholinguistic_style_frame(review_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in review_df.sort_values("review_node_id").itertuples(index=False):
        text = str(getattr(row, "review_text", "") or "")
        tokens = _simple_tokens(text)
        sentences = _simple_sentences(text)
        sentence_token_lengths = [len(_simple_tokens(sentence)) for sentence in sentences]
        token_count = len(tokens)
        unique_count = len(set(tokens))
        all_caps_count = sum(1 for token in re.findall(r"\b[A-Z]{2,}\b", text))
        first_person_count = sum(1 for token in tokens if token in _FIRST_PERSON)
        superlative_count = sum(1 for token in tokens if _SUPERLATIVE_RE.match(token))
        exclamation_count = text.count("!")
        question_count = text.count("?")
        punctuation_count = len(re.findall(r"[!?.,;:]", text))
        number_count = len(re.findall(r"\d", text))
        pos_hits = sum(1 for token in tokens if token in _POSITIVE_WORDS)
        neg_hits = sum(1 for token in tokens if token in _NEGATIVE_WORDS)
        sentiment_intensity = _safe_div(abs(pos_hits - neg_hits), max(token_count, 1))
        rows.append(
            {
                "review_node_id": int(getattr(row, "review_node_id")),
                "review_length": float(token_count),
                "sentence_count": float(len(sentences)),
                "avg_sentence_length": float(np.mean(sentence_token_lengths) if sentence_token_lengths else 0.0),
                "exclamation_count": float(exclamation_count),
                "question_count": float(question_count),
                "all_caps_ratio": _safe_div(all_caps_count, max(token_count, 1)),
                "first_person_ratio": _safe_div(first_person_count, max(token_count, 1)),
                "superlative_ratio": _safe_div(superlative_count, max(token_count, 1)),
                "lexical_diversity": _safe_div(unique_count, max(token_count, 1)),
                "punctuation_density": _safe_div(punctuation_count, max(len(text), 1)),
                "number_ratio": _safe_div(number_count, max(len(text), 1)),
                "sentiment_intensity": float(sentiment_intensity),
            }
        )
    frame = pd.DataFrame(rows).sort_values("review_node_id").reset_index(drop=True)
    style_cols = [c for c in frame.columns if c != "review_node_id"]
    for col in style_cols:
        values = pd.to_numeric(frame[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        if len(values) > 1:
            lo = float(np.min(values))
            hi = float(np.max(values))
            if hi > lo:
                values = (values - lo) / (hi - lo)
        frame[col] = values.astype(np.float32)
    return frame.set_index("review_node_id", drop=False)


def compute_review_text_embeddings(
    model: RouteLTextEvidenceEncoder,
    dataloader: Any,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                extra_features=None,
            )
            review_ids = batch["review_id"].detach().cpu().numpy().astype(np.int64)
            text_vectors = outputs.text_vector.detach().cpu().numpy().astype(np.float32)
            for idx, review_id in enumerate(review_ids):
                rows.append(
                    {
                        "review_node_id": int(review_id),
                        "text_vector": text_vectors[idx],
                    }
                )
    return pd.DataFrame(rows).sort_values("review_node_id").reset_index(drop=True)


def build_semantic_drift_frame(
    review_df: pd.DataFrame,
    text_embedding_frame: pd.DataFrame,
) -> pd.DataFrame:
    merged = review_df[["review_node_id", "product_id", "split"]].merge(
        text_embedding_frame,
        on="review_node_id",
        how="left",
        validate="one_to_one",
    )
    train_df = merged[merged["split"] == "train"].copy()
    centroids: dict[str, np.ndarray] = {}
    for product_id, group in train_df.groupby("product_id"):
        vectors = np.stack(group["text_vector"].to_list()).astype(np.float32)
        centroids[str(product_id)] = vectors.mean(axis=0).astype(np.float32)

    fallback_centroid = None
    if centroids:
        fallback_centroid = np.stack(list(centroids.values())).mean(axis=0).astype(np.float32)

    rows: list[dict[str, Any]] = []
    for row in merged.itertuples(index=False):
        vector = np.asarray(getattr(row, "text_vector"), dtype=np.float32)
        centroid = centroids.get(str(getattr(row, "product_id")), fallback_centroid)
        if centroid is None:
            drift = 0.0
        else:
            drift = float(cosine_distances(vector.reshape(1, -1), centroid.reshape(1, -1))[0, 0])
        rows.append(
            {
                "review_node_id": int(getattr(row, "review_node_id")),
                "semantic_drift": drift,
            }
        )
    frame = pd.DataFrame(rows).sort_values("review_node_id").reset_index(drop=True)
    values = pd.to_numeric(frame["semantic_drift"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    if len(values) > 1:
        lo = float(np.min(values))
        hi = float(np.max(values))
        if hi > lo:
            values = (values - lo) / (hi - lo)
    frame["semantic_drift"] = values.astype(np.float32)
    return frame.set_index("review_node_id", drop=False)


def build_routeL_text_model(
    primary_model_name_or_path: str,
    vector_dim: int,
    experiment_kind: str,
    extra_feature_dim: int = 0,
    topk_tokens: int = 8,
) -> RouteLTextEvidenceEncoder:
    return RouteLTextEvidenceEncoder(
        primary_model_name_or_path=primary_model_name_or_path,
        vector_dim=vector_dim,
        experiment_kind=experiment_kind,
        extra_feature_dim=extra_feature_dim,
        topk_tokens=topk_tokens,
    )


def compute_routeL_text_losses(
    outputs: RouteLTextEvidenceOutput,
    labels: torch.Tensor,
    pos_weight_value: float,
    lambda_evidence: float,
    lambda_sparse: float,
) -> dict[str, torch.Tensor]:
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, device=labels.device))
    main_loss = criterion(outputs.review_logit, labels)
    evidence_loss = criterion(outputs.evidence_logit, labels)
    sparse_loss = outputs.token_evidence_scores.mean()
    total_loss = main_loss + float(lambda_evidence) * evidence_loss + float(lambda_sparse) * sparse_loss
    return {
        "main_loss": main_loss,
        "evidence_loss": evidence_loss,
        "sparse_loss": sparse_loss,
        "total_loss": total_loss,
    }


def _lookup_extra_features(
    review_ids: np.ndarray,
    extra_feature_frame: pd.DataFrame | None,
) -> torch.Tensor | None:
    if extra_feature_frame is None:
        return None
    sub = extra_feature_frame.loc[review_ids.astype(np.int64)]
    feature_cols = [c for c in sub.columns if c not in {"review_node_id"}]
    return torch.tensor(sub[feature_cols].to_numpy(dtype=np.float32))


def _run_epoch(
    model: RouteLTextEvidenceEncoder,
    dataloader: Any,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    pos_weight_value: float,
    lambda_evidence: float,
    lambda_sparse: float,
    extra_feature_frame: pd.DataFrame | None = None,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    losses: list[float] = []
    all_labels: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    all_evidence_probs: list[np.ndarray] = []
    training = optimizer is not None
    model.train(training)

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        extra_tensor = None
        if extra_feature_frame is not None:
            row_ids = batch["review_id"].detach().cpu().numpy().astype(np.int64)
            extra_tensor = _lookup_extra_features(row_ids, extra_feature_frame)
            if extra_tensor is not None:
                extra_tensor = extra_tensor.to(device)

        with torch.set_grad_enabled(training):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                extra_features=extra_tensor,
            )
            loss_parts = compute_routeL_text_losses(
                outputs=outputs,
                labels=labels,
                pos_weight_value=pos_weight_value,
                lambda_evidence=lambda_evidence,
                lambda_sparse=lambda_sparse,
            )
            loss = loss_parts["total_loss"]
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        losses.append(float(loss.detach().cpu()))
        all_labels.append(labels.detach().cpu().numpy())
        all_probs.append(torch.sigmoid(outputs.review_logit).detach().cpu().numpy())
        all_evidence_probs.append(torch.sigmoid(outputs.evidence_logit).detach().cpu().numpy())

    return (
        float(np.mean(losses) if losses else 0.0),
        np.concatenate(all_labels) if all_labels else np.asarray([], dtype=np.float32),
        np.concatenate(all_probs) if all_probs else np.asarray([], dtype=np.float32),
        np.concatenate(all_evidence_probs) if all_evidence_probs else np.asarray([], dtype=np.float32),
    )


def train_routeL_text_encoder(
    model: RouteLTextEvidenceEncoder,
    dataloaders: dict[str, Any],
    output_dir: str | Path,
    device: torch.device,
    learning_rate: float,
    num_epochs: int,
    patience: int,
    lambda_evidence: float,
    lambda_sparse: float,
    extra_feature_frame: pd.DataFrame | None = None,
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
        train_loss, train_y, train_prob, train_evidence_prob = _run_epoch(
            model=model,
            dataloader=dataloaders["train"],
            device=device,
            optimizer=optimizer,
            pos_weight_value=pos_weight_value,
            lambda_evidence=lambda_evidence,
            lambda_sparse=lambda_sparse,
            extra_feature_frame=extra_feature_frame,
        )
        val_loss, val_y, val_prob, val_evidence_prob = _run_epoch(
            model=model,
            dataloader=dataloaders["val"],
            device=device,
            optimizer=None,
            pos_weight_value=pos_weight_value,
            lambda_evidence=lambda_evidence,
            lambda_sparse=lambda_sparse,
            extra_feature_frame=extra_feature_frame,
        )
        train_metrics = compute_binary_metrics(train_y, train_prob)
        val_metrics = compute_binary_metrics(val_y, val_prob)
        epoch_rows.append(
            {
                "epoch": epoch_index + 1,
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
                "train_evidence_prob_mean": float(np.mean(train_evidence_prob)) if len(train_evidence_prob) else 0.0,
                "val_evidence_prob_mean": float(np.mean(val_evidence_prob)) if len(val_evidence_prob) else 0.0,
            }
        )
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
    model: RouteLTextEvidenceEncoder,
    dataloader: Any,
    review_df: pd.DataFrame,
    checkpoint_path: str | Path,
    device: torch.device,
    extra_feature_frame: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
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
            extra_tensor = None
            if extra_feature_frame is not None:
                extra_tensor = _lookup_extra_features(review_ids.astype(np.int64), extra_feature_frame)
                if extra_tensor is not None:
                    extra_tensor = extra_tensor.to(device)
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                extra_features=extra_tensor,
            )
            probs = torch.sigmoid(outputs.review_logit).cpu().numpy()
            evidence_probs = torch.sigmoid(outputs.evidence_logit).cpu().numpy()
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
                        "evidence_prob": float(evidence_probs[index]),
                        "gate": float(gates[index]),
                    }
                )
                review_vectors.append(review_vec[index])
                text_vectors.append(text_vec[index])
    review_output_df = pd.DataFrame(rows).sort_values("review_node_id").reset_index(drop=True)
    return review_output_df, np.asarray(review_vectors, dtype=np.float32), np.asarray(text_vectors, dtype=np.float32)


def write_feature_hash(path: str | Path, array: np.ndarray) -> None:
    array = np.ascontiguousarray(array.astype(np.float32))
    digest = hashlib.sha256(array.tobytes()).hexdigest()
    Path(path).write_text(digest + "\n", encoding="utf-8")


def make_llm_mask_stats_from_review_scores(review_scores_df: pd.DataFrame) -> pd.DataFrame:
    hit = (
        review_scores_df["num_abnormal_patterns"].gt(0)
        if "num_abnormal_patterns" in review_scores_df.columns
        else pd.Series(False, index=review_scores_df.index)
    )
    fake = review_scores_df.get("review_label", pd.Series(0, index=review_scores_df.index)).astype(int) == 1
    real = ~fake
    pattern_cols = [c for c in review_scores_df.columns if c.startswith("pattern_type__")]
    row = {
        "num_reviews": int(len(review_scores_df)),
        "mask_hit_rate": float(hit.mean()) if len(hit) else 0.0,
        "avg_pattern_count": float(pd.to_numeric(review_scores_df.get("num_abnormal_patterns", 0), errors="coerce").fillna(0.0).mean()) if len(review_scores_df) else 0.0,
        "llm_error_rate": float(review_scores_df.get("mask_source", pd.Series("", index=review_scores_df.index)).astype(str).eq("LLM_ERROR").mean()) if len(review_scores_df) else 0.0,
        "fake_mask_hit_rate": float(hit[fake].mean()) if fake.any() else 0.0,
        "real_mask_hit_rate": float(hit[real].mean()) if real.any() else 0.0,
        "pattern_type_distribution": json.dumps(
            {col: float(pd.to_numeric(review_scores_df[col], errors="coerce").fillna(0.0).mean()) for col in pattern_cols},
            ensure_ascii=False,
        ),
    }
    return pd.DataFrame([row])
