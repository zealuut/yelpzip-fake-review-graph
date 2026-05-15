"""RouteV queue runner: vector-quality-aware D1 variants.

This route intentionally keeps the D1 data and graph protocol intact. The only
moving part is how the review encoder checkpoint/vector objective is chosen:

- V_control: Route baseline pack constructed by the baseline route.
- V0: same training as D1, checkpoint selected by a train->val user-vector proxy.
- V1: D1 training plus a user-vector separability regularizer.
- V2: D1 review classifier plus a separate graph-vector head trained by the
  user-vector regularizer.

The runner is route-local and does not require shared run_final_experiment.py
changes such as --save_all_checkpoints.
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
DEFAULT_CONFIG = ROUTE_ROOT / "configs" / "routeV_variants.json"

sys.path.insert(0, str(PROJECT_ROOT))

from graph.data_utils import ensure_dir, prepare_graph_data  # noqa: E402
from graph.graph_pipeline import (  # noqa: E402
    build_edge_frames,
    build_review_and_user_artifacts,
    build_self_feature_matrix,
    compute_edge_stats,
)
from graph.llm_utils import build_llm_features_and_masks, numeric_feature_columns  # noqa: E402
from graph.relation_model import run_relation_aggregation_experiments  # noqa: E402
from graph.review_training import build_review_model, build_tokenizer, compute_binary_metrics  # noqa: E402
from graph.routes.routeV_vector_quality.src.dual_head_encoder import DualHeadWrapper  # noqa: E402
from graph.routes.routeV_vector_quality.src.user_level_proxy import (  # noqa: E402
    compute_user_vector_proxy_train_eval,
)
from graph.routes.routeV_vector_quality.src.vector_reg_loss import (  # noqa: E402
    TripletUserVectorLoss,
    UserVectorSeparabilityLoss,
)


@dataclass
class RouteVEncodingArtifacts:
    review_output_df: pd.DataFrame
    review_vectors: np.ndarray
    text_vectors: np.ndarray
    checkpoint_path: Path
    metrics_path: Path


@dataclass
class RouteVContext:
    prepared: Any
    llm_feature_df: pd.DataFrame
    abnormal_masks: np.ndarray
    tokenizer: Any
    dataloaders: dict[str, DataLoader]
    ordered_reviews: pd.DataFrame
    config: dict[str, Any]


class RouteVReviewDataset(Dataset):
    def __init__(
        self,
        review_ids: np.ndarray,
        user_indices: np.ndarray,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        abnormal_mask: torch.Tensor,
        numeric_features: torch.Tensor,
        labels: torch.Tensor,
        user_labels: torch.Tensor,
    ) -> None:
        self.review_ids = torch.tensor(review_ids, dtype=torch.long)
        self.user_indices = torch.tensor(user_indices, dtype=torch.long)
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.abnormal_mask = abnormal_mask
        self.numeric_features = numeric_features
        self.labels = labels
        self.user_labels = user_labels

    def __len__(self) -> int:
        return int(self.review_ids.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "review_id": self.review_ids[index],
            "user_id_idx": self.user_indices[index],
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "abnormal_mask": self.abnormal_mask[index],
            "numeric_features": self.numeric_features[index],
            "label": self.labels[index],
            "user_label": self.user_labels[index],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RouteV vector quality queue runner")
    parser.add_argument("--output_root", required=True, help="Root output directory for this run")
    parser.add_argument("--config_path", default=str(DEFAULT_CONFIG))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--smoke_max_users", type=int, default=160)
    parser.add_argument("--variants", nargs="*", default=None, help="Run only these variant names")
    parser.add_argument("--skip_control", action="store_true")
    parser.add_argument("--d1_floor_auc", type=float, default=None)
    return parser.parse_args()


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_target_row(csv_path: Path, edge_set: str, model_name: str | None = None) -> dict[str, Any]:
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    if df.empty:
        return {}
    rows = df.copy()
    if edge_set and "edge_set" in rows.columns:
        filtered = rows[rows["edge_set"].eq(edge_set)]
        if not filtered.empty:
            rows = filtered
    if model_name and "model_name" in rows.columns:
        filtered = rows[rows["model_name"].eq(model_name)]
        if not filtered.empty:
            rows = filtered
    if "auc" in rows.columns:
        rows = rows.sort_values("auc", ascending=False)
    return rows.iloc[0].to_dict()


def _copytree_no_overwrite(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Missing source directory for RouteV control: {source}")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing RouteV control path: {destination}")
    shutil.copytree(source, destination, symlinks=True)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _shared_defaults(config: dict[str, Any]) -> dict[str, Any]:
    shared = config.get("shared_defaults") or config.get("base_protocol_args")
    if not shared:
        raise KeyError("RouteV config must contain shared_defaults or base_protocol_args")
    return dict(shared)


def _routev_baseline_metadata(config: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    route_contract = dict(config.get("route_contract") or {})
    baseline_reference = dict(config.get("baseline_reference") or {})
    control_variant = route_contract.get("baseline_variant", "V_control")
    variant_name = variant.get("name")
    is_control = bool(variant.get("is_control", variant_name == control_variant))
    is_route_baseline = bool(variant.get("is_route_baseline", False))
    return {
        "route": "routeV_vector_quality",
        "variant": variant_name,
        "is_route_baseline": is_route_baseline,
        "is_route_control": is_control,
        "control_id": variant.get("control_id") if is_control else None,
        "control_role": variant.get("control_role") if is_control else None,
        "comparison_control": control_variant if not is_control else None,
        "reference_route_baseline": baseline_reference.get("route_baseline"),
        "reference_route_baseline_id": baseline_reference.get("route_baseline_id"),
        "reference_route_baseline_mode": baseline_reference.get("route_baseline_mode"),
        "reference_route_baseline_fresh_retrained_artifact": baseline_reference.get("route_baseline_fresh_retrained_artifact"),
        "reference_route_baseline_fixed_artifact_graph_only": baseline_reference.get("route_baseline_fixed_artifact_graph_only"),
        "comparison_scope": variant.get(
            "comparison_scope",
            route_contract.get("comparison_rule", "Compare only within the same RouteV queue/protocol."),
        ),
        "route_contract": route_contract,
        "baseline_reference": baseline_reference,
    }


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


def _attach_user_labels_to_reviews(review_df: pd.DataFrame, user_df: pd.DataFrame) -> pd.DataFrame:
    """Attach explicit prepared.user_df user labels to review rows."""
    if "user_id" not in review_df.columns:
        raise ValueError("review_df must contain user_id to attach user_label.")
    if "user_id" not in user_df.columns or "user_label" not in user_df.columns:
        raise ValueError("user_df must contain user_id and user_label for RouteV strict labels.")
    user_label_map = user_df.set_index(user_df["user_id"].astype(str))["user_label"].astype(int).to_dict()
    labeled = review_df.copy()
    mapped = labeled["user_id"].astype(str).map(user_label_map)
    if mapped.isna().any():
        missing = sorted(labeled.loc[mapped.isna(), "user_id"].astype(str).unique().tolist())
        raise ValueError(f"Missing user_label for {len(missing)} RouteV users; examples: {missing[:5]}")
    labeled["user_label"] = mapped.astype(int)
    return labeled


def _align_user_artifacts_to_prepared_labels(user_df: pd.DataFrame, prepared_user_df: pd.DataFrame) -> pd.DataFrame:
    """Force RouteV graph-stage user labels/splits to the prepared user table."""
    required_columns = {"user_id", "user_label", "split"}
    missing_columns = sorted(required_columns - set(prepared_user_df.columns))
    if missing_columns:
        raise ValueError(f"prepared.user_df missing required RouteV columns: {missing_columns}")
    lookup = prepared_user_df.copy()
    lookup["user_id"] = lookup["user_id"].astype(str)
    user_label_map = lookup.set_index("user_id")["user_label"].astype(int).to_dict()
    split_map = lookup.set_index("user_id")["split"].astype(str).to_dict()

    aligned = user_df.copy()
    aligned["user_id"] = aligned["user_id"].astype(str)
    mapped_labels = aligned["user_id"].map(user_label_map)
    mapped_splits = aligned["user_id"].map(split_map)
    if mapped_labels.isna().any() or mapped_splits.isna().any():
        missing = sorted(aligned.loc[mapped_labels.isna() | mapped_splits.isna(), "user_id"].unique().tolist())
        raise ValueError(f"Missing prepared user labels/splits for {len(missing)} RouteV users; examples: {missing[:5]}")
    aligned["user_label"] = mapped_labels.astype(int)
    aligned["split"] = mapped_splits.astype(str)
    return aligned


def _artifact_reuse_manifest() -> dict[str, Any]:
    return {
        "experiment_type": "fresh_d1_train",
        "policy": "RouteV regenerates all learned/vector/edge artifacts for every variant; only unchanged data protocol artifacts and static behavior components are strict-reusable.",
        "items": [
            {
                "path": "prepared_data/reviews_canonical.csv",
                "class": "canonical_data_split",
                "reuse_mode": "strict_reusable",
                "reason": "Source data, filtering, balancing, and split policy are unchanged under D1 ce6a9d6.",
            },
            {
                "path": "prepared_data/users_canonical.csv",
                "class": "canonical_user_split",
                "reuse_mode": "strict_reusable",
                "reason": "Used as the explicit source of user_label for RouteV proxy/regularizer semantics.",
            },
            {
                "path": "review_encoder/best_review_encoder.pt",
                "class": "review_checkpoint",
                "reuse_mode": "regenerated",
                "reason": "Every RouteV variant trains a fresh encoder checkpoint under its declared objective.",
            },
            {
                "path": "logic_vectors/review_text_vectors.npy",
                "class": "learned_text_vectors",
                "reuse_mode": "regenerated",
                "reason": "Text vectors are emitted by the freshly trained review encoder.",
            },
            {
                "path": "logic_vectors/user_text_vectors.npy",
                "class": "learned_user_text_vectors",
                "reuse_mode": "regenerated",
                "reason": "User text vectors are aggregated from the freshly emitted review text vectors.",
            },
            {
                "path": "logic_vectors/review_abnormal_vectors.npy",
                "class": "learned_abnormal_vectors",
                "reuse_mode": "regenerated",
                "reason": "Abnormal vectors are the RouteV optimization target and must be regenerated.",
            },
            {
                "path": "logic_vectors/user_abnormal_vectors.npy",
                "class": "learned_user_abnormal_vectors",
                "reuse_mode": "regenerated",
                "reason": "User abnormal vectors are aggregated from newly generated review vectors.",
            },
            {
                "path": "review_scores_enriched.csv",
                "class": "review_scores",
                "reuse_mode": "regenerated",
                "reason": "p_fake_review, review_gate, and evidence_score come from the current fresh encoder.",
            },
            {
                "path": "edges/TextSim_edges.csv",
                "class": "text_vector_edges",
                "reuse_mode": "regenerated",
                "reason": "TextSim edges depend on regenerated user_text_vectors.",
            },
            {
                "path": "edges/CB_edges.csv",
                "class": "combined_behavior_text_edges",
                "reuse_mode": "regenerated",
                "reason": "CB depends on TextSim candidates and current user features.",
            },
            {
                "path": "edges/LogicAE_CB_edges.csv",
                "class": "abnormal_vector_edges",
                "reuse_mode": "regenerated",
                "reason": "LogicAE_CB depends on regenerated user_abnormal_vectors.",
            },
            {
                "path": "edges/TNSGuided_LogicAE_CB_edges.csv",
                "class": "tns_guided_abnormal_edges",
                "reuse_mode": "regenerated",
                "reason": "TNS-guided logic edges depend on current LogicAE/review-score context.",
            },
            {
                "path": "edges/UPU_edges.csv",
                "class": "static_behavior_edges",
                "reuse_mode": "strict_reusable",
                "reason": "UPU does not depend on RouteV learned vectors when D1 graph mode/top-k/data protocol are unchanged; the runner still rebuilds it into each output.",
            },
            {
                "path": "edges/UTU_edges.csv",
                "class": "static_behavior_edges",
                "reuse_mode": "strict_reusable",
                "reason": "UTU does not depend on RouteV learned vectors when D1 graph mode/top-k/data protocol are unchanged; the runner still rebuilds it into each output.",
            },
            {
                "path": "edges/USU_edges.csv",
                "class": "static_behavior_edges",
                "reuse_mode": "strict_reusable",
                "reason": "USU does not depend on RouteV learned vectors when D1 graph mode/top-k/data protocol are unchanged; the runner still rebuilds it into each output.",
            },
        ],
    }


def _route_baseline_control_artifact_reuse(config: dict[str, Any]) -> dict[str, Any]:
    baseline_reference = dict(config.get("baseline_reference") or {})
    return {
        "experiment_type": "route_baseline_pack_control",
        "policy": "V_control is constructed identically to the promoted route baseline pack: complete fresh retrain artifact plus D1 graph-stage output.",
        "source_pack": baseline_reference.get("route_baseline"),
        "items": [
            {
                "path": "artifact/",
                "class": "complete_fresh_retrain_artifact",
                "reuse_mode": "strict_reusable",
                "reason": "This is the promoted route baseline artifact copied as the no-change control reference.",
            },
            {
                "path": "d1_graph/",
                "class": "routeD_d1_graph_stage",
                "reuse_mode": "strict_reusable",
                "reason": "This is the D1 graph-stage result paired with the promoted route baseline artifact.",
            },
        ],
    }


def _strict_label_policy() -> dict[str, str]:
    return {
        "user_label_source": "prepared.user_df.user_label",
        "review_bce_label_source": "prepared.review_df.review_label",
        "vector_proxy_label_column": "user_label",
        "vector_regularizer_label_source": "batch.user_label",
        "forbidden_vector_label_source": "review_label.max_per_user",
    }


def _build_routev_dataloaders(
    review_df: pd.DataFrame,
    llm_feature_df: pd.DataFrame,
    abnormal_masks: np.ndarray,
    tokenizer: Any,
    max_seq_length: int,
    batch_size: int,
) -> tuple[dict[str, DataLoader], pd.DataFrame]:
    ordered_reviews = review_df.sort_values("review_node_id").reset_index(drop=True)
    ordered_features = llm_feature_df.sort_values("review_node_id").reset_index(drop=True)
    if ordered_reviews["review_node_id"].tolist() != ordered_features["review_node_id"].tolist():
        raise ValueError("Review frame and LLM feature frame are misaligned on review_node_id.")
    if len(abnormal_masks) != len(ordered_reviews):
        raise ValueError("Abnormal mask row count does not match ordered review count.")

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
    numeric_tensor = torch.tensor(
        ordered_features[numeric_feature_columns()].to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )
    label_tensor = torch.tensor(ordered_reviews["review_label"].to_numpy(dtype=np.float32), dtype=torch.float32)
    if "user_label" not in ordered_reviews.columns:
        raise ValueError("RouteV strict mode requires review_df to include explicit user_label.")
    user_label_tensor = torch.tensor(ordered_reviews["user_label"].to_numpy(dtype=np.float32), dtype=torch.float32)
    user_to_idx = {
        user_id: idx
        for idx, user_id in enumerate(sorted(ordered_reviews["user_id"].astype(str).unique().tolist()))
    }
    user_indices = ordered_reviews["user_id"].astype(str).map(user_to_idx).to_numpy(dtype=np.int64)

    dataloaders: dict[str, DataLoader] = {}
    for split_name in ["train", "val", "test"]:
        index_mask = ordered_reviews["split"].eq(split_name).to_numpy()
        tensor_mask = torch.tensor(index_mask, dtype=torch.bool)
        dataset = RouteVReviewDataset(
            review_ids=ordered_reviews.loc[index_mask, "review_node_id"].to_numpy(dtype=np.int64),
            user_indices=user_indices[index_mask],
            input_ids=input_ids[tensor_mask],
            attention_mask=attention_mask[tensor_mask],
            abnormal_mask=abnormal_mask[tensor_mask],
            numeric_features=numeric_tensor[tensor_mask],
            labels=label_tensor[tensor_mask],
            user_labels=user_label_tensor[tensor_mask],
        )
        dataloaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=0,
        )

    dataloaders["all"] = DataLoader(
        RouteVReviewDataset(
            review_ids=ordered_reviews["review_node_id"].to_numpy(dtype=np.int64),
            user_indices=user_indices,
            input_ids=input_ids,
            attention_mask=attention_mask,
            abnormal_mask=abnormal_mask,
            numeric_features=numeric_tensor,
            labels=label_tensor,
            user_labels=user_label_tensor,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return dataloaders, ordered_reviews


def _prepare_context(
    exp_dir: Path,
    shared: dict[str, Any],
    *,
    smoke_test: bool,
    dual_head: bool,
    smoke_max_users: int,
) -> RouteVContext:
    run_cfg = dict(shared)
    if smoke_test:
        run_cfg["num_epochs"] = 1
        run_cfg["patience"] = 1
        run_cfg["batch_size"] = min(int(run_cfg.get("batch_size", 16)), 8)
        run_cfg["smoke_max_users"] = smoke_max_users
        run_cfg["debug_use_empty_mask"] = not dual_head
        if not dual_head:
            run_cfg["review_encoder"] = "mock"
    else:
        run_cfg["smoke_max_users"] = 0
        run_cfg["debug_use_empty_mask"] = False

    prepared = prepare_graph_data(
        graph_data_dir=run_cfg["graph_data_dir"],
        output_dir=exp_dir / "prepared_data",
        data_path=run_cfg.get("data_path"),
        seed=int(run_cfg.get("seed", 42)),
        train_ratio=float(run_cfg.get("train_ratio", 0.64)),
        val_ratio=float(run_cfg.get("val_ratio", 0.16)),
        test_ratio=float(run_cfg.get("test_ratio", 0.20)),
        min_user_reviews=int(run_cfg.get("min_user_reviews", 3)),
        min_product_reviews=int(run_cfg.get("min_product_reviews", 3)),
        prefer_corrected_reviews=bool(run_cfg.get("prefer_corrected_reviews", True)),
        overwrite_combined_files=bool(run_cfg.get("overwrite_combined_files", False)),
        smoke_max_users=int(run_cfg.get("smoke_max_users", 0)),
        balance_user_labels=bool(run_cfg.get("balance_user_labels", True)),
        balanced_user_count=int(run_cfg.get("balanced_user_count", 6742)),
    )
    prepared.review_df = _attach_user_labels_to_reviews(prepared.review_df, prepared.user_df)
    tokenizer = build_tokenizer(
        review_encoder=run_cfg.get("review_encoder", "llm_masked_logic"),
        primary_model_name_or_path=run_cfg["primary_model_name_or_path"],
        max_seq_length=int(run_cfg.get("max_seq_length", 256)),
    )
    llm_feature_df, abnormal_masks = build_llm_features_and_masks(
        review_df=prepared.review_df,
        tokenizer=tokenizer,
        llm_jsonl_path=run_cfg.get("llm_jsonl_path"),
        output_dir=ensure_dir(exp_dir / "llm_mask"),
        max_seq_length=int(run_cfg.get("max_seq_length", 256)),
        debug_use_empty_mask=bool(run_cfg.get("debug_use_empty_mask", False)),
        mask_source=run_cfg.get("mask_source", "full_text"),
    )
    dataloaders, ordered_reviews = _build_routev_dataloaders(
        review_df=prepared.review_df,
        llm_feature_df=llm_feature_df,
        abnormal_masks=abnormal_masks,
        tokenizer=tokenizer,
        max_seq_length=int(run_cfg.get("max_seq_length", 256)),
        batch_size=int(run_cfg.get("batch_size", 16)),
    )
    return RouteVContext(
        prepared=prepared,
        llm_feature_df=llm_feature_df,
        abnormal_masks=abnormal_masks,
        tokenizer=tokenizer,
        dataloaders=dataloaders,
        ordered_reviews=ordered_reviews,
        config=run_cfg,
    )


def _build_model_for_variant(shared: dict[str, Any], variant: dict[str, Any], *, smoke_test: bool) -> nn.Module:
    review_encoder_name = shared.get("review_encoder", "llm_masked_logic")
    if smoke_test and not bool(variant.get("dual_head", False)):
        review_encoder_name = "mock"
    base_model = build_review_model(
        review_encoder=review_encoder_name,
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
    if not bool(variant.get("dual_head", False)):
        return base_model
    return DualHeadWrapper(
        base_encoder=base_model,
        vector_dim=int(shared.get("vector_dim", 256)),
        graph_hidden_dim=variant.get("graph_hidden_dim"),
        dropout=float(variant.get("graph_head_dropout", 0.1)),
        detach_fusion=bool(variant.get("detach_fusion", True)),
    )


def _make_reg_loss(variant: dict[str, Any]) -> nn.Module | None:
    reg_lambda = _variant_reg_lambda(variant)
    if reg_lambda <= 0:
        return None
    reg_type = str(variant.get("vector_reg_type", "supcon")).lower()
    if reg_type == "triplet":
        return TripletUserVectorLoss(margin=float(variant.get("vector_reg_margin", 0.3)))
    return UserVectorSeparabilityLoss(temperature=float(variant.get("vector_reg_temperature", 0.1)))


def _variant_reg_lambda(variant: dict[str, Any]) -> float:
    if bool(variant.get("dual_head", False)):
        return float(variant.get("dual_head_lambda", variant.get("vector_reg_lambda", 0.0)))
    return float(variant.get("vector_reg_lambda", 0.0))


def _run_epoch_routev(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.Module,
    *,
    reg_loss_fn: nn.Module | None,
    reg_lambda: float,
    reg_vector_field: str,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    losses: list[float] = []
    reg_losses: list[float] = []
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
        user_labels = batch["user_label"].to(device)
        user_indices = batch["user_id_idx"].to(device)

        with torch.set_grad_enabled(training):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                abnormal_token_mask=abnormal_mask,
                numeric_features=numeric_features,
            )
            bce_loss = criterion(outputs.review_logit, labels)
            loss = bce_loss
            reg_loss_value = torch.tensor(0.0, device=device)
            if training and reg_loss_fn is not None and reg_lambda > 0.0:
                reg_vectors = getattr(outputs, reg_vector_field, outputs.review_vector)
                reg_loss_value = reg_loss_fn(reg_vectors, user_indices, user_labels)
                loss = loss + float(reg_lambda) * reg_loss_value
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        losses.append(float(loss.detach().cpu()))
        reg_losses.append(float(reg_loss_value.detach().cpu()))
        all_probs.append(torch.sigmoid(outputs.review_logit).detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    return (
        float(np.mean(losses) if losses else 0.0),
        float(np.mean(reg_losses) if reg_losses else 0.0),
        np.concatenate(all_labels) if all_labels else np.asarray([], dtype=np.float32),
        np.concatenate(all_probs) if all_probs else np.asarray([], dtype=np.float32),
    )


def _train_review_encoder_routev(
    model: nn.Module,
    dataloaders: dict[str, DataLoader],
    output_dir: Path,
    device: torch.device,
    *,
    learning_rate: float,
    num_epochs: int,
    patience: int,
    checkpoint_selection: str,
    reg_loss_fn: nn.Module | None,
    reg_lambda: float,
    reg_vector_field: str,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)

    train_label_batches = [batch["label"].numpy() for batch in dataloaders["train"]]
    if not train_label_batches:
        raise ValueError("Train split is empty; cannot train RouteV review encoder.")
    train_labels = np.concatenate(train_label_batches)
    positive_count = float(train_labels.sum())
    negative_count = float(len(train_labels) - positive_count)
    pos_weight = negative_count / max(positive_count, 1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(learning_rate),
    )

    model.to(device)
    best_val_auc = float("-inf")
    best_payload: dict[str, Any] | None = None
    best_epoch_path: Path | None = None
    bad_epochs = 0
    history: list[dict[str, Any]] = []
    select_by_review_auc = checkpoint_selection == "review_val_auc"

    for epoch_index in range(int(num_epochs)):
        train_loss, train_reg_loss, train_y, train_prob = _run_epoch_routev(
            model=model,
            dataloader=dataloaders["train"],
            device=device,
            optimizer=optimizer,
            criterion=criterion,
            reg_loss_fn=reg_loss_fn,
            reg_lambda=reg_lambda,
            reg_vector_field=reg_vector_field,
        )
        val_loss, val_reg_loss, val_y, val_prob = _run_epoch_routev(
            model=model,
            dataloader=dataloaders["val"],
            device=device,
            optimizer=None,
            criterion=criterion,
            reg_loss_fn=None,
            reg_lambda=0.0,
            reg_vector_field=reg_vector_field,
        )
        train_metrics = compute_binary_metrics(train_y, train_prob)
        val_metrics = compute_binary_metrics(val_y, val_prob)
        epoch = epoch_index + 1
        checkpoint_path = output_dir / f"review_encoder_epoch{epoch:02d}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        epoch_payload = {
            "epoch": epoch,
            "checkpoint_path": str(checkpoint_path),
            "train_loss": train_loss,
            "train_reg_loss": train_reg_loss,
            "val_loss": val_loss,
            "val_reg_loss": val_reg_loss,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "checkpoint_selection": checkpoint_selection,
            "vector_reg_lambda": float(reg_lambda),
            "reg_vector_field": reg_vector_field,
        }
        _save_json(output_dir / f"review_encoder_epoch{epoch:02d}.metrics.json", epoch_payload)
        history.append(epoch_payload)
        print(
            f"    epoch {epoch:02d}: val_auc={val_metrics['auc']:.6f} "
            f"val_ap={val_metrics['ap']:.6f} train_reg={train_reg_loss:.5f}",
            flush=True,
        )

        if val_metrics["auc"] >= best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_epoch_path = checkpoint_path
            best_payload = epoch_payload
            bad_epochs = 0
        else:
            bad_epochs += 1
            if select_by_review_auc and bad_epochs >= int(patience):
                break

    _save_json(output_dir / "review_encoder_training_history.json", history)
    if best_epoch_path is None or best_payload is None:
        raise RuntimeError("No review checkpoints were produced.")
    shutil.copyfile(best_epoch_path, output_dir / "best_review_encoder.pt")
    _save_json(output_dir / "review_encoder_metrics.json", best_payload)
    return history


def _encode_reviews_routev(
    model: nn.Module,
    dataloader: DataLoader,
    review_df: pd.DataFrame,
    checkpoint_path: Path,
    metrics_path: Path,
    device: torch.device,
    *,
    use_graph_vector: bool,
) -> RouteVEncodingArtifacts:
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    rows: list[dict[str, Any]] = []
    review_vectors: list[np.ndarray] = []
    text_vectors: list[np.ndarray] = []
    review_lookup = review_df.set_index("review_node_id")
    if "user_label" not in review_lookup.columns:
        raise ValueError("RouteV encoding requires explicit user_label in review_df.")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Encoding reviews", leave=False):
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                abnormal_token_mask=batch["abnormal_mask"].to(device),
                numeric_features=batch["numeric_features"].to(device),
            )
            probs = torch.sigmoid(outputs.review_logit).detach().cpu().numpy()
            gates = outputs.gate.detach().cpu().numpy()
            vector_tensor = getattr(outputs, "graph_vector", outputs.review_vector) if use_graph_vector else outputs.review_vector
            review_vec = vector_tensor.detach().cpu().numpy()
            text_vec = outputs.text_vector.detach().cpu().numpy()
            review_ids = batch["review_id"].cpu().numpy()
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
                        "user_label": int(row["user_label"]),
                        "split": str(row["split"]),
                        "p_fake_review": float(probs[index]),
                        "review_gate": float(gates[index]),
                    }
                )
                review_vectors.append(review_vec[index])
                text_vectors.append(text_vec[index])

    review_output_df = pd.DataFrame(rows)
    order = np.argsort(review_output_df["review_node_id"].to_numpy(dtype=np.int64))
    review_output_df = review_output_df.iloc[order].reset_index(drop=True)
    review_vectors_np = np.asarray(review_vectors, dtype=np.float32)[order]
    text_vectors_np = np.asarray(text_vectors, dtype=np.float32)[order]
    return RouteVEncodingArtifacts(
        review_output_df=review_output_df,
        review_vectors=review_vectors_np,
        text_vectors=text_vectors_np,
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
    )


def _checkpoint_epoch(path: Path) -> int:
    stem = path.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits[-2:] or 0)


def _select_checkpoint_by_user_proxy(
    model: nn.Module,
    context: RouteVContext,
    review_encoder_dir: Path,
    device: torch.device,
    *,
    use_graph_vector: bool,
) -> dict[str, Any]:
    checkpoint_paths = sorted(review_encoder_dir.glob("review_encoder_epoch*.pt"))
    if not checkpoint_paths:
        raise RuntimeError(f"No epoch checkpoints found in {review_encoder_dir}")

    proxy_dir = ensure_dir(review_encoder_dir / "checkpoint_proxy")
    all_proxy_rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None

    print(f"    selecting among {len(checkpoint_paths)} checkpoints by train->val user-vector proxy", flush=True)
    for checkpoint_path in checkpoint_paths:
        epoch = _checkpoint_epoch(checkpoint_path)
        metrics_path = review_encoder_dir / f"review_encoder_epoch{epoch:02d}.metrics.json"
        artifacts = _encode_reviews_routev(
            model=model,
            dataloader=context.dataloaders["all"],
            review_df=context.prepared.review_df,
            checkpoint_path=checkpoint_path,
            metrics_path=metrics_path,
            device=device,
            use_graph_vector=use_graph_vector,
        )
        epoch_proxy_dir = ensure_dir(proxy_dir / f"epoch_{epoch:02d}")
        artifacts.review_output_df.to_csv(epoch_proxy_dir / "review_output.csv", index=False)
        np.save(epoch_proxy_dir / "review_vectors.npy", artifacts.review_vectors)
        np.save(epoch_proxy_dir / "text_vectors.npy", artifacts.text_vectors)
        proxy_metrics = compute_user_vector_proxy_train_eval(
            artifacts.review_vectors,
            artifacts.review_output_df,
            train_split="train",
            eval_split="val",
            top_m=int(context.config.get("top_m", 3)),
            label_column="user_label",
        )
        row = {
            "epoch": epoch,
            "checkpoint_path": str(checkpoint_path),
            "cache_dir": str(epoch_proxy_dir),
            **proxy_metrics,
        }
        all_proxy_rows.append(row)
        print(
            f"      epoch {epoch:02d}: user_auc={row.get('user_auc', 0.0):.6f} "
            f"user_ap={row.get('user_ap', 0.0):.6f}",
            flush=True,
        )
        if best_row is None:
            best_row = row
        else:
            current_key = (float(row.get("user_auc", 0.0)), float(row.get("user_ap", 0.0)))
            best_key = (float(best_row.get("user_auc", 0.0)), float(best_row.get("user_ap", 0.0)))
            if current_key > best_key:
                best_row = row

    assert best_row is not None
    selection = {
        "selection_metric": "train_to_val_user_vector_proxy",
        "best_checkpoint": best_row,
        "all_checkpoints": all_proxy_rows,
        "use_graph_vector": bool(use_graph_vector),
    }
    _save_json(review_encoder_dir / "routeV_proxy_selection.json", selection)
    return selection


def _load_cached_artifacts(selection: dict[str, Any], checkpoint_path: Path, metrics_path: Path) -> RouteVEncodingArtifacts | None:
    best = selection.get("best_checkpoint") if selection else None
    if not best:
        return None
    cache_dir = Path(best.get("cache_dir", ""))
    review_output_path = cache_dir / "review_output.csv"
    review_vectors_path = cache_dir / "review_vectors.npy"
    text_vectors_path = cache_dir / "text_vectors.npy"
    if not (review_output_path.exists() and review_vectors_path.exists() and text_vectors_path.exists()):
        return None
    return RouteVEncodingArtifacts(
        review_output_df=pd.read_csv(review_output_path),
        review_vectors=np.load(review_vectors_path),
        text_vectors=np.load(text_vectors_path),
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
    )


def _finalize_graph_stage(
    exp_dir: Path,
    variant: dict[str, Any],
    shared: dict[str, Any],
    config: dict[str, Any],
    context: RouteVContext,
    model: nn.Module,
    device: torch.device,
    *,
    selected_checkpoint: Path,
    selected_metrics_path: Path,
    selection: dict[str, Any] | None,
    use_graph_vector: bool,
    smoke_test: bool,
) -> dict[str, Any]:
    review_encoder_dir = ensure_dir(exp_dir / "review_encoder")
    best_path = review_encoder_dir / "best_review_encoder.pt"
    shutil.copyfile(selected_checkpoint, best_path)

    selected_metrics = _load_json(selected_metrics_path) if selected_metrics_path.exists() else {}
    if selection:
        selected_metrics["routeV_proxy_selection"] = selection.get("best_checkpoint")
    _save_json(review_encoder_dir / "review_encoder_metrics.json", selected_metrics)

    artifacts = _load_cached_artifacts(selection or {}, best_path, review_encoder_dir / "review_encoder_metrics.json")
    if artifacts is None:
        artifacts = _encode_reviews_routev(
            model=model,
            dataloader=context.dataloaders["all"],
            review_df=context.prepared.review_df,
            checkpoint_path=best_path,
            metrics_path=review_encoder_dir / "review_encoder_metrics.json",
            device=device,
            use_graph_vector=use_graph_vector,
        )
    artifacts.review_output_df.to_csv(review_encoder_dir / "review_output.csv", index=False)
    np.save(review_encoder_dir / "selected_review_vectors.npy", artifacts.review_vectors)
    np.save(review_encoder_dir / "selected_text_vectors.npy", artifacts.text_vectors)

    review_scores_df, user_df, _, user_abnormal_vectors, user_text_vectors = build_review_and_user_artifacts(
        review_df=context.prepared.review_df.sort_values("review_node_id").reset_index(drop=True),
        llm_feature_df=context.llm_feature_df,
        review_output_df=artifacts.review_output_df,
        review_vectors=artifacts.review_vectors,
        text_vectors=artifacts.text_vectors,
        output_dir=exp_dir,
        top_m=int(shared.get("top_m", 3)),
        time_bucket=shared.get("time_bucket", "week"),
    )
    user_df = _align_user_artifacts_to_prepared_labels(user_df, context.prepared.user_df)
    user_df.to_csv(exp_dir / "logic_vectors" / "user_summary.csv", index=False)
    edge_frames = build_edge_frames(
        user_df=user_df,
        user_text_vectors=user_text_vectors,
        user_abnormal_vectors=user_abnormal_vectors,
        output_dir=exp_dir,
        top_k=int(shared.get("top_k", 20)),
        review_features=review_scores_df,
        logic_threshold_mode=shared.get("logic_threshold_mode", "quantile"),
        logic_threshold_quantile=float(shared.get("logic_threshold_quantile", 0.60)),
        logic_threshold_value=float(shared.get("logic_threshold_value", 0.30)),
        graph_mode=shared.get("graph_mode", "current"),
        senior_usu_ratio=float(shared.get("senior_usu_ratio", 0.10)),
        use_tns_guided_logic=bool(shared.get("use_tns_guided_logic", False)),
        tns_phi_days=int(shared.get("tns_phi_days", 5)),
        tns_logic_mode=shared.get("tns_logic_mode", "boost"),
        tns_logic_lambda=float(shared.get("tns_logic_lambda", 1.0)),
        logic_tns_topk=int(shared.get("logic_tns_topk", 20)),
    )
    edge_stats_df = compute_edge_stats(edge_frames=edge_frames, user_df=user_df, output_dir=exp_dir)
    self_features = build_self_feature_matrix(user_df, user_abnormal_vectors)
    metrics_dir = ensure_dir(exp_dir / "metrics")
    model_results_df = run_relation_aggregation_experiments(
        user_df=user_df,
        self_features=self_features,
        edge_frames=edge_frames,
        output_dir=metrics_dir,
        review_encoder_name=shared.get("review_encoder", "llm_masked_logic"),
        model_kind=shared.get("relation_model", "edge_aware_gat"),
        seed=int(shared.get("seed", 42)),
        backbone=shared.get("model_backbone", "current_egat"),
        relation_model=shared.get("relation_model", "edge_aware_gat"),
        use_abnormal_edge_weight=bool(shared.get("use_abnormal_edge_weight", False)),
        use_abnormal_gate=bool(shared.get("use_abnormal_gate", False)),
        use_abnormal_value_gate=bool(shared.get("use_abnormal_value_gate", False)),
        use_abnormal_attention_bias=bool(shared.get("use_abnormal_attention_bias", False)),
        abnormal_score_source=shared.get("abnormal_score_source", "auto"),
        abnormal_edge_lambda=float(shared.get("abnormal_edge_lambda", 1.0)),
        abnormal_edge_eta=float(shared.get("abnormal_edge_eta", 0.5)),
        abnormal_gate_eta=float(shared.get("abnormal_gate_eta", 0.5)),
        abnormal_pair_mode=shared.get("abnormal_pair_mode", "both_high"),
        abnormal_gate_learnable=bool(shared.get("abnormal_gate_learnable", False)),
        abnormal_attention_gamma=float(shared.get("abnormal_attention_gamma", 1.0)),
        review_scores_df=review_scores_df,
        selected_edge_set=shared.get("edge_set", "Base_LogicAE_CB"),
        use_node_gat=bool(shared.get("use_node_gat", False)),
        max_epochs_override=6 if smoke_test else None,
        patience_override=2 if smoke_test else None,
    )

    review_scores_df.to_csv(exp_dir / "review_scores_enriched.csv", index=False)
    user_df.to_csv(exp_dir / "user_scores_enriched.csv", index=False)
    baseline_metadata = _routev_baseline_metadata(config, variant)
    run_config = dict(shared)
    run_config.update(
        {
            "variant": variant,
            "baseline_metadata": baseline_metadata,
            "experiment_type": "fresh_d1_train",
            "strict_label_policy": _strict_label_policy(),
            "artifact_reuse": _artifact_reuse_manifest(),
            "routeV_use_graph_vector": bool(use_graph_vector),
            "selected_checkpoint": str(selected_checkpoint),
            "selected_metrics_path": str(selected_metrics_path),
            "smoke_test": bool(smoke_test),
            "resolved_device": str(device),
        }
    )
    _save_json(exp_dir / "run_config.json", run_config)

    target_edge = shared.get("edge_set", "Base_LogicAE_CB")
    target_rows = model_results_df[model_results_df.get("edge_set", pd.Series(dtype=str)).eq(target_edge)]
    if target_rows.empty:
        target_rows = model_results_df
    best_graph_model = target_rows.sort_values("auc", ascending=False).iloc[0].to_dict() if not target_rows.empty else None
    summary = {
        "status": "ok",
        "variant": variant.get("name"),
        "output_dir": str(exp_dir),
        "review_count": int(len(context.prepared.review_df)),
        "user_count": int(len(user_df)),
        "selected_checkpoint": str(selected_checkpoint),
        "checkpoint_selection": variant.get("checkpoint_selection"),
        "experiment_type": "fresh_d1_train",
        "strict_label_policy": run_config["strict_label_policy"],
        "artifact_reuse": run_config["artifact_reuse"],
        "use_graph_vector": bool(use_graph_vector),
        "baseline_metadata": baseline_metadata,
        "proxy_selection": selection,
        "selected_review_metrics": selected_metrics,
        "best_graph_model": best_graph_model,
        "best_edge_type_by_fake_fake_ratio": edge_stats_df.sort_values("fake_fake_ratio", ascending=False).iloc[0].to_dict()
        if not edge_stats_df.empty
        else None,
    }
    _save_json(exp_dir / "run_summary.json", summary)
    return summary


def _run_variant(
    output_root: Path,
    config: dict[str, Any],
    variant: dict[str, Any],
    *,
    smoke_test: bool,
    smoke_max_users: int,
) -> dict[str, Any]:
    name = variant["name"]
    exp_dir = output_root / name
    exp_dir.mkdir(parents=True, exist_ok=True)

    shared = _shared_defaults(config)
    if smoke_test and bool(variant.get("dual_head", False)):
        print("    [smoke] V2 uses the real encoder because DualHeadWrapper needs D1 internals", flush=True)
    seed_everything(int(shared.get("seed", 42)))
    device = resolve_device(shared.get("device", "auto"))

    context = _prepare_context(
        exp_dir,
        shared,
        smoke_test=smoke_test,
        dual_head=bool(variant.get("dual_head", False)),
        smoke_max_users=smoke_max_users,
    )
    model = _build_model_for_variant(context.config, variant, smoke_test=smoke_test)
    reg_loss_fn = _make_reg_loss(variant)
    reg_lambda = _variant_reg_lambda(variant)
    use_graph_vector = bool(variant.get("dual_head", False))
    reg_vector_field = "graph_vector" if use_graph_vector else "review_vector"
    review_encoder_dir = ensure_dir(exp_dir / "review_encoder")
    checkpoint_selection = str(variant.get("checkpoint_selection", "review_val_auc"))

    _save_json(
        exp_dir / "routeV_variant_config.json",
        {
            "experiment_type": "fresh_d1_train",
            "variant": variant,
            "shared": context.config,
            "baseline_metadata": _routev_baseline_metadata(config, variant),
            "strict_label_policy": _strict_label_policy(),
            "artifact_reuse": _artifact_reuse_manifest(),
        },
    )
    print(
        f"    training {name}: selection={checkpoint_selection} reg_lambda={reg_lambda} "
        f"reg_field={reg_vector_field}",
        flush=True,
    )
    history = _train_review_encoder_routev(
        model=model,
        dataloaders=context.dataloaders,
        output_dir=review_encoder_dir,
        device=device,
        learning_rate=float(context.config.get("learning_rate", 2e-5)),
        num_epochs=int(context.config.get("num_epochs", 3)),
        patience=int(context.config.get("patience", 2)),
        checkpoint_selection=checkpoint_selection,
        reg_loss_fn=reg_loss_fn,
        reg_lambda=reg_lambda,
        reg_vector_field=reg_vector_field,
    )

    selection = None
    if checkpoint_selection == "user_vector_proxy":
        selection = _select_checkpoint_by_user_proxy(
            model=model,
            context=context,
            review_encoder_dir=review_encoder_dir,
            device=device,
            use_graph_vector=use_graph_vector,
        )
        selected_checkpoint = Path(selection["best_checkpoint"]["checkpoint_path"])
        selected_metrics_path = review_encoder_dir / f"review_encoder_epoch{int(selection['best_checkpoint']['epoch']):02d}.metrics.json"
    else:
        selected_payload = _load_json(review_encoder_dir / "review_encoder_metrics.json")
        selected_checkpoint = Path(selected_payload.get("checkpoint_path", review_encoder_dir / "best_review_encoder.pt"))
        selected_metrics_path = Path(selected_checkpoint).with_suffix(".metrics.json")
        if not selected_metrics_path.exists():
            selected_metrics_path = review_encoder_dir / "review_encoder_metrics.json"

    summary = _finalize_graph_stage(
        exp_dir=exp_dir,
        variant=variant,
        shared=context.config,
        config=config,
        context=context,
        model=model,
        device=device,
        selected_checkpoint=selected_checkpoint,
        selected_metrics_path=selected_metrics_path,
        selection=selection,
        use_graph_vector=use_graph_vector,
        smoke_test=smoke_test,
    )
    summary["training_history"] = history
    _save_json(exp_dir / "run_summary.json", summary)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def _run_route_baseline_pack_control(
    output_root: Path,
    config: dict[str, Any],
    variant: dict[str, Any],
) -> dict[str, Any]:
    name = variant["name"]
    exp_dir = output_root / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    baseline_reference = dict(config.get("baseline_reference") or {})
    source_pack = Path(str(baseline_reference.get("route_baseline", "")))
    if not source_pack.is_absolute():
        source_pack = PROJECT_ROOT / source_pack
    source_pack = source_pack.resolve()
    source_artifact = source_pack / "artifact"
    source_d1_graph = source_pack / "d1_graph"
    target_artifact = exp_dir / "artifact"
    target_d1_graph = exp_dir / "d1_graph"

    _copytree_no_overwrite(source_artifact, target_artifact)
    _copytree_no_overwrite(source_d1_graph, target_d1_graph)
    for filename in [
        "baseline_manifest.json",
        "baseline_metric_summary.json",
        "baseline_metric_summary.csv",
        "baseline_metric_summary.md",
        "baseline_metric_comparison.json",
        "baseline_metric_comparison.csv",
        "baseline_metric_comparison.md",
    ]:
        source_file = source_pack / filename
        if source_file.exists():
            shutil.copy2(source_file, exp_dir / filename)

    shared = _shared_defaults(config)
    target_edge = str(shared.get("edge_set", "Base_LogicAE_CB"))
    target_model = str(shared.get("relation_model", "edge_aware_gat"))
    if target_model == "edge_aware_gat":
        target_model = "current_egat_edge_aware_gat"
    target_row = _read_csv_target_row(
        target_d1_graph / "metrics" / "model_results.csv",
        edge_set=target_edge,
        model_name=target_model,
    )
    if not target_row:
        raise RuntimeError(f"Could not read RouteV V_control target row from {target_d1_graph / 'metrics/model_results.csv'}")

    review_metrics_path = target_artifact / "review_encoder" / "review_encoder_metrics.json"
    review_metrics = _load_json(review_metrics_path) if review_metrics_path.exists() else {}
    manifest_path = exp_dir / "baseline_manifest.json"
    manifest = _load_json(manifest_path) if manifest_path.exists() else {}
    baseline_metadata = _routev_baseline_metadata(config, variant)
    run_config = {
        **shared,
        "variant": variant,
        "control_construction": "route_baseline_pack",
        "source_route_baseline_pack": str(source_pack),
        "source_artifact": str(source_artifact),
        "source_d1_graph": str(source_d1_graph),
        "baseline_metadata": baseline_metadata,
        "experiment_type": "route_baseline_pack_control",
        "strict_label_policy": _strict_label_policy(),
        "artifact_reuse": _route_baseline_control_artifact_reuse(config),
        "smoke_test": False,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    summary = {
        "status": "ok",
        "variant": name,
        "output_dir": str(exp_dir),
        "control_construction": "route_baseline_pack",
        "source_route_baseline_pack": str(source_pack),
        "artifact_dir": str(target_artifact),
        "d1_graph_dir": str(target_d1_graph),
        "experiment_type": "route_baseline_pack_control",
        "strict_label_policy": run_config["strict_label_policy"],
        "artifact_reuse": run_config["artifact_reuse"],
        "baseline_metadata": baseline_metadata,
        "baseline_manifest": manifest,
        "selected_checkpoint": str(target_artifact / "review_encoder" / "best_review_encoder.pt"),
        "checkpoint_selection": variant.get("checkpoint_selection"),
        "selected_review_metrics": review_metrics,
        "best_graph_model": target_row,
    }
    _save_json(exp_dir / "routeV_variant_config.json", run_config)
    _save_json(exp_dir / "run_config.json", run_config)
    _save_json(exp_dir / "run_summary.json", summary)
    return summary


def _load_graph_metrics(exp_dir: Path, edge_set: str = "Base_LogicAE_CB") -> dict[str, float]:
    csv_path = exp_dir / "metrics" / "model_results.csv"
    if not csv_path.exists():
        csv_path = exp_dir / "d1_graph" / "metrics" / "model_results.csv"
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    target = df[df.get("edge_set", pd.Series(dtype=str)).eq(edge_set)]
    if target.empty:
        target = df
    if target.empty:
        return {}
    row = target.sort_values("auc", ascending=False).iloc[0]
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
    shared = _shared_defaults(config)
    d1_floor_auc = float(args.d1_floor_auc if args.d1_floor_auc is not None else config.get("d1_floor_auc", 0.840))
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    status_file = PROJECT_ROOT / "graph" / "logs" / "status" / f"routeV_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    print("=" * 72, flush=True)
    print("RouteV Vector Quality Queue", flush=True)
    print(f"Output: {output_root}", flush=True)
    print(f"Config: {args.config_path}", flush=True)
    print(f"Smoke: {args.smoke_test}", flush=True)
    print("=" * 72, flush=True)

    available_variants = config["variants"]
    requested_names = set(args.variants or [variant["name"] for variant in available_variants])
    selected_variants = [variant for variant in available_variants if variant["name"] in requested_names]
    if not selected_variants:
        raise ValueError(f"No requested RouteV variants matched: {sorted(requested_names)}")

    results: list[dict[str, Any]] = []
    if not args.skip_control and "V_control" not in {variant["name"] for variant in selected_variants}:
        control = next((variant for variant in available_variants if variant["name"] == "V_control"), None)
        if control is not None:
            selected_variants = [control] + selected_variants

    for variant in selected_variants:
        name = variant["name"]
        exp_dir = output_root / name
        print(f"\n[RouteV] Running {name}", flush=True)
        if args.resume and (exp_dir / "run_summary.json").exists():
            print(f"    [resume] using existing {exp_dir / 'run_summary.json'}", flush=True)
            graph_metrics = _load_graph_metrics(exp_dir, edge_set=shared.get("edge_set", "Base_LogicAE_CB"))
            results.append({"name": name, "status": "resumed", **graph_metrics})
        else:
            try:
                if (
                    bool(variant.get("is_control"))
                    and variant.get("control_construction") == "route_baseline_pack"
                    and not args.smoke_test
                ):
                    _run_route_baseline_pack_control(
                        output_root=output_root,
                        config=config,
                        variant=variant,
                    )
                else:
                    _run_variant(
                        output_root=output_root,
                        config=config,
                        variant=variant,
                        smoke_test=bool(args.smoke_test),
                        smoke_max_users=int(args.smoke_max_users),
                    )
                graph_metrics = _load_graph_metrics(exp_dir, edge_set=shared.get("edge_set", "Base_LogicAE_CB"))
                result = {"name": name, "status": "ok", **graph_metrics}
                results.append(result)
            except Exception as exc:
                failure = {"name": name, "status": "failed", "error": repr(exc)}
                results.append(failure)
                _save_json(output_root / "routeV_summary.json", {"status": "failed", "results": results})
                _save_json(status_file, {"status": "failed", "failed_variant": name, "results": results})
                raise

        if name == "V_control" and not args.smoke_test:
            control_auc = float(results[-1].get("graph_auc", 0.0))
            print(f"    V_control graph AUC={control_auc:.6f}; floor={d1_floor_auc:.6f}", flush=True)
            if control_auc < d1_floor_auc:
                _save_json(output_root / "routeV_summary.json", {
                    "status": "stopped_below_floor",
                    "route_contract": config.get("route_contract"),
                    "baseline_reference": config.get("baseline_reference"),
                    "baseline_variant": config.get("route_contract", {}).get("baseline_variant", "V_control"),
                    "control_auc": control_auc,
                    "floor": d1_floor_auc,
                    "results": results,
                })
                raise SystemExit("V_control below D1 floor; stop before variants.")

    summary = {
        "status": "complete",
        "output_root": str(output_root),
        "config_path": str(args.config_path),
        "smoke_test": bool(args.smoke_test),
        "route_contract": config.get("route_contract"),
        "baseline_reference": config.get("baseline_reference"),
        "baseline_variant": config.get("route_contract", {}).get("baseline_variant", "V_control"),
        "results": results,
    }
    _save_json(output_root / "routeV_summary.json", summary)
    _save_json(status_file, summary)
    print("\nRouteV summary:", flush=True)
    for result in results:
        print(
            f"  {result['name']}: {result['status']} "
            f"AUC={result.get('graph_auc', 'NA')} AP={result.get('graph_ap', 'NA')} F1={result.get('graph_f1', 'NA')}",
            flush=True,
        )
    print(f"Summary: {output_root / 'routeV_summary.json'}", flush=True)


if __name__ == "__main__":
    main()

