from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from graph.review_training import build_review_dataloaders, build_tokenizer
from .review_training_routeL_text import (
    build_psycholinguistic_style_frame,
    build_routeL_text_model,
    compute_routeL_text_losses,
    load_routeL_review_frames,
)
from .routeL_utils import ensure_dir, load_d1_bundle, load_yaml_config


def _prepare_extra_feature_frame(cfg: dict[str, Any], review_df: pd.DataFrame, dataloaders: dict[str, Any], bundle: Any, device: torch.device) -> pd.DataFrame | None:
    experiment_kind = str(cfg.get("EXPERIMENT_KIND", ""))
    if experiment_kind == "exp4_psycholinguistic_style":
        return build_psycholinguistic_style_frame(review_df)
    if experiment_kind == "exp5_semantic_drift":
        # Dry-run only needs a numerically valid review-level feature frame to
        # verify shape / forward / loss plumbing. Avoid full semantic-drift
        # precomputation here.
        frame = review_df[["review_node_id", "product_id"]].copy()
        product_counts = frame["product_id"].astype(str).value_counts()
        frame["semantic_drift"] = frame["product_id"].astype(str).map(product_counts).astype(np.float32)
        values = frame["semantic_drift"].to_numpy(dtype=np.float32)
        if len(values) > 1:
            lo = float(np.min(values))
            hi = float(np.max(values))
            if hi > lo:
                values = (values - lo) / (hi - lo)
        frame["semantic_drift"] = values.astype(np.float32)
        return frame[["review_node_id", "semantic_drift"]].set_index("review_node_id", drop=False)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_paths", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_seq_length", type=int, default=128)
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    bundle = load_d1_bundle()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    review_df, llm_feature_df, abnormal_masks = load_routeL_review_frames(bundle.base_dir)

    sampled_ids: list[int] = []
    for split_name in ["train", "val", "test"]:
        sampled_ids.extend(
            review_df[review_df["split"] == split_name]
            .sort_values("review_node_id")
            .head(max(args.batch_size, 2))["review_node_id"]
            .astype(int)
            .tolist()
        )
    sampled_id_set = set(sampled_ids)
    sampled_review_df = review_df[review_df["review_node_id"].isin(sampled_id_set)].sort_values("review_node_id").reset_index(drop=True)
    sampled_llm_feature_df = llm_feature_df[llm_feature_df["review_node_id"].isin(sampled_id_set)].sort_values("review_node_id").reset_index(drop=True)
    full_order = review_df.sort_values("review_node_id")["review_node_id"].astype(int).tolist()
    mask_lookup = {review_id: abnormal_masks[idx] for idx, review_id in enumerate(full_order)}
    sampled_masks = np.stack([mask_lookup[int(review_id)] for review_id in sampled_review_df["review_node_id"].astype(int).tolist()]).astype(np.float32)

    tokenizer = build_tokenizer(
        "llm_masked_logic",
        bundle.run_config["primary_model_name_or_path"],
        max_seq_length=args.max_seq_length,
    )
    dataloaders = build_review_dataloaders(
        review_df=sampled_review_df,
        llm_feature_df=sampled_llm_feature_df,
        abnormal_masks=sampled_masks,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
    )

    batch = next(iter(dataloaders["train"]))
    labels = batch["label"].to(device)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    row_ids = batch["review_id"].detach().cpu().numpy().astype(np.int64)

    shape_rows: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []

    for cfg_path_str in args.config_paths:
        cfg_path = Path(cfg_path_str)
        cfg = load_yaml_config(cfg_path)
        extra_feature_frame = _prepare_extra_feature_frame(cfg, sampled_review_df, dataloaders, bundle, device)
        extra_feature_dim = 0
        extra_tensor = None
        if extra_feature_frame is not None:
            extra_feature_dim = len([c for c in extra_feature_frame.columns if c != "review_node_id"])
            sub = extra_feature_frame.loc[row_ids]
            feature_cols = [c for c in sub.columns if c not in {"review_node_id"}]
            extra_tensor = torch.tensor(sub[feature_cols].to_numpy(dtype=np.float32), device=device)

        model = build_routeL_text_model(
            primary_model_name_or_path=bundle.run_config["primary_model_name_or_path"],
            vector_dim=int(bundle.run_config.get("vector_dim", 256)),
            experiment_kind=str(cfg.get("EXPERIMENT_KIND")),
            extra_feature_dim=extra_feature_dim,
            topk_tokens=int(cfg.get("TOPK_TOKENS", 8)),
        )
        model.to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, extra_features=extra_tensor)
        losses = compute_routeL_text_losses(
            outputs=outputs,
            labels=labels,
            pos_weight_value=1.0,
            lambda_evidence=float(cfg.get("LAMBDA_EVIDENCE", 0.2)),
            lambda_sparse=float(cfg.get("LAMBDA_SPARSE", 0.0)),
        )
        shape_rows.append(
            {
                "config": cfg_path.name,
                "experiment_kind": str(cfg.get("EXPERIMENT_KIND")),
                "review_vector_shape": list(outputs.review_vector.shape),
                "review_logit_shape": list(outputs.review_logit.shape),
                "evidence_logit_shape": list(outputs.evidence_logit.shape),
                "text_vector_shape": list(outputs.text_vector.shape),
                "token_scores_shape": list(outputs.token_evidence_scores.shape),
                "extra_feature_dim": int(extra_feature_dim),
            }
        )
        loss_rows.append(
            {
                "config": cfg_path.name,
                "main_loss": float(losses["main_loss"].detach().cpu()),
                "evidence_loss": float(losses["evidence_loss"].detach().cpu()),
                "sparse_loss": float(losses["sparse_loss"].detach().cpu()),
                "total_loss": float(losses["total_loss"].detach().cpu()),
                "loss_finite": bool(torch.isfinite(losses["total_loss"]).item()),
            }
        )

    (output_dir / "dryrun_textheads_report.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "checked_configs": [Path(p).name for p in args.config_paths],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    pd.DataFrame(shape_rows).to_json(output_dir / "dryrun_textheads_shape_check.json", orient="records", indent=2)
    pd.DataFrame(loss_rows).to_json(output_dir / "dryrun_textheads_loss_check.json", orient="records", indent=2)


if __name__ == "__main__":
    main()
