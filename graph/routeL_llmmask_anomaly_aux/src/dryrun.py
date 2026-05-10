from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from graph.llm_utils import numeric_feature_columns
from graph.review_training import build_review_dataloaders, build_tokenizer

from .export_user_features_routeL import assert_aux_not_exported, export_review_feature_frame
from .review_training_routeL import (
    build_routeL_dataloaders,
    build_routeL_model,
    compute_pos_weight,
    compute_routeL_losses,
    load_routeL_review_frames,
)
from .routeL_utils import ensure_dir, json_dump, load_d1_bundle, load_yaml_config, project_root_from_here


def _load_model_from_d1(bundle, cfg):
    model = build_routeL_model(
        primary_model_name_or_path=bundle.run_config["primary_model_name_or_path"],
        numeric_feature_dim=len(numeric_feature_columns()),
        vector_dim=int(bundle.run_config.get("vector_dim", 256)),
        secondary_model_name_or_path=bundle.run_config.get("secondary_model_name_or_path"),
        freeze_primary=bool(bundle.run_config.get("freeze_primary", False)),
        freeze_secondary=bool(bundle.run_config.get("freeze_secondary", False)),
        fusion_mode=str(cfg.get("FUSION_MODE", "early")),
        use_anomaly_aux_loss=bool(int(cfg.get("USE_ANOMALY_AUX_LOSS", 0))),
        anomaly_warmup_ratio=float(cfg.get("ANOMALY_WARMUP_RATIO", 0.3)),
        lambda_aux=float(cfg.get("lambda_aux", 0.2)),
        model_variant=str(cfg.get("MODEL_VARIANT", "single_tower")),
    )
    state = torch.load(bundle.base_dir / "review_encoder/best_review_encoder.pt", map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def _first_batch(dataloaders):
    return next(iter(dataloaders["train"]))


def _run_variant(model, batch, lambda_aux, pos_weight_value, warmup_active=False):
    with torch.no_grad():
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            abnormal_token_mask=batch["abnormal_mask"],
            numeric_features=batch["numeric_features"],
            warmup_active=warmup_active,
        )
        losses = compute_routeL_losses(
            outputs,
            batch["label"],
            pos_weight_value=pos_weight_value,
            lambda_aux=lambda_aux,
            abnormal_mask=batch["abnormal_mask"],
            use_label_filtered_mask=bool(batch.get("use_label_filtered_mask", False)),
            lambda_mask_align=float(batch.get("lambda_mask_align", 0.0)),
        )
    return outputs, losses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=256)
    args = parser.parse_args()

    outdir = ensure_dir(args.output_dir)
    bundle = load_d1_bundle(project_root_from_here())
    config_files = sorted(Path(args.configs_dir).glob("*.yaml"))

    review_df, llm_feature_df, abnormal_masks = load_routeL_review_frames(bundle.base_dir)
    max_seq_length = int(abnormal_masks.shape[1]) if getattr(abnormal_masks, "ndim", 0) == 2 else int(args.max_seq_length)
    dataloaders = build_review_dataloaders(
        review_df=review_df,
        llm_feature_df=llm_feature_df,
        abnormal_masks=abnormal_masks,
        tokenizer=build_tokenizer("llm_masked_logic", bundle.run_config["primary_model_name_or_path"], max_seq_length),
        max_seq_length=max_seq_length,
        batch_size=args.batch_size,
    )
    batch = _first_batch(dataloaders)
    pos_weight_value = compute_pos_weight(dataloaders["train"], device=torch.device("cpu"))

    shape_rows = []
    loss_rows = []
    export_ok = []

    for cfg_path in config_files:
        cfg = load_yaml_config(cfg_path)
        model = _load_model_from_d1(bundle, cfg)
        warmup_active = bool(cfg.get("FUSION_MODE", "early").lower() == "late")
        batch["use_label_filtered_mask"] = bool(int(cfg.get("USE_LABEL_FILTERED_MASK", 0)))
        batch["lambda_mask_align"] = float(cfg.get("lambda_mask_align", 0.0))
        outputs, losses = _run_variant(
            model,
            batch,
            lambda_aux=float(cfg.get("lambda_aux", 0.0)),
            pos_weight_value=pos_weight_value,
            warmup_active=warmup_active,
        )
        shape_rows.append(
            {
                "config": cfg_path.name,
                "fusion_mode": cfg.get("FUSION_MODE"),
                "use_anomaly_aux_loss": int(cfg.get("USE_ANOMALY_AUX_LOSS", 0)),
                "input_ids_shape": list(batch["input_ids"].shape),
                "review_vector_shape": list(outputs.review_vector.shape),
                "review_logit_shape": list(outputs.review_logit.shape),
                "aux_logit_shape": list(outputs.aux_logit.shape),
                "text_vector_shape": list(outputs.text_vector.shape),
                "gate_shape": list(outputs.gate.shape),
                "mask_token_weights_shape": list(outputs.mask_token_weights.shape),
                "warmup_active": warmup_active,
                "aux_exported": False,
            }
        )
        loss_rows.append(
            {
                "config": cfg_path.name,
                "fusion_mode": cfg.get("FUSION_MODE"),
                "use_anomaly_aux_loss": int(cfg.get("USE_ANOMALY_AUX_LOSS", 0)),
                "main_loss": float(losses["main_loss"].detach().cpu()),
                "aux_loss": float(losses["aux_loss"].detach().cpu()),
                "mask_align_loss": float(losses["mask_align_loss"].detach().cpu()),
                "total_loss": float(losses["total_loss"].detach().cpu()),
                "main_loss_finite": bool(torch.isfinite(losses["main_loss"])),
                "aux_loss_finite": bool(torch.isfinite(losses["aux_loss"])),
                "mask_align_loss_finite": bool(torch.isfinite(losses["mask_align_loss"])),
                "total_loss_finite": bool(torch.isfinite(losses["total_loss"])),
            }
        )
        export_frame = export_review_feature_frame(
            pd.DataFrame(
                {
                    "review_node_id": batch["review_id"].numpy(),
                    "review_logit": outputs.review_logit.detach().cpu().numpy(),
                    "review_vector_norm": outputs.review_vector.norm(dim=1).detach().cpu().numpy(),
                    "text_vector_norm": outputs.text_vector.norm(dim=1).detach().cpu().numpy(),
                }
            ),
            outputs.review_vector.detach().cpu().numpy(),
            outputs.text_vector.detach().cpu().numpy(),
            outdir / cfg_path.stem,
        )
        export_ok.append(assert_aux_not_exported(export_frame))

    dryrun_report = {
        "status": "pass" if all(export_ok) else "fail",
        "num_configs": len(config_files),
        "aux_not_exported": bool(all(export_ok)),
        "d1_review_encoder": bundle.run_summary.get("best_graph_model", {}).get("review_encoder", "UNKNOWN_FROM_D1"),
        "d1_feature_dim": 288,
    }
    json_dump(outdir / "model_shape_check.json", {"rows": shape_rows})
    json_dump(outdir / "loss_check.json", {"rows": loss_rows})
    json_dump(outdir / "dryrun_report.json", dryrun_report)
    (outdir / "dryrun_report.md").write_text(
        "\n".join(
            [
                "# Route L Phase 1 Dry-run",
                "",
                f"Status: {'PASS' if dryrun_report['status'] == 'pass' else 'FAIL'}",
                f"Configs checked: {len(config_files)}",
                f"Aux not exported: {all(export_ok)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    changed = [
        "graph/routeL_llmmask_anomaly_aux/__init__.py",
        "graph/routeL_llmmask_anomaly_aux/README.md",
        "graph/routeL_llmmask_anomaly_aux/src/__init__.py",
        "graph/routeL_llmmask_anomaly_aux/src/routeL_utils.py",
        "graph/routeL_llmmask_anomaly_aux/src/review_models_routeL.py",
        "graph/routeL_llmmask_anomaly_aux/src/review_training_routeL.py",
        "graph/routeL_llmmask_anomaly_aux/src/export_user_features_routeL.py",
        "graph/routeL_llmmask_anomaly_aux/src/dryrun.py",
        "graph/routeL_llmmask_anomaly_aux/scripts/run_routeL_dryrun.sh",
        "graph/routeL_llmmask_anomaly_aux/configs/L1_early_noaux.yaml",
        "graph/routeL_llmmask_anomaly_aux/configs/L2_early_aux.yaml",
        "graph/routeL_llmmask_anomaly_aux/configs/L3_late_noaux.yaml",
        "graph/routeL_llmmask_anomaly_aux/configs/L4_late_aux.yaml",
        str(outdir / "dryrun_report.md"),
        str(outdir / "dryrun_report.json"),
        str(outdir / "model_shape_check.json"),
        str(outdir / "loss_check.json"),
    ]
    (outdir / "changed_files.txt").write_text("\n".join(changed) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
