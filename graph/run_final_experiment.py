from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data_utils import ensure_dir, prepare_graph_data
from .graph_pipeline import (
    apply_graph_guided_evidence_reweighting,
    build_edge_frames,
    build_review_and_user_artifacts,
    build_self_feature_matrix,
    compute_edge_stats,
)
from .llm_utils import build_llm_features_and_masks, numeric_feature_columns
from .relation_model import run_relation_aggregation_experiments
from .review_training import (
    build_review_dataloaders,
    build_review_model,
    build_tokenizer,
    encode_all_reviews,
    train_review_encoder,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GRAPH_DATA_DIR = PROJECT_ROOT / "graph data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "graph" / "outputs" / "yelpzip_final"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the final YelpZip graph experiment pipeline.")
    parser.add_argument("--graph_data_dir", default=str(DEFAULT_GRAPH_DATA_DIR))
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--llm_jsonl_path", default=None)
    parser.add_argument("--mask_source", choices=["llm", "full_text", "empty"], default="llm")
    parser.add_argument("--prefer_corrected_reviews", action="store_true", default=True)
    parser.add_argument("--overwrite_combined_files", action="store_true", default=False)
    parser.add_argument("--debug_use_empty_mask", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--senior_protocol", action="store_true", default=False)
    parser.add_argument("--balance_user_labels", action="store_true", default=False)
    parser.add_argument("--balanced_user_count", type=int, default=0)
    parser.add_argument("--min_user_reviews", type=int, default=3)
    parser.add_argument("--min_product_reviews", type=int, default=3)
    parser.add_argument("--time_bucket", choices=["month", "week"], default="week")
    parser.add_argument("--graph_mode", choices=["current", "senior", "senior_enhanced"], default="current")
    parser.add_argument("--senior_usu_ratio", type=float, default=0.10)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--top_m", type=int, default=3)
    parser.add_argument("--model_backbone", choices=["current_relation", "current_egat", "senior_exact", "senior_topk"], default="current_relation")
    parser.add_argument("--relation_model", choices=["mean", "edge_aware_gat", "relation_attn", "logreg", "mlp"], default="relation_attn")
    parser.add_argument("--edge_set", default=None)
    parser.add_argument("--use_abnormal_edge_weight", action="store_true", default=False)
    parser.add_argument("--use_abnormal_gate", action="store_true", default=False)
    parser.add_argument("--use_abnormal_value_gate", action="store_true", default=False)
    parser.add_argument("--use_abnormal_attention_bias", action="store_true", default=False)
    parser.add_argument("--abnormal_edge_lambda", type=float, default=1.0)
    parser.add_argument("--abnormal_edge_eta", type=float, default=0.5)
    parser.add_argument("--abnormal_gate_eta", type=float, default=0.5)
    parser.add_argument("--abnormal_pair_mode", choices=["both_high", "mean"], default="both_high")
    parser.add_argument("--abnormal_gate_learnable", action="store_true", default=False)
    parser.add_argument("--abnormal_attention_gamma", type=float, default=1.0)
    parser.add_argument("--abnormal_score_source", choices=["auto", "logic_ae", "llm_mask", "review_fake_score", "behavior"], default="auto")
    parser.add_argument("--use_tns_guided_logic", action="store_true", default=False)
    parser.add_argument("--tns_phi_days", type=int, default=5)
    parser.add_argument("--tns_logic_mode", choices=["boost", "product"], default="boost")
    parser.add_argument("--tns_logic_lambda", type=float, default=1.0)
    parser.add_argument("--logic_tns_topk", type=int, default=20)
    parser.add_argument("--use_node_gat", action="store_true", default=False)
    parser.add_argument("--node_gat_layers", type=int, default=1)
    parser.add_argument("--node_gat_heads", type=int, default=2)
    parser.add_argument("--node_gat_hidden_dim", type=int, default=64)
    parser.add_argument("--disable_graph_reweighting", action="store_true", default=False)
    parser.add_argument("--graph_reweight_alpha", type=float, default=0.70)
    parser.add_argument("--graph_support_top_k", type=int, default=20)
    parser.add_argument("--graph_support_neighbor_review_cap", type=int, default=20)
    parser.add_argument("--logic_threshold_mode", choices=["quantile", "fixed", "none"], default="quantile")
    parser.add_argument("--logic_threshold_quantile", type=float, default=0.60)
    parser.add_argument("--logic_threshold_value", type=float, default=0.30)
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--review_encoder", choices=["llm_masked_logic", "mock"], default="llm_masked_logic")
    parser.add_argument("--primary_model_name_or_path", default="roberta-base")
    parser.add_argument("--secondary_model_name_or_path", default=None)
    parser.add_argument("--vector_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--freeze_primary", action="store_true", default=False)
    parser.add_argument("--freeze_secondary", action="store_true", default=False)
    parser.add_argument("--abnormal_aux_lambda", type=float, default=0.0)
    parser.add_argument(
        "--abnormal_aux_position",
        choices=[
            "logic_query",
            "cross_context",
            "gated_cross",
            "logic_gated_cross",
            "logic_cross_context",
            "final_review_vector",
        ],
        default="logic_gated_cross",
    )
    parser.add_argument("--disable_cross_attention", action="store_true", default=False)
    parser.add_argument("--disable_logic_bilstm", action="store_true", default=False)
    parser.add_argument("--logic_pooling", choices=["attention", "mean"], default="attention")
    parser.add_argument("--gate_mode", choices=["learned", "no_gate", "fixed_half", "numeric_only", "text_only"], default="learned")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run_legacy_baselines", action="store_true", default=False)
    parser.add_argument("--legacy_roberta_model_dir", default="roberta-base")
    parser.add_argument("--legacy_gpu_ids", default="0")
    parser.add_argument("--legacy_num_epochs", type=int, default=3)
    parser.add_argument("--legacy_train_batch_size", type=int, default=16)
    parser.add_argument("--legacy_dev_batch_size", type=int, default=32)
    parser.add_argument("--legacy_test_batch_size", type=int, default=32)
    parser.add_argument("--legacy_print_step", type=int, default=200)
    parser.add_argument("--legacy_early_stop", type=int, default=3)
    parser.add_argument("--smoke_test", action="store_true", default=False)
    parser.add_argument("--smoke_max_users", type=int, default=0)
    return parser.parse_args()


def maybe_apply_senior_protocol_defaults(args: argparse.Namespace) -> None:
    if not args.senior_protocol:
        return
    args.balance_user_labels = True
    if args.balanced_user_count <= 0:
        args.balanced_user_count = 6742
    args.train_ratio = 0.64
    args.val_ratio = 0.16
    args.test_ratio = 0.20
    args.time_bucket = "week"
    if args.graph_mode == "current":
        args.graph_mode = "senior"


def maybe_apply_smoke_defaults(args: argparse.Namespace) -> None:
    if not args.smoke_test:
        return
    if args.review_encoder == "llm_masked_logic":
        args.review_encoder = "mock"
    args.debug_use_empty_mask = True
    args.secondary_model_name_or_path = None
    args.num_epochs = 1
    args.patience = 1
    args.batch_size = min(args.batch_size, 8)
    args.run_legacy_baselines = False
    if args.smoke_max_users <= 0:
        args.smoke_max_users = 120


def main() -> None:
    args = parse_args()
    maybe_apply_senior_protocol_defaults(args)
    maybe_apply_smoke_defaults(args)
    seed_everything(args.seed)
    device = resolve_device(args.device)

    output_root = ensure_dir(args.output_dir)
    llm_dir = ensure_dir(output_root / "llm_mask")
    review_encoder_dir = ensure_dir(output_root / "review_encoder")
    metrics_dir = ensure_dir(output_root / "metrics")
    legacy_dir = ensure_dir(output_root / "legacy_baselines")

    prepared = prepare_graph_data(
        graph_data_dir=args.graph_data_dir,
        output_dir=output_root / "prepared_data",
        data_path=args.data_path,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        min_user_reviews=args.min_user_reviews,
        min_product_reviews=args.min_product_reviews,
        prefer_corrected_reviews=args.prefer_corrected_reviews,
        overwrite_combined_files=args.overwrite_combined_files,
        smoke_max_users=args.smoke_max_users,
        balance_user_labels=args.balance_user_labels,
        balanced_user_count=args.balanced_user_count,
    )

    tokenizer = build_tokenizer(
        review_encoder=args.review_encoder,
        primary_model_name_or_path=args.primary_model_name_or_path,
        max_seq_length=args.max_seq_length,
    )
    llm_feature_df, abnormal_masks = build_llm_features_and_masks(
        review_df=prepared.review_df,
        tokenizer=tokenizer,
        llm_jsonl_path=args.llm_jsonl_path,
        output_dir=llm_dir,
        max_seq_length=args.max_seq_length,
        debug_use_empty_mask=args.debug_use_empty_mask,
        mask_source=args.mask_source,
    )

    dataloaders = build_review_dataloaders(
        review_df=prepared.review_df,
        llm_feature_df=llm_feature_df,
        abnormal_masks=abnormal_masks,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
    )

    model = build_review_model(
        review_encoder=args.review_encoder,
        primary_model_name_or_path=args.primary_model_name_or_path,
        numeric_feature_dim=len(numeric_feature_columns()),
        vector_dim=args.vector_dim,
        secondary_model_name_or_path=args.secondary_model_name_or_path,
        freeze_primary=args.freeze_primary,
        freeze_secondary=args.freeze_secondary,
        abnormal_aux_enabled=args.abnormal_aux_lambda > 0.0,
        abnormal_aux_position=args.abnormal_aux_position,
        disable_cross_attention=args.disable_cross_attention,
        disable_logic_bilstm=args.disable_logic_bilstm,
        logic_pooling=args.logic_pooling,
        gate_mode=args.gate_mode,
    )
    checkpoint_path, review_metrics_path = train_review_encoder(
        model=model,
        dataloaders=dataloaders,
        output_dir=review_encoder_dir,
        device=device,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        patience=args.patience,
        abnormal_aux_lambda=args.abnormal_aux_lambda,
    )
    encoding_artifacts = encode_all_reviews(
        model=model,
        dataloader=dataloaders["all"],
        review_df=prepared.review_df,
        checkpoint_path=checkpoint_path,
        metrics_path=review_metrics_path,
        device=device,
    )

    review_scores_df, user_df, _, user_abnormal_vectors, user_text_vectors = build_review_and_user_artifacts(
        review_df=prepared.review_df.sort_values("review_node_id").reset_index(drop=True),
        llm_feature_df=llm_feature_df,
        review_output_df=encoding_artifacts.review_output_df,
        review_vectors=encoding_artifacts.review_vectors,
        text_vectors=encoding_artifacts.text_vectors,
        output_dir=output_root,
        top_m=args.top_m,
        time_bucket=args.time_bucket,
    )
    edge_frames = build_edge_frames(
        user_df=user_df,
        user_text_vectors=user_text_vectors,
        user_abnormal_vectors=user_abnormal_vectors,
        output_dir=output_root,
        top_k=args.top_k,
        review_features=review_scores_df,
        logic_threshold_mode=args.logic_threshold_mode,
        logic_threshold_quantile=args.logic_threshold_quantile,
        logic_threshold_value=args.logic_threshold_value,
        graph_mode=args.graph_mode,
        senior_usu_ratio=args.senior_usu_ratio,
        use_tns_guided_logic=args.use_tns_guided_logic,
        tns_phi_days=args.tns_phi_days,
        tns_logic_mode=args.tns_logic_mode,
        tns_logic_lambda=args.tns_logic_lambda,
        logic_tns_topk=args.logic_tns_topk,
    )
    graph_reweighted_vectors = None
    if not args.disable_graph_reweighting:
        graph_support_top_k = args.graph_support_top_k if args.graph_support_top_k > 0 else args.top_k
        review_scores_df, user_df, graph_reweighted_vectors, graph_support_edges = apply_graph_guided_evidence_reweighting(
            review_features=review_scores_df,
            user_df=user_df,
            review_vectors=encoding_artifacts.review_vectors,
            edge_frames=edge_frames,
            output_dir=output_root,
            top_m=args.top_m,
            graph_top_k=graph_support_top_k,
            alpha=args.graph_reweight_alpha,
            neighbor_review_cap=args.graph_support_neighbor_review_cap,
        )
        edge_frames["GraphSupport"] = graph_support_edges
    edge_stats_df = compute_edge_stats(
        edge_frames=edge_frames,
        user_df=user_df,
        output_dir=output_root,
    )
    initial_self_features = build_self_feature_matrix(user_df, user_abnormal_vectors)
    if graph_reweighted_vectors is None:
        model_results_df = run_relation_aggregation_experiments(
            user_df=user_df,
            self_features=initial_self_features,
            edge_frames=edge_frames,
            output_dir=metrics_dir,
            review_encoder_name=args.review_encoder,
            model_kind=args.relation_model,
            seed=args.seed,
            backbone=args.model_backbone,
            relation_model=args.relation_model,
            use_abnormal_edge_weight=args.use_abnormal_edge_weight,
            use_abnormal_gate=args.use_abnormal_gate,
            use_abnormal_value_gate=args.use_abnormal_value_gate,
            use_abnormal_attention_bias=args.use_abnormal_attention_bias,
            abnormal_score_source=args.abnormal_score_source,
            abnormal_edge_lambda=args.abnormal_edge_lambda,
            abnormal_edge_eta=args.abnormal_edge_eta,
            abnormal_gate_eta=args.abnormal_gate_eta,
            abnormal_pair_mode=args.abnormal_pair_mode,
            abnormal_gate_learnable=args.abnormal_gate_learnable,
            abnormal_attention_gamma=args.abnormal_attention_gamma,
            review_scores_df=review_scores_df,
            selected_edge_set=args.edge_set,
            use_node_gat=args.use_node_gat,
        )
    else:
        initial_results_df = run_relation_aggregation_experiments(
            user_df=user_df,
            self_features=initial_self_features,
            edge_frames=edge_frames,
            output_dir=metrics_dir,
            review_encoder_name=f"{args.review_encoder}_initial",
            model_kind=args.relation_model,
            seed=args.seed,
            results_filename="model_results_initial.csv",
            backbone=args.model_backbone,
            relation_model=args.relation_model,
            use_abnormal_edge_weight=args.use_abnormal_edge_weight,
            use_abnormal_gate=args.use_abnormal_gate,
            use_abnormal_value_gate=args.use_abnormal_value_gate,
            use_abnormal_attention_bias=args.use_abnormal_attention_bias,
            abnormal_score_source=args.abnormal_score_source,
            abnormal_edge_lambda=args.abnormal_edge_lambda,
            abnormal_edge_eta=args.abnormal_edge_eta,
            abnormal_gate_eta=args.abnormal_gate_eta,
            abnormal_pair_mode=args.abnormal_pair_mode,
            abnormal_gate_learnable=args.abnormal_gate_learnable,
            abnormal_attention_gamma=args.abnormal_attention_gamma,
            review_scores_df=review_scores_df,
            selected_edge_set=args.edge_set,
            use_node_gat=args.use_node_gat,
        )
        reweighted_self_features = build_self_feature_matrix(user_df, graph_reweighted_vectors)
        reweighted_results_df = run_relation_aggregation_experiments(
            user_df=user_df,
            self_features=reweighted_self_features,
            edge_frames=edge_frames,
            output_dir=metrics_dir,
            review_encoder_name=f"{args.review_encoder}_graph_reweighted",
            model_kind=args.relation_model,
            seed=args.seed,
            results_filename="model_results_graph_reweighted.csv",
            backbone=args.model_backbone,
            relation_model=args.relation_model,
            use_abnormal_edge_weight=args.use_abnormal_edge_weight,
            use_abnormal_gate=args.use_abnormal_gate,
            use_abnormal_value_gate=args.use_abnormal_value_gate,
            use_abnormal_attention_bias=args.use_abnormal_attention_bias,
            abnormal_score_source=args.abnormal_score_source,
            abnormal_edge_lambda=args.abnormal_edge_lambda,
            abnormal_edge_eta=args.abnormal_edge_eta,
            abnormal_gate_eta=args.abnormal_gate_eta,
            abnormal_pair_mode=args.abnormal_pair_mode,
            abnormal_gate_learnable=args.abnormal_gate_learnable,
            abnormal_attention_gamma=args.abnormal_attention_gamma,
            review_scores_df=review_scores_df,
            selected_edge_set=args.edge_set,
            use_node_gat=args.use_node_gat,
        )
        model_results_df = pd.concat([initial_results_df, reweighted_results_df], ignore_index=True)
        model_results_df.to_csv(metrics_dir / "model_results.csv", index=False)

    legacy_results_df = None
    if args.run_legacy_baselines:
        from .legacy_textcls import run_legacy_review_baselines

        legacy_results_df = run_legacy_review_baselines(
            data_dir=prepared.legacy_tsv_dir,
            output_dir=legacy_dir,
            roberta_model_dir=args.legacy_roberta_model_dir,
            gpu_ids=args.legacy_gpu_ids,
            num_epochs=args.legacy_num_epochs,
            train_batch_size=args.legacy_train_batch_size,
            dev_batch_size=args.legacy_dev_batch_size,
            test_batch_size=args.legacy_test_batch_size,
            print_step=args.legacy_print_step,
            early_stop=args.legacy_early_stop,
        )

    summary = {
        "project_root": str(PROJECT_ROOT),
        "output_root": str(output_root),
        "prepared_source": str(prepared.source_path),
        "review_encoder": args.review_encoder,
        "review_encoder_semantics": "full_text_logic_aux" if args.mask_source == "full_text" else args.review_encoder,
        "abnormal_aux_lambda": float(args.abnormal_aux_lambda),
        "abnormal_aux_position": args.abnormal_aux_position,
        "disable_cross_attention": bool(args.disable_cross_attention),
        "disable_logic_bilstm": bool(args.disable_logic_bilstm),
        "logic_pooling": args.logic_pooling,
        "gate_mode": args.gate_mode,
        "device": str(device),
        "debug_use_empty_mask": bool(args.debug_use_empty_mask),
        "mask_source": args.mask_source,
        "smoke_test": bool(args.smoke_test),
        "senior_protocol": bool(args.senior_protocol),
        "balance_user_labels": bool(args.balance_user_labels),
        "balanced_user_count": int(args.balanced_user_count),
        "graph_mode": args.graph_mode,
        "model_backbone": args.model_backbone,
        "relation_model": args.relation_model,
        "edge_set": args.edge_set,
        "use_abnormal_edge_weight": bool(args.use_abnormal_edge_weight),
        "use_abnormal_gate": bool(args.use_abnormal_gate),
        "use_abnormal_value_gate": bool(args.use_abnormal_value_gate),
        "use_abnormal_attention_bias": bool(args.use_abnormal_attention_bias),
        "abnormal_score_source": args.abnormal_score_source,
        "abnormal_pair_mode": args.abnormal_pair_mode,
        "abnormal_edge_eta": float(args.abnormal_edge_eta),
        "abnormal_gate_eta": float(args.abnormal_gate_eta),
        "abnormal_gate_learnable": bool(args.abnormal_gate_learnable),
        "use_tns_guided_logic": bool(args.use_tns_guided_logic),
        "tns_phi_days": int(args.tns_phi_days),
        "tns_logic_mode": args.tns_logic_mode,
        "tns_logic_lambda": float(args.tns_logic_lambda),
        "logic_tns_topk": int(args.logic_tns_topk),
        "use_node_gat": bool(args.use_node_gat),
        "node_gat_layers": int(args.node_gat_layers),
        "node_gat_heads": int(args.node_gat_heads),
        "node_gat_hidden_dim": int(args.node_gat_hidden_dim),
        "graph_reweighting_enabled": not bool(args.disable_graph_reweighting),
        "review_count": int(len(prepared.review_df)),
        "user_count": int(len(user_df)),
        "best_graph_model": model_results_df.sort_values("auc", ascending=False).iloc[0].to_dict()
        if not model_results_df.empty
        else None,
        "best_edge_type_by_fake_fake_ratio": edge_stats_df.sort_values("fake_fake_ratio", ascending=False).iloc[0].to_dict()
        if not edge_stats_df.empty
        else None,
    }
    if legacy_results_df is not None and not legacy_results_df.empty:
        summary["legacy_results"] = legacy_results_df.to_dict(orient="records")

    run_config = vars(args).copy()
    run_config.update(
        {
            "resolved_device": str(device),
            "prepared_data_dir": str(output_root / "prepared_data"),
            "llm_dir": str(llm_dir),
            "review_encoder_dir": str(review_encoder_dir),
            "metrics_dir": str(metrics_dir),
        }
    )
    (output_root / "run_config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_root / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    review_scores_df.to_csv(output_root / "review_scores_enriched.csv", index=False)
    user_df.to_csv(output_root / "user_scores_enriched.csv", index=False)

    print("Graph final experiment finished.")
    print(f"Prepared data : {prepared.canonical_csv_path}")
    print(f"Model results : {metrics_dir / 'model_results.csv'}")
    print(f"Edge stats    : {metrics_dir / 'edge_stats.csv'}")
    if args.run_legacy_baselines:
        print(f"Legacy review baselines : {legacy_dir / 'legacy_baseline_results.csv'}")


if __name__ == "__main__":
    main()
