from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from graph.graph_pipeline import build_edge_frames, build_self_feature_matrix, compute_edge_stats
from graph.relation_model import run_relation_aggregation_experiments


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "graph" / "outputs"
ROUTE_A_BASE = PROJECT_ROOT / "graph" / "outputs" / "yelpzip_balanced_current_graph_no_reweight_20260502_160620"
ROUTE_B_BASE = PROJECT_ROOT / "graph" / "outputs" / "yelpzip_senior_backbone_clean_formal_20260502_101931"


ROUTE_EXPERIMENTS = {
    "A": [
        {"name": "A0_current_EGAT_Base_CB", "edge_set": "Base_CB", "backbone": "current_egat", "relation_model": "edge_aware_gat"},
        {"name": "A1_current_EGAT_Base_LogicAE_CB", "edge_set": "Base_LogicAE_CB", "backbone": "current_egat", "relation_model": "edge_aware_gat"},
        {"name": "A2_current_EGAT_Full", "edge_set": "Full", "backbone": "current_egat", "relation_model": "edge_aware_gat"},
    ],
    "B": [
        {
            "name": "B0_SeniorTopK20_Base",
            "edge_set": "Base",
            "backbone": "senior_topk",
            "relation_model": "edge_aware_gat",
            "relation_topk": 20,
        },
        {
            "name": "B1_SeniorTopK20_Base_LogicAE_CB",
            "edge_set": "Base_LogicAE_CB",
            "backbone": "senior_topk",
            "relation_model": "edge_aware_gat",
            "relation_topk": 20,
        },
        {
            "name": "B2_SeniorTopK20_Full",
            "edge_set": "Full",
            "backbone": "senior_topk",
            "relation_model": "edge_aware_gat",
            "relation_topk": 20,
        },
    ],
    "C": [
        {"name": "C0_current_Base_LogicAE_CB", "edge_set": "Base_LogicAE_CB", "backbone": "current_relation", "relation_model": "relation_attn"},
        {"name": "C1_current_Base_CB_abnormal_edge_weight", "edge_set": "Base_CB", "backbone": "current_relation", "relation_model": "relation_attn", "use_abnormal_edge_weight": True},
        {"name": "C2_current_Base_CB_abnormal_gate", "edge_set": "Base_CB", "backbone": "current_relation", "relation_model": "relation_attn", "use_abnormal_gate": True, "use_abnormal_edge_weight": True},
        {"name": "C3_current_EGAT_Base_CB_abnormal_attention_bias", "edge_set": "Base_CB", "backbone": "current_egat", "relation_model": "edge_aware_gat", "use_abnormal_attention_bias": True},
        {"name": "C4_senior_SeniorBaseExact_LogicAE_CB", "edge_set": "Base_LogicAE_CB", "backbone": "senior_exact", "relation_model": "edge_aware_gat", "base_dir": str(ROUTE_B_BASE)},
        {"name": "C5_senior_SeniorBaseExact_CB_abnormal_edge_weight", "edge_set": "Base_CB", "backbone": "senior_exact", "relation_model": "edge_aware_gat", "base_dir": str(ROUTE_B_BASE), "use_abnormal_edge_weight": True},
        {"name": "C6_senior_SeniorBaseExact_abnormal_attention_bias", "edge_set": "Base_CB", "backbone": "senior_exact", "relation_model": "edge_aware_gat", "base_dir": str(ROUTE_B_BASE), "use_abnormal_attention_bias": True},
    ],
}


def _read_senior_notes() -> dict:
    txt_path = PROJECT_ROOT / "graph" / "outputs" / "senior_paper_text.txt"
    text = txt_path.read_text(encoding="utf-8")
    return {
        "txt_path": str(txt_path),
        "key_structures": [
            "UPU / UTU / USU",
            "behavioral + textual statistical features",
            "relation-specific multi-head GAT",
            "relation-level attention fusion",
            "node-level multi-layer GAT with residual",
            "LayerNorm + dropout",
            "multi-scale feature fusion",
        ],
        "notes": "SeniorBaseExact is paper-aligned using in-repo code and text stats only; not a claim of full exact paper reproduction.",
        "txt_excerpt_hint": "GAT heads main=8, relation=4, 1:1 undersampling, 6.4/1.6/2.0 split, USU top 10%.",
        "senior_topk_note": "SeniorTopK20_EGAT keeps senior relation definitions (UPU/UTU/USU) but prunes each relation per node to top-k neighbors; this is a senior relation top-k baseline, not full SeniorBaseExact.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run route experiments from existing in-repo artifacts.")
    parser.add_argument("--route", choices=["A", "B", "C"], required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--abnormal_edge_lambda", type=float, default=1.0)
    parser.add_argument("--abnormal_attention_gamma", type=float, default=1.0)
    parser.add_argument("--abnormal_score_source", default="auto")
    return parser.parse_args()


def _default_base_dir(route: str) -> Path:
    if route == "B":
        return ROUTE_B_BASE
    return ROUTE_A_BASE


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _load_base_artifacts(base_dir: Path) -> dict:
    prepared_dir = base_dir / "prepared_data"
    logic_dir = base_dir / "logic_vectors"
    review_scores_df = pd.read_csv(base_dir / "review_scores_enriched.csv")
    user_scores_path = base_dir / "user_scores_enriched.csv"
    user_df = pd.read_csv(user_scores_path) if user_scores_path.exists() else pd.read_csv(logic_dir / "user_summary.csv")
    review_df = pd.read_csv(prepared_dir / "reviews_canonical.csv")
    user_text_vectors = np.load(logic_dir / "user_text_vectors.npy")
    user_abnormal_vectors = np.load(logic_dir / "user_abnormal_vectors.npy")
    metadata = json.loads((prepared_dir / "dataset_metadata.json").read_text(encoding="utf-8"))
    run_config = json.loads((base_dir / "run_config.json").read_text(encoding="utf-8"))
    return {
        "review_scores_df": review_scores_df,
        "user_df": user_df,
        "review_df": review_df,
        "user_text_vectors": user_text_vectors,
        "user_abnormal_vectors": user_abnormal_vectors,
        "dataset_metadata": metadata,
        "base_run_config": run_config,
    }


def _build_route_edges(base_artifacts: dict, graph_mode: str, top_k: int, senior_usu_ratio: float, route_output_dir: Path) -> dict[str, pd.DataFrame]:
    return build_edge_frames(
        user_df=base_artifacts["user_df"],
        user_text_vectors=base_artifacts["user_text_vectors"],
        user_abnormal_vectors=base_artifacts["user_abnormal_vectors"],
        output_dir=route_output_dir,
        top_k=top_k,
        review_features=base_artifacts["review_scores_df"],
        logic_threshold_mode="quantile",
        logic_threshold_quantile=0.60,
        logic_threshold_value=0.30,
        graph_mode=graph_mode,
        senior_usu_ratio=senior_usu_ratio,
    )


def _write_train_log(route_dir: Path, lines: list[str]) -> None:
    (route_dir / "train.log").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_route(args: argparse.Namespace) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    base_dir = _default_base_dir(args.route)
    base_artifacts = _load_base_artifacts(base_dir)
    base_cfg = base_artifacts["base_run_config"]
    graph_mode = "senior" if args.route == "B" else "current"
    top_k = int(base_cfg.get("top_k", 20))
    senior_usu_ratio = float(base_cfg.get("senior_usu_ratio", 0.10))

    edge_frames = _build_route_edges(
        base_artifacts=base_artifacts,
        graph_mode=graph_mode,
        top_k=top_k,
        senior_usu_ratio=senior_usu_ratio,
        route_output_dir=output_root,
    )
    edge_stats_df = compute_edge_stats(edge_frames=edge_frames, user_df=base_artifacts["user_df"], output_dir=output_root)
    self_features = build_self_feature_matrix(base_artifacts["user_df"], base_artifacts["user_abnormal_vectors"])

    senior_notes = _read_senior_notes() if args.route in {"B", "C"} else None
    route_rows = []
    train_lines = [
        f"route={args.route}",
        f"timestamp={timestamp}",
        f"base_dir={base_dir}",
        f"graph_mode={graph_mode}",
        f"top_k={top_k}",
        f"senior_usu_ratio={senior_usu_ratio}",
    ]

    for exp in ROUTE_EXPERIMENTS[args.route]:
        exp_name = exp["name"]
        exp_dir = output_root / exp_name
        metrics_dir = exp_dir / "metrics"
        exp_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)

        exp_base_dir = Path(exp.get("base_dir", str(base_dir)))
        exp_artifacts = base_artifacts if exp_base_dir == base_dir else _load_base_artifacts(exp_base_dir)
        exp_graph_mode = "senior" if "senior" in exp_name.lower() or exp["backbone"] == "senior_exact" else "current"
        exp_edge_frames = edge_frames if exp_base_dir == base_dir and exp_graph_mode == graph_mode else _build_route_edges(
            base_artifacts=exp_artifacts,
            graph_mode=exp_graph_mode,
            top_k=top_k,
            senior_usu_ratio=senior_usu_ratio,
            route_output_dir=exp_dir,
        )
        exp_edge_stats_df = edge_stats_df if exp_base_dir == base_dir and exp_graph_mode == graph_mode else compute_edge_stats(
            edge_frames=exp_edge_frames,
            user_df=exp_artifacts["user_df"],
            output_dir=exp_dir,
        )
        exp_self_features = self_features if exp_base_dir == base_dir else build_self_feature_matrix(
            exp_artifacts["user_df"],
            exp_artifacts["user_abnormal_vectors"],
        )

        result_df = run_relation_aggregation_experiments(
            user_df=exp_artifacts["user_df"],
            self_features=exp_self_features,
            edge_frames=exp_edge_frames,
            output_dir=metrics_dir,
            review_encoder_name="llm_masked_logic",
            model_kind=exp["relation_model"],
            seed=args.seed,
            backbone=exp["backbone"],
            relation_model=exp["relation_model"],
            use_abnormal_edge_weight=bool(exp.get("use_abnormal_edge_weight", False)),
            use_abnormal_gate=bool(exp.get("use_abnormal_gate", False)),
            use_abnormal_attention_bias=bool(exp.get("use_abnormal_attention_bias", False)),
            abnormal_score_source=args.abnormal_score_source,
            abnormal_edge_lambda=args.abnormal_edge_lambda,
            abnormal_attention_gamma=args.abnormal_attention_gamma,
            review_scores_df=exp_artifacts["review_scores_df"],
            selected_edge_set=exp["edge_set"],
            relation_topk=exp.get("relation_topk"),
        )

        graph_rows = result_df[result_df["edge_set"] == exp["edge_set"]].copy()
        best_row = graph_rows.sort_values("auc", ascending=False).iloc[0].to_dict() if not graph_rows.empty else {}
        exp_config = {
            "route": args.route,
            "experiment_name": exp_name,
            "output_dir": str(exp_dir),
            "base_dir": str(exp_base_dir),
            "graph_mode": exp_graph_mode,
            "edge_set": exp["edge_set"],
            "model_backbone": exp["backbone"],
            "relation_model": exp["relation_model"],
            "relation_topk": exp.get("relation_topk"),
            "use_abnormal_edge_weight": bool(exp.get("use_abnormal_edge_weight", False)),
            "use_abnormal_gate": bool(exp.get("use_abnormal_gate", False)),
            "use_abnormal_attention_bias": bool(exp.get("use_abnormal_attention_bias", False)),
            "abnormal_edge_lambda": float(args.abnormal_edge_lambda),
            "abnormal_attention_gamma": float(args.abnormal_attention_gamma),
            "abnormal_score_source": args.abnormal_score_source,
            "seed": int(args.seed),
        }
        if args.route == "B":
            exp_config["senior_exact_notes"] = senior_notes
        (exp_dir / "config.json").write_text(json.dumps(exp_config, indent=2, ensure_ascii=False), encoding="utf-8")
        (exp_dir / "run_config.json").write_text(json.dumps(exp_config, indent=2, ensure_ascii=False), encoding="utf-8")
        (exp_dir / "run_summary.json").write_text(
            json.dumps(
                {
                    **exp_config,
                    "best_graph_model": best_row,
                    "dataset_metadata": exp_artifacts["dataset_metadata"],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        if exp_graph_mode == graph_mode and exp_base_dir == base_dir:
            _copy_if_exists(output_root / "edges" / "edge_build_config.json", exp_dir / "edges" / "edge_build_config.json")
            _copy_if_exists(output_root / "metrics" / "edge_stats.csv", exp_dir / "metrics" / "edge_stats.csv")
        else:
            pass
        train_lines.append(f"{exp_name}: auc={best_row.get('auc')} ap={best_row.get('ap')} f1={best_row.get('f1')}")
        route_rows.append(
            {
                "route": args.route,
                "output_dir": str(exp_dir),
                "model_name": best_row.get("model_name"),
                "graph_mode": exp_graph_mode,
                "edge_set": exp["edge_set"],
                "backbone": exp["backbone"],
                "relation_model": exp["relation_model"],
                "relation_topk": exp.get("relation_topk"),
                "use_abnormal_edge_weight": bool(exp.get("use_abnormal_edge_weight", False)),
                "use_abnormal_gate": bool(exp.get("use_abnormal_gate", False)),
                "use_abnormal_attention_bias": bool(exp.get("use_abnormal_attention_bias", False)),
                "abnormal_score_source": args.abnormal_score_source,
                "AUC": best_row.get("auc"),
                "AP": best_row.get("ap"),
                "F1": best_row.get("f1"),
                "Recall": best_row.get("recall"),
                "Precision": best_row.get("precision"),
                "best_epoch": best_row.get("best_epoch"),
                "seed": int(args.seed),
                "notes": senior_notes["notes"] if args.route == "B" else "",
            }
        )

    summary_df = pd.DataFrame(route_rows)
    summary_df.to_csv(output_root / "route_summary.csv", index=False)
    route_config = {
        "route": args.route,
        "output_root": str(output_root),
        "seed": int(args.seed),
        "abnormal_edge_lambda": float(args.abnormal_edge_lambda),
        "abnormal_attention_gamma": float(args.abnormal_attention_gamma),
        "abnormal_score_source": args.abnormal_score_source,
        "base_dir": str(base_dir),
        "graph_mode": graph_mode,
        "top_k": top_k,
        "senior_usu_ratio": senior_usu_ratio,
        "senior_relation_topk": 20 if args.route == "B" else None,
    }
    route_best = summary_df.sort_values("AUC", ascending=False).iloc[0].to_dict() if not summary_df.empty else None
    (output_root / "config.json").write_text(json.dumps(route_config, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_root / "run_summary.json").write_text(
        json.dumps(
            {
                **route_config,
                "best_experiment": route_best,
                "senior_exact_notes": senior_notes if args.route in {"B", "C"} else None,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _write_train_log(output_root, train_lines)
    return output_root


def main() -> None:
    args = parse_args()
    output_dir = run_route(args)
    print(output_dir)


if __name__ == "__main__":
    main()
