from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score

from graph.baseline_comparison.src.data_loader import load_protocol_bundle
from graph.baseline_comparison.src.utils import save_json, setup_logger
from graph.graph_pipeline import (
    build_routek_d1main_rns_topk_graph_frames,
    compute_edge_stats,
    _routek_compute_threshold_operating_points,
)
from graph.relation_model import run_relation_aggregation_experiments


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GRAPH_DIR = PROJECT_ROOT / "graph"
BASE_PROTOCOL_DIR = GRAPH_DIR / "outputs" / "yelpzip_balanced_current_graph_no_reweight_20260502_160620"
REFERENCE_D1_DIR = GRAPH_DIR / "outputs" / "routeD_tns_guided_logic_egat_20260504_200855" / "D1_EGAT_Base_LogicAE_CB"


K4_ARMS = [
    {"arm_id": 0, "UPU": 20, "UTU": 20, "USU": 20, "LogicAE_CB": 20},
    {"arm_id": 1, "UPU": 10, "UTU": 10, "USU": 20, "LogicAE_CB": 20},
    {"arm_id": 2, "UPU": 10, "UTU": 10, "USU": 10, "LogicAE_CB": 30},
    {"arm_id": 3, "UPU": 5, "UTU": 5, "USU": 20, "LogicAE_CB": 30},
    {"arm_id": 4, "UPU": 20, "UTU": 10, "USU": 10, "LogicAE_CB": 30},
    {"arm_id": 5, "UPU": 10, "UTU": 20, "USU": 10, "LogicAE_CB": 30},
    {"arm_id": 6, "UPU": 5, "UTU": 10, "USU": 10, "LogicAE_CB": 40},
    {"arm_id": 7, "UPU": 15, "UTU": 15, "USU": 15, "LogicAE_CB": 25},
    {"arm_id": 8, "UPU": 5, "UTU": 5, "USU": 10, "LogicAE_CB": 40},
    {"arm_id": 9, "UPU": 10, "UTU": 5, "USU": 20, "LogicAE_CB": 30},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Route K D1Main + RNS-style top-k experiments.")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha_abnormal", type=float, default=0.5)
    parser.add_argument("--beta_tns", type=float, default=0.2)
    parser.add_argument("--gamma_interaction", type=float, default=0.2)
    parser.add_argument("--abnormal_score_source", default="auto")
    parser.add_argument("--k4_warmup_epochs", type=int, default=15)
    parser.add_argument("--k5_warmup_epochs", type=int, default=20)
    parser.add_argument("--bandit_lambda_density", type=float, default=0.02)
    return parser.parse_args()


def _sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _sha256_edge_frame(df: pd.DataFrame) -> str:
    if df.empty:
        return "EMPTY"
    hashed = pd.util.hash_pandas_object(df, index=False).values.tobytes()
    return hashlib.sha256(hashed).hexdigest()


def _load_d1_assets() -> dict[str, Any]:
    bundle = load_protocol_bundle()
    review_scores_df = pd.read_csv(BASE_PROTOCOL_DIR / "review_scores_enriched.csv")
    user_text_vectors = np.load(BASE_PROTOCOL_DIR / "logic_vectors" / "user_text_vectors.npy")
    user_abnormal_vectors = np.load(BASE_PROTOCOL_DIR / "logic_vectors" / "user_abnormal_vectors.npy")
    run_summary = json.loads((REFERENCE_D1_DIR / "run_summary.json").read_text(encoding="utf-8"))
    config = json.loads((REFERENCE_D1_DIR / "config.json").read_text(encoding="utf-8"))
    model_results = pd.read_csv(REFERENCE_D1_DIR / "metrics" / "model_results.csv")
    graph_row = model_results.loc[model_results["edge_set"] == "Base_LogicAE_CB"].iloc[0].to_dict()
    return {
        "bundle": bundle,
        "review_scores_df": review_scores_df,
        "user_text_vectors": user_text_vectors,
        "user_abnormal_vectors": user_abnormal_vectors,
        "run_summary": run_summary,
        "config": config,
        "graph_row": graph_row,
    }


def _copy_selected_artifacts(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in ["run_summary.json", "config.json", "train.log", "val_predictions.csv", "test_predictions.csv", "epoch_metrics.csv"]:
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, dst_dir / name)
    for rel in [
        "metrics/model_results.csv",
        "metrics/epoch_metrics.csv",
        "metrics/test_predictions.csv",
        "metrics/edge_stats.csv",
        "metrics/topk_edge_quality_by_relation.csv",
        "metrics/topk_rank_score_stats.csv",
        "metrics/topk_relation_degree_stats.csv",
        "metrics/topk_selection_overlap.csv",
        "metrics/k2s_relation_specific_stats.csv",
        "metrics/threshold_operating_points.csv",
        "metrics/k5_rns_arm_search.csv",
        "metrics/k5_base_reserve_stats.csv",
        "metrics/k5_neighbor_distance_stats.csv",
        "metrics/k5_selection_overlap.csv",
        "metrics/bandit_arm_search.csv",
        "edges/edge_build_config.json",
    ]:
        src = src_dir / rel
        if src.exists():
            dst = dst_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def _load_completed_experiment_row(exp_dir: Path) -> dict[str, Any] | None:
    metrics_path = exp_dir / "metrics" / "model_results.csv"
    summary_path = exp_dir / "run_summary.json"
    if not metrics_path.exists() and not summary_path.exists():
        return None
    row: dict[str, Any] = {}
    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        if "edge_set" in df.columns and not df.empty:
            match = df[df["edge_set"] == "Base_LogicAE_CB"]
            if match.empty:
                match = df.head(1)
            if not match.empty:
                row = match.iloc[0].to_dict()
    if summary_path.exists() and not row:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        row = payload.get("best_graph_model") or payload.get("best_metrics") or {}
    if not row:
        return None
    return row


def _prediction_threshold_metrics(val_predictions_path: Path, test_predictions_path: Path) -> pd.DataFrame:
    return _routek_compute_threshold_operating_points(val_predictions_path, test_predictions_path, test_predictions_path.parent / "threshold_operating_points.csv")


def _sanity_diagnostics(
    *,
    exp_dir: Path,
    base_artifacts: dict[str, Any],
    edge_frames: dict[str, pd.DataFrame],
    feature_source: str,
    review_encoder_name: str,
) -> dict[str, Any]:
    reference_edges = base_artifacts["bundle"].edge_frames
    feature_hash = _sha256_array(base_artifacts["bundle"].node_features)
    split_user_ids = {
        split: sorted(base_artifacts["bundle"].user_df.loc[base_artifacts["bundle"].user_df["split"].astype(str) == split, "user_id"].astype(str).tolist())
        for split in ["train", "val", "test"]
    }
    split_hash = {split: hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest() for split, values in split_user_ids.items()}
    edge_count_cmp = {
        relation: {
            "current": int(len(edge_frames.get(relation, pd.DataFrame()))),
            "reference": int(len(reference_edges.get(relation, pd.DataFrame()))),
        }
        for relation in ["UPU", "UTU", "USU", "LogicAE_CB"]
    }
    edge_hash_cmp = {
        relation: {
            "current": _sha256_edge_frame(edge_frames.get(relation, pd.DataFrame())),
            "reference": _sha256_edge_frame(reference_edges.get(relation, pd.DataFrame())),
        }
        for relation in ["UPU", "UTU", "USU", "LogicAE_CB"]
    }
    diagnostics = {
        "feature_source": feature_source,
        "review_encoder": review_encoder_name,
        "feature_dim": int(base_artifacts["bundle"].node_features.shape[1]),
        "feature_hash": feature_hash,
        "split_user_hash": split_hash,
        "edge_count_comparison": edge_count_cmp,
        "edge_hash_comparison": edge_hash_cmp,
        "reference_threshold": float(base_artifacts["graph_row"]["threshold"]),
        "reference_model_config": {
            "review_encoder": base_artifacts["graph_row"]["review_encoder"],
            "backbone": base_artifacts["graph_row"]["backbone"],
            "relation_model": base_artifacts["graph_row"]["relation_model"],
        },
    }
    save_json(exp_dir / "d1_reproduction_diagnostics.json", diagnostics)
    return diagnostics


def _build_routek_summary_row(
    *,
    exp_name: str,
    metrics_row: dict[str, Any],
    feature_source: str,
    relation_k: dict[str, int],
    topk_mode: str,
    use_bandit: bool,
    selected_arm_id: int | str | None,
    candidate_topm: dict[str, int] | None,
    use_recall_constrained_reward: bool,
    use_base_reserve: bool,
    base_reserve_ratio: float,
    use_dual_channel: bool,
    use_relation_specific_denoise: bool,
    notes: str,
) -> dict[str, Any]:
    candidate_topm = candidate_topm or {}
    return {
        "experiment_name": exp_name,
        "feature_source": feature_source,
        "review_encoder": "llm_masked_logic",
        "is_strict_d1_feature": True,
        "topk_mode": topk_mode,
        "UPU_k": int(relation_k["UPU"]),
        "UTU_k": int(relation_k["UTU"]),
        "USU_k": int(relation_k["USU"]),
        "LogicAE_CB_k": int(relation_k["LogicAE_CB"]),
        "candidate_topM": json.dumps(candidate_topm, ensure_ascii=False, sort_keys=True) if candidate_topm else "",
        "alpha_abnormal": 0.5,
        "beta_tns": 0.2,
        "gamma_interaction": 0.2,
        "use_bandit": bool(use_bandit),
        "selected_arm_id": selected_arm_id if selected_arm_id is not None else "",
        "use_recall_constrained_reward": bool(use_recall_constrained_reward),
        "use_base_reserve": bool(use_base_reserve),
        "base_reserve_ratio": float(base_reserve_ratio),
        "use_dual_channel": bool(use_dual_channel),
        "use_relation_specific_denoise": bool(use_relation_specific_denoise),
        "num_edges": int(metrics_row.get("num_edges", 0)),
        "AUC": metrics_row.get("auc"),
        "AP": metrics_row.get("ap"),
        "F1": metrics_row.get("f1"),
        "Recall": metrics_row.get("recall"),
        "Precision": metrics_row.get("precision"),
        "best_epoch": metrics_row.get("best_epoch", "UNKNOWN_FROM_D1"),
        "test_threshold": metrics_row.get("threshold"),
        "notes": notes,
    }


def _train_one_experiment(
    *,
    exp_dir: Path,
    exp_name: str,
    base_artifacts: dict[str, Any],
    topk_mode: str,
    relation_k: dict[str, int],
    alpha_abnormal: float,
    beta_tns: float,
    gamma_interaction: float,
    abnormal_score_source: str,
    seed: int,
    preserve_d1_for_fixed_k0: bool = False,
    candidate_topm: dict[str, int] | None = None,
    use_relation_specific_denoise: bool = False,
    use_base_reserve: bool = False,
    base_reserve_ratio: float = 0.30,
    max_epochs_override: int | None = None,
    patience_override: int | None = None,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    exp_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = exp_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    edge_frames = build_routek_d1main_rns_topk_graph_frames(
        user_df=base_artifacts["bundle"].user_df.copy(),
        review_features=base_artifacts["review_scores_df"].copy(),
        user_text_vectors=base_artifacts["user_text_vectors"],
        user_abnormal_vectors=base_artifacts["user_abnormal_vectors"],
        d1_edge_frames=base_artifacts["bundle"].edge_frames,
        output_dir=exp_dir,
        topk_mode=topk_mode,
        relation_k=relation_k,
        abnormal_score_source=abnormal_score_source,
        alpha_abnormal=alpha_abnormal,
        beta_tns=beta_tns,
        gamma_interaction=gamma_interaction,
        tns_phi_days=5,
        candidate_topm=candidate_topm,
        use_relation_specific_denoise=use_relation_specific_denoise,
        use_base_reserve=use_base_reserve,
        base_reserve_ratio=base_reserve_ratio,
        preserve_d1_for_fixed_k0=preserve_d1_for_fixed_k0,
    )
    edge_stats = compute_edge_stats(edge_frames=edge_frames, user_df=base_artifacts["bundle"].user_df, output_dir=exp_dir)
    diagnostics = _sanity_diagnostics(
        exp_dir=exp_dir,
        base_artifacts=base_artifacts,
        edge_frames=edge_frames,
        feature_source="D1 llm_masked_logic self-feature matrix from user_scores_enriched.csv + logic_vectors/user_abnormal_vectors.npy via build_self_feature_matrix",
        review_encoder_name="llm_masked_logic",
    )

    result_df = run_relation_aggregation_experiments(
        user_df=base_artifacts["bundle"].user_df,
        self_features=base_artifacts["bundle"].node_features.astype(np.float32),
        edge_frames=edge_frames,
        output_dir=metrics_dir,
        review_encoder_name="llm_masked_logic",
        model_kind="edge_aware_gat",
        seed=seed,
        backbone="current_egat",
        relation_model="edge_aware_gat",
        review_scores_df=base_artifacts["review_scores_df"],
        selected_edge_set="Base_LogicAE_CB",
        use_node_gat=False,
        use_self_graph_gate=False,
        use_relation_sigmoid_gate=False,
        use_self_aux_loss=False,
        use_abnormal_edge_weight=False,
        use_abnormal_gate=False,
        use_abnormal_value_gate=False,
        use_abnormal_attention_bias=False,
        abnormal_score_source=abnormal_score_source,
        tns_attention_relations=[],
        max_epochs_override=max_epochs_override,
        patience_override=patience_override,
        return_training_details=True,
    )
    graph_row = result_df.loc[result_df["edge_set"] == "Base_LogicAE_CB"].iloc[0].to_dict()
    if (metrics_dir / "val_predictions.csv").exists() and (metrics_dir / "test_predictions.csv").exists():
        _prediction_threshold_metrics(metrics_dir / "val_predictions.csv", metrics_dir / "test_predictions.csv")

    run_summary = {
        "route": "K_D1MAIN_RNS",
        "experiment_name": exp_name,
        "output_dir": str(exp_dir),
        "base_dir": str(BASE_PROTOCOL_DIR),
        "reference_dir": str(REFERENCE_D1_DIR),
        "graph_mode": "current",
        "edge_set": "Base_LogicAE_CB",
        "model_backbone": "current_egat",
        "relation_model": "edge_aware_gat",
        "review_encoder": "llm_masked_logic",
        "feature_source": diagnostics["feature_source"],
        "seed": int(seed),
        "topk_mode": topk_mode,
        "relation_k": {k: int(v) for k, v in relation_k.items()},
        "candidate_topM": {k: int(v) for k, v in (candidate_topm or {}).items()},
        "alpha_abnormal": float(alpha_abnormal),
        "beta_tns": float(beta_tns),
        "gamma_interaction": float(gamma_interaction),
        "abnormal_score_source": str(abnormal_score_source),
        "best_graph_model": graph_row,
    }
    reference_auc = float(base_artifacts["graph_row"]["auc"])
    reference_ap = float(base_artifacts["graph_row"]["ap"])
    notes = []
    if exp_name == "K0_D1Main":
        if graph_row["auc"] < reference_auc - 0.006 or graph_row["ap"] < reference_ap - 0.006:
            notes.append("K0_D1Main_NOT_STRICT_REPRODUCTION")
    run_summary["notes"] = notes
    save_json(exp_dir / "config.json", run_summary)
    save_json(exp_dir / "run_summary.json", run_summary)
    (exp_dir / "train.log").write_text(
        "\n".join(
            [
                f"experiment={exp_name}",
                f"topk_mode={topk_mode}",
                f"relation_k={relation_k}",
                f"auc={graph_row.get('auc')}",
                f"ap={graph_row.get('ap')}",
                f"f1={graph_row.get('f1')}",
                f"notes={notes}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return graph_row, edge_frames


def _k4_reward(val_auc: float, val_ap: float, edge_density_norm: float, lambda_density: float) -> float:
    return float(val_auc) + 0.5 * float(val_ap) - float(lambda_density) * float(edge_density_norm)


def _k5_reward(
    *,
    val_auc: float,
    val_ap: float,
    val_f2: float,
    edge_density_norm: float,
    lambda_density: float,
    k0_val_recall: float,
    val_recall: float,
) -> tuple[float, float]:
    recall_drop_penalty = max(0.0, float(k0_val_recall) - float(val_recall) - 0.02)
    reward = float(val_auc) + 0.5 * float(val_ap) + 0.3 * float(val_f2) - float(lambda_density) * float(edge_density_norm) - recall_drop_penalty
    return reward, recall_drop_penalty


def _route_specs() -> list[dict[str, Any]]:
    return [
        {"experiment_name": "K0_D1Main", "topk_mode": "fixed_original", "relation_k": {"UPU": 20, "UTU": 20, "USU": 20, "LogicAE_CB": 20}},
        {"experiment_name": "K1_D1Main", "topk_mode": "abnormal_aware", "relation_k": {"UPU": 20, "UTU": 20, "USU": 20, "LogicAE_CB": 20}},
        {"experiment_name": "K2_D1Main", "topk_mode": "abnormal_tns_aware", "relation_k": {"UPU": 20, "UTU": 20, "USU": 20, "LogicAE_CB": 20}},
        {"experiment_name": "K3_D1Main", "topk_mode": "abnormal_tns_aware", "relation_k": {"UPU": 10, "UTU": 10, "USU": 10, "LogicAE_CB": 30}},
        {"experiment_name": "K4_D1Main", "topk_mode": "abnormal_tns_aware", "relation_k": {"UPU": 20, "UTU": 20, "USU": 20, "LogicAE_CB": 20}, "use_bandit": True},
        {
            "experiment_name": "K2S_D1Main",
            "topk_mode": "abnormal_tns_aware",
            "relation_k": {"UPU": 20, "UTU": 20, "USU": 20, "LogicAE_CB": 20},
            "candidate_topm": {"UPU": 100, "UTU": 100, "USU": 50, "LogicAE_CB": 50},
            "use_relation_specific_denoise": True,
        },
        {
            "experiment_name": "K5_D1Main",
            "topk_mode": "abnormal_tns_aware",
            "relation_k": {"UPU": 20, "UTU": 20, "USU": 20, "LogicAE_CB": 20},
            "use_bandit": True,
            "is_k5": True,
            "candidate_topm": {"UPU": 100, "UTU": 100, "USU": 50, "LogicAE_CB": 50},
            "use_base_reserve": True,
            "base_reserve_ratio": 0.30,
        },
    ]


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_root / f"routeK_d1main_rns_{timestamp}.log")

    artifacts = _load_d1_assets()
    route_rows: list[dict[str, Any]] = []
    reference_row = {
        "experiment_name": "CurrentTopK_EGAT_Base_LogicAE_CB_REFERENCE",
        "feature_source": "D1 llm_masked_logic",
        "review_encoder": "llm_masked_logic",
        "is_strict_d1_feature": True,
        "topk_mode": "reference_only",
        "UPU_k": 20,
        "UTU_k": 20,
        "USU_k": 20,
        "LogicAE_CB_k": 20,
        "candidate_topM": "",
        "alpha_abnormal": 0.5,
        "beta_tns": 0.2,
        "gamma_interaction": 0.2,
        "use_bandit": False,
        "selected_arm_id": "",
        "use_recall_constrained_reward": False,
        "use_base_reserve": False,
        "base_reserve_ratio": 0.0,
        "use_dual_channel": False,
        "use_relation_specific_denoise": False,
        "num_edges": int(sum(len(artifacts["bundle"].edge_frames[name]) for name in ["UPU", "UTU", "USU", "LogicAE_CB"])),
        "AUC": artifacts["graph_row"]["auc"],
        "AP": artifacts["graph_row"]["ap"],
        "F1": artifacts["graph_row"]["f1"],
        "Recall": artifacts["graph_row"]["recall"],
        "Precision": artifacts["graph_row"]["precision"],
        "best_epoch": "UNKNOWN_FROM_D1",
        "test_threshold": artifacts["graph_row"]["threshold"],
        "notes": "Reference only; not trained in this route.",
    }

    k0_val_recall = None

    for spec in _route_specs():
        exp_name = spec["experiment_name"]
        logger.info("starting %s", exp_name)
        exp_dir = output_root / exp_name

        existing_root = _load_completed_experiment_row(exp_dir)
        if existing_root is not None and (exp_dir / "run_summary.json").exists():
            logger.info("skipping completed %s", exp_name)
            route_rows.append(
                _build_routek_summary_row(
                    exp_name=exp_name,
                    metrics_row=existing_root,
                    feature_source="D1 llm_masked_logic self-feature matrix from user_scores_enriched.csv + logic_vectors/user_abnormal_vectors.npy via build_self_feature_matrix",
                    relation_k=spec["relation_k"],
                    topk_mode=spec["topk_mode"],
                    use_bandit=bool(spec.get("use_bandit", False)),
                    selected_arm_id=None,
                    candidate_topm=spec.get("candidate_topm"),
                    use_recall_constrained_reward=bool(spec.get("is_k5")),
                    use_base_reserve=bool(spec.get("use_base_reserve", False)),
                    base_reserve_ratio=float(spec.get("base_reserve_ratio", 0.30)),
                    use_dual_channel=bool(spec.get("candidate_topm")),
                    use_relation_specific_denoise=bool(spec.get("use_relation_specific_denoise", False)),
                    notes="resumed_from_existing_output",
                )
            )
            if exp_name == "K0_D1Main":
                k0_val_recall = float(existing_root.get("val_recall", 0.0))
            continue

        if not spec.get("use_bandit"):
            graph_row, edge_frames = _train_one_experiment(
                exp_dir=exp_dir,
                exp_name=exp_name,
                base_artifacts=artifacts,
                topk_mode=spec["topk_mode"],
                relation_k=spec["relation_k"],
                alpha_abnormal=args.alpha_abnormal,
                beta_tns=args.beta_tns,
                gamma_interaction=args.gamma_interaction,
                abnormal_score_source=args.abnormal_score_source,
                seed=args.seed,
                preserve_d1_for_fixed_k0=(exp_name == "K0_D1Main"),
                candidate_topm=spec.get("candidate_topm"),
                use_relation_specific_denoise=bool(spec.get("use_relation_specific_denoise", False)),
                use_base_reserve=bool(spec.get("use_base_reserve", False)),
                base_reserve_ratio=float(spec.get("base_reserve_ratio", 0.30)),
            )
            if exp_name == "K0_D1Main":
                k0_val_recall = float(graph_row.get("val_recall", 0.0))
            route_rows.append(
                _build_routek_summary_row(
                    exp_name=exp_name,
                    metrics_row=graph_row,
                    feature_source="D1 llm_masked_logic self-feature matrix from user_scores_enriched.csv + logic_vectors/user_abnormal_vectors.npy via build_self_feature_matrix",
                    relation_k=spec["relation_k"],
                    topk_mode=spec["topk_mode"],
                    use_bandit=False,
                    selected_arm_id=None,
                    candidate_topm=spec.get("candidate_topm"),
                    use_recall_constrained_reward=False,
                    use_base_reserve=bool(spec.get("use_base_reserve", False)),
                    base_reserve_ratio=float(spec.get("base_reserve_ratio", 0.30)),
                    use_dual_channel=bool(spec.get("candidate_topm")),
                    use_relation_specific_denoise=bool(spec.get("use_relation_specific_denoise", False)),
                    notes="K2S mechanism upgrade" if exp_name == "K2S_D1Main" else "",
                )
            )
            logger.info("%s done auc=%s ap=%s f1=%s", exp_name, graph_row.get("auc"), graph_row.get("ap"), graph_row.get("f1"))
            continue

        arm_rows = []
        arm_dirs = {}
        for arm in K4_ARMS:
            arm_id = int(arm["arm_id"])
            relation_k = {k: int(v) for k, v in arm.items() if k != "arm_id"}
            arm_dir = exp_dir / f"arm_{arm_id:02d}" / "warmup"
            arm_dirs[arm_id] = arm_dir
            existing_warmup = _load_completed_experiment_row(arm_dir)
            if existing_warmup is not None:
                graph_row = existing_warmup
            else:
                graph_row, _ = _train_one_experiment(
                    exp_dir=arm_dir,
                    exp_name=f"{exp_name}_arm_{arm_id:02d}_warmup",
                    base_artifacts=artifacts,
                    topk_mode=spec["topk_mode"],
                    relation_k=relation_k,
                    alpha_abnormal=args.alpha_abnormal,
                    beta_tns=args.beta_tns,
                    gamma_interaction=args.gamma_interaction,
                    abnormal_score_source=args.abnormal_score_source,
                    seed=args.seed,
                    candidate_topm=spec.get("candidate_topm"),
                    use_relation_specific_denoise=bool(spec.get("use_relation_specific_denoise", False)),
                    use_base_reserve=bool(spec.get("use_base_reserve", False)),
                    base_reserve_ratio=float(spec.get("base_reserve_ratio", 0.30)),
                    max_epochs_override=args.k5_warmup_epochs if spec.get("is_k5") else args.k4_warmup_epochs,
                    patience_override=max(3, (args.k5_warmup_epochs if spec.get("is_k5") else args.k4_warmup_epochs) // 2),
                )
            arm_rows.append(
                {
                    "arm_id": arm_id,
                    "UPU_k": relation_k["UPU"],
                    "UTU_k": relation_k["UTU"],
                    "USU_k": relation_k["USU"],
                    "LogicAE_CB_k": relation_k["LogicAE_CB"],
                    "num_edges": int(sum(len(pd.read_csv(arm_dir / "edges" / f"{name}_edges.csv")) for name in ["UPU", "UTU", "USU", "LogicAE_CB"] if (arm_dir / "edges" / f"{name}_edges.csv").exists())),
                    "warmup_val_auc": graph_row.get("val_auc"),
                    "warmup_val_ap": graph_row.get("val_ap"),
                    "warmup_val_f1": graph_row.get("val_f1"),
                    "warmup_val_f2": float(fbeta_score(np.asarray([0, 1]), np.asarray([0, 1]), beta=2.0)) if False else graph_row.get("val_f2", 0.0),
                    "warmup_val_recall": graph_row.get("val_recall"),
                    "warmup_val_precision": graph_row.get("val_precision"),
                    "selected_for_full_train": False,
                }
            )
        arm_df = pd.DataFrame(arm_rows)
        arm0_edges = max(float(arm_df.loc[arm_df["arm_id"] == 0, "num_edges"].iloc[0]), 1.0)
        arm_df["edge_density_norm"] = arm_df["num_edges"].astype(float) / arm0_edges

        if spec.get("is_k5"):
            if k0_val_recall is None:
                raise RuntimeError("K5 requires K0 val recall to be available.")
            rewards = []
            penalties = []
            for row in arm_df.itertuples(index=False):
                reward, penalty = _k5_reward(
                    val_auc=float(row.warmup_val_auc or 0.0),
                    val_ap=float(row.warmup_val_ap or 0.0),
                    val_f2=float(row.warmup_val_f2 or 0.0),
                    edge_density_norm=float(row.edge_density_norm),
                    lambda_density=args.bandit_lambda_density,
                    k0_val_recall=float(k0_val_recall),
                    val_recall=float(row.warmup_val_recall or 0.0),
                )
                rewards.append(reward)
                penalties.append(penalty)
            arm_df["warmup_recall_drop"] = np.maximum(0.0, float(k0_val_recall) - arm_df["warmup_val_recall"].astype(float))
            arm_df["warmup_reward"] = rewards
            arm_df["warmup_recall_drop_penalty"] = penalties
        else:
            arm_df["warmup_reward"] = arm_df.apply(
                lambda row: _k4_reward(row["warmup_val_auc"], row["warmup_val_ap"], row["edge_density_norm"], args.bandit_lambda_density),
                axis=1,
            )

        top3 = arm_df.sort_values("warmup_reward", ascending=False).head(3)["arm_id"].astype(int).tolist()
        for arm_id in top3:
            arm_df.loc[arm_df["arm_id"] == arm_id, "selected_for_full_train"] = True
            arm = next(item for item in K4_ARMS if int(item["arm_id"]) == int(arm_id))
            relation_k = {k: int(v) for k, v in arm.items() if k != "arm_id"}
            full_dir = exp_dir / f"arm_{arm_id:02d}" / "full"
            existing_full = _load_completed_experiment_row(full_dir)
            if existing_full is not None:
                graph_row = existing_full
            else:
                graph_row, _ = _train_one_experiment(
                    exp_dir=full_dir,
                    exp_name=f"{exp_name}_arm_{arm_id:02d}_full",
                    base_artifacts=artifacts,
                    topk_mode=spec["topk_mode"],
                    relation_k=relation_k,
                    alpha_abnormal=args.alpha_abnormal,
                    beta_tns=args.beta_tns,
                    gamma_interaction=args.gamma_interaction,
                    abnormal_score_source=args.abnormal_score_source,
                    seed=args.seed,
                    candidate_topm=spec.get("candidate_topm"),
                    use_relation_specific_denoise=bool(spec.get("use_relation_specific_denoise", False)),
                    use_base_reserve=bool(spec.get("use_base_reserve", False)),
                    base_reserve_ratio=float(spec.get("base_reserve_ratio", 0.30)),
                )
            if spec.get("is_k5"):
                reward, penalty = _k5_reward(
                    val_auc=float(graph_row.get("val_auc", 0.0)),
                    val_ap=float(graph_row.get("val_ap", 0.0)),
                    val_f2=float(graph_row.get("val_f2", 0.0)),
                    edge_density_norm=float(arm_df.loc[arm_df["arm_id"] == arm_id, "edge_density_norm"].iloc[0]),
                    lambda_density=args.bandit_lambda_density,
                    k0_val_recall=float(k0_val_recall),
                    val_recall=float(graph_row.get("val_recall", 0.0)),
                )
                arm_df.loc[arm_df["arm_id"] == arm_id, ["full_train_val_auc", "full_train_val_ap", "full_train_val_f1", "full_train_val_f2", "full_train_val_recall", "full_train_val_precision", "full_train_recall_drop", "full_train_reward", "test_auc", "test_ap", "test_f1", "test_recall", "test_precision"]] = [
                    graph_row.get("val_auc"),
                    graph_row.get("val_ap"),
                    graph_row.get("val_f1"),
                    graph_row.get("val_f2"),
                    graph_row.get("val_recall"),
                    graph_row.get("val_precision"),
                    max(0.0, float(k0_val_recall) - float(graph_row.get("val_recall", 0.0))),
                    reward,
                    graph_row.get("auc"),
                    graph_row.get("ap"),
                    graph_row.get("f1"),
                    graph_row.get("recall"),
                    graph_row.get("precision"),
                ]
            else:
                reward = _k4_reward(
                    graph_row.get("val_auc", 0.0),
                    graph_row.get("val_ap", 0.0),
                    float(arm_df.loc[arm_df["arm_id"] == arm_id, "edge_density_norm"].iloc[0]),
                    args.bandit_lambda_density,
                )
                arm_df.loc[arm_df["arm_id"] == arm_id, ["full_train_val_auc", "full_train_val_ap", "full_train_reward", "test_auc", "test_ap", "test_f1", "test_recall", "test_precision"]] = [
                    graph_row.get("val_auc"),
                    graph_row.get("val_ap"),
                    reward,
                    graph_row.get("auc"),
                    graph_row.get("ap"),
                    graph_row.get("f1"),
                    graph_row.get("recall"),
                    graph_row.get("precision"),
                ]

        if spec.get("is_k5"):
            eligible = arm_df.loc[arm_df["selected_for_full_train"] & (arm_df["full_train_recall_drop"].fillna(999.0) <= 0.02)].copy()
            if eligible.empty:
                min_drop = arm_df.loc[arm_df["selected_for_full_train"], "full_train_recall_drop"].min()
                eligible = arm_df.loc[arm_df["selected_for_full_train"] & (arm_df["full_train_recall_drop"] == min_drop)].copy()
                selected = eligible.sort_values("full_train_val_ap", ascending=False).iloc[0].to_dict()
            else:
                selected = eligible.sort_values("full_train_reward", ascending=False).iloc[0].to_dict()
            search_name = "k5_rns_arm_search.csv"
        else:
            selected = arm_df.sort_values("full_train_reward", ascending=False).iloc[0].to_dict()
            search_name = "bandit_arm_search.csv"

        (exp_dir / "metrics").mkdir(parents=True, exist_ok=True)
        arm_df.to_csv(exp_dir / "metrics" / search_name, index=False)
        selected_arm_id = int(selected["arm_id"])
        selected_relation_k = {
            "UPU": int(selected["UPU_k"]),
            "UTU": int(selected["UTU_k"]),
            "USU": int(selected["USU_k"]),
            "LogicAE_CB": int(selected["LogicAE_CB_k"]),
        }
        selected_src_dir = exp_dir / f"arm_{selected_arm_id:02d}" / "full"
        _copy_selected_artifacts(selected_src_dir, exp_dir)

        if spec.get("is_k5"):
            pd.DataFrame(
                [
                    {
                        "reserve_ratio": float(spec.get("base_reserve_ratio", 0.30)),
                        "selected_arm_id": selected_arm_id,
                        "selected_num_edges": int(selected["num_edges"]),
                    }
                ]
            ).to_csv(exp_dir / "metrics" / "k5_base_reserve_stats.csv", index=False)
            neighbor_stats_path = selected_src_dir / "metrics" / "k2s_relation_specific_stats.csv"
            if neighbor_stats_path.exists():
                shutil.copy2(neighbor_stats_path, exp_dir / "metrics" / "k5_neighbor_distance_stats.csv")
            overlap_path = selected_src_dir / "metrics" / "topk_selection_overlap.csv"
            if overlap_path.exists():
                shutil.copy2(overlap_path, exp_dir / "metrics" / "k5_selection_overlap.csv")

        route_rows.append(
            _build_routek_summary_row(
                exp_name=exp_name,
                metrics_row={
                    "auc": selected.get("test_auc"),
                    "ap": selected.get("test_ap"),
                    "f1": selected.get("test_f1"),
                    "recall": selected.get("test_recall"),
                    "precision": selected.get("test_precision"),
                    "num_edges": selected.get("num_edges"),
                    "threshold": "UNKNOWN_FROM_D1",
                    "best_epoch": "UNKNOWN_FROM_D1",
                },
                feature_source="D1 llm_masked_logic self-feature matrix from user_scores_enriched.csv + logic_vectors/user_abnormal_vectors.npy via build_self_feature_matrix",
                relation_k=selected_relation_k,
                topk_mode=spec["topk_mode"],
                use_bandit=True,
                selected_arm_id=selected_arm_id,
                candidate_topm=spec.get("candidate_topm"),
                use_recall_constrained_reward=bool(spec.get("is_k5")),
                use_base_reserve=bool(spec.get("use_base_reserve", False)),
                base_reserve_ratio=float(spec.get("base_reserve_ratio", 0.30)),
                use_dual_channel=bool(spec.get("candidate_topm")),
                use_relation_specific_denoise=bool(spec.get("use_relation_specific_denoise", False)),
                notes="K5 recall-constrained RNS-style arm selection" if spec.get("is_k5") else "K4 legacy bandit arm selection",
            )
        )
        save_json(
            exp_dir / "run_summary.json",
            {
                "route": "K_D1MAIN_RNS",
                "experiment_name": exp_name,
                "selected_arm_id": selected_arm_id,
                "selected_relation_k": selected_relation_k,
                "use_recall_constrained_reward": bool(spec.get("is_k5")),
                "reference_dir": str(REFERENCE_D1_DIR),
            },
        )
        logger.info("%s done selected_arm=%s auc=%s ap=%s f1=%s", exp_name, selected_arm_id, selected.get("test_auc"), selected.get("test_ap"), selected.get("test_f1"))

    summary_df = pd.DataFrame([reference_row, *route_rows])
    summary_df.to_csv(output_root / "routeK_d1main_rns_summary.csv", index=False)
    (output_root / "routeK_d1main_rns_summary.md").write_text(summary_df.to_csv(index=False), encoding="utf-8")
    save_json(
        output_root / "routeK_d1main_rns_run_summary.json",
        {
            "timestamp": timestamp,
            "seed": args.seed,
            "reference_dir": str(REFERENCE_D1_DIR),
            "output_dir": str(output_root),
        },
    )


if __name__ == "__main__":
    main()
