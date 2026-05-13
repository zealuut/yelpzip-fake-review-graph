from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from graph.graph_pipeline import build_self_feature_matrix, compute_edge_stats
from graph.relation_model import run_relation_aggregation_experiments
from graph.scripts.route_runner import _load_base_artifacts

MAIN_PROJECT_ROOT = Path("/home/xyz/HuChao (2)/Bert-TextClassification")
BASE_PROTOCOL_DIR = MAIN_PROJECT_ROOT / "graph" / "outputs" / "yelpzip_balanced_current_graph_no_reweight_20260502_160620"
REFERENCE_D1_DIR = MAIN_PROJECT_ROOT / "graph" / "outputs" / "routeD_tns_guided_logic_egat_20260504_200855" / "D1_EGAT_Base_LogicAE_CB"
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "d1_kattach_fixed_prefix.json"
RELATIONS = ["UPU", "UTU", "USU", "LogicAE_CB"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--config_path", default=str(DEFAULT_CONFIG))
    parser.add_argument("--k4_warmup_epochs", type=int, default=15)
    parser.add_argument("--bandit_lambda_density", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke_edges_only", action="store_true")
    return parser.parse_args()


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_completed_row(exp_dir: Path) -> dict | None:
    summary_path = exp_dir / "run_summary.json"
    if summary_path.exists():
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            row = payload.get("best_graph_model")
            if isinstance(row, dict):
                return row
        except Exception as exc:
            print(f"[resume] could not read {summary_path}: {exc}", flush=True)
    csv_path = exp_dir / "metrics" / "model_results.csv"
    if not csv_path.exists():
        return None
    try:
        result_df = pd.read_csv(csv_path)
        if result_df.empty:
            return None
        if "edge_set" in result_df.columns:
            base_rows = result_df.loc[result_df["edge_set"] == "Base_LogicAE_CB"]
            if not base_rows.empty:
                return base_rows.iloc[-1].to_dict()
        return result_df.iloc[-1].to_dict()
    except Exception as exc:
        print(f"[resume] could not read {csv_path}: {exc}", flush=True)
        return None


def _count_relation_edges(exp_dir: Path) -> int:
    total = 0
    for relation in RELATIONS:
        path = exp_dir / "edges" / f"{relation}_edges.csv"
        if path.exists():
            total += len(pd.read_csv(path))
    return total


def _load_assets() -> dict:
    artifacts = _load_base_artifacts(BASE_PROTOCOL_DIR)
    d1_summary = json.loads((REFERENCE_D1_DIR / "run_summary.json").read_text(encoding="utf-8"))
    user_df = artifacts["user_df"].copy()
    review_scores_df = artifacts["review_scores_df"].copy()
    self_features = build_self_feature_matrix(user_df, artifacts["user_abnormal_vectors"])
    d1_edge_frames = {
        relation: pd.read_csv(REFERENCE_D1_DIR / "edges" / f"{relation}_edges.csv")
        for relation in RELATIONS
    }
    return {
        "user_df": user_df,
        "review_scores_df": review_scores_df,
        "user_text_vectors": artifacts["user_text_vectors"],
        "user_abnormal_vectors": artifacts["user_abnormal_vectors"],
        "self_features": self_features,
        "d1_edge_frames": d1_edge_frames,
        "d1_best": d1_summary["best_graph_model"],
    }


def _reward(val_auc: float, val_ap: float, edge_density_norm: float, lambda_density: float) -> float:
    return float(val_auc) + 0.5 * float(val_ap) - float(lambda_density) * float(edge_density_norm)


def _validate_relation_k(relation_k: dict[str, int]) -> dict[str, int]:
    cleaned = {relation: int(relation_k.get(relation, 20)) for relation in RELATIONS}
    for relation, k in cleaned.items():
        if k < 1:
            raise ValueError(f"{relation} k must be >= 1, got {k}")
        if k > 20:
            raise ValueError(
                f"{relation} k={k} is invalid for fixed-prefix mode: D1 edge files are top20 only."
            )
    return cleaned


def _build_d1_fixed_prefix_frames(
    *,
    assets: dict,
    output_dir: Path,
    relation_k: dict[str, int],
    experiment_name: str,
) -> dict[str, pd.DataFrame]:
    relation_k = _validate_relation_k(relation_k)
    edge_dir = output_dir / "edges"
    metrics_dir = output_dir / "metrics"
    edge_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    user_df = assets["user_df"]
    user_label_map = user_df.set_index(user_df["user_id"].astype(str))["user_label"].astype(int).to_dict()
    all_user_ids = set(user_df["user_id"].astype(str))
    selected_frames: dict[str, pd.DataFrame] = {}
    quality_rows = []
    rank_rows = []
    degree_rows = []
    overlap_rows = []
    sanity_rows = []

    for relation in RELATIONS:
        d1_frame = assets["d1_edge_frames"][relation].copy()
        if d1_frame.empty:
            selected = d1_frame.copy()
        else:
            d1_frame["_d1_order"] = np.arange(len(d1_frame), dtype=np.int64)
            selected = (
                d1_frame.groupby("src_user_id", sort=False, group_keys=False)
                .head(int(relation_k[relation]))
                .sort_values("_d1_order", kind="mergesort")
                .drop(columns=["_d1_order"])
                .reset_index(drop=True)
            )
        selected_frames[relation] = selected
        selected.to_csv(edge_dir / f"{relation}_edges.csv", index=False)

        d1_pairs = set(zip(d1_frame["src_user_id"].astype(str), d1_frame["dst_user_id"].astype(str))) if not d1_frame.empty else set()
        new_pairs = set(zip(selected["src_user_id"].astype(str), selected["dst_user_id"].astype(str))) if not selected.empty else set()
        degree_counter = Counter(selected["src_user_id"].astype(str).tolist()) if not selected.empty else Counter()
        selected_src = set(degree_counter.keys())
        num_edges = int(len(selected))
        base_score = pd.to_numeric(selected.get("edge_weight", 0.0), errors="coerce").fillna(0.0)

        fake_fake = 0
        fake_real = 0
        real_real = 0
        for edge in selected.itertuples(index=False):
            src_label = int(user_label_map.get(str(edge.src_user_id), 0))
            dst_label = int(user_label_map.get(str(edge.dst_user_id), 0))
            if src_label == 1 and dst_label == 1:
                fake_fake += 1
            elif src_label == 0 and dst_label == 0:
                real_real += 1
            else:
                fake_real += 1

        quality_rows.append(
            {
                "relation": relation,
                "k": int(relation_k[relation]),
                "candidate_topM": 20,
                "num_edges": num_edges,
                "avg_degree": float(np.mean(list(degree_counter.values())) if degree_counter else 0.0),
                "isolated_user_count": int(len(all_user_ids - selected_src)),
                "same_label_ratio": float((fake_fake + real_real) / max(num_edges, 1)),
                "fake_fake_ratio": float(fake_fake / max(num_edges, 1)),
                "fake_real_ratio": float(fake_real / max(num_edges, 1)),
                "real_real_ratio": float(real_real / max(num_edges, 1)),
                "avg_base_score": float(base_score.mean()) if num_edges else 0.0,
                "avg_abnormal_pair": 0.0,
                "avg_tns_score": 0.0,
                "avg_rank_score": float(base_score.mean()) if num_edges else 0.0,
            }
        )
        rank_rows.append(
            {
                "relation": relation,
                "k": int(relation_k[relation]),
                "candidate_topM": 20,
                "min_rank_score": float(base_score.min()) if num_edges else 0.0,
                "mean_rank_score": float(base_score.mean()) if num_edges else 0.0,
                "max_rank_score": float(base_score.max()) if num_edges else 0.0,
                "mean_base_score": float(base_score.mean()) if num_edges else 0.0,
                "mean_abnormal_pair": 0.0,
                "mean_tns_score": 0.0,
            }
        )
        degree_rows.append(
            {
                "relation": relation,
                "k": int(relation_k[relation]),
                "candidate_topM": 20,
                "num_edges": num_edges,
                "avg_degree": float(np.mean(list(degree_counter.values())) if degree_counter else 0.0),
                "isolated_user_count": int(len(all_user_ids - selected_src)),
            }
        )
        overlap_rows.append(
            {
                "relation": relation,
                "candidate_topM": 20,
                "final_k": int(relation_k[relation]),
                "overlap_with_K0_top20": float(len(d1_pairs & new_pairs) / max(len(d1_pairs), 1)),
                "new_edges_ratio": float(len(new_pairs - d1_pairs) / max(len(new_pairs), 1)),
                "dropped_edges_ratio": float(len(d1_pairs - new_pairs) / max(len(d1_pairs), 1)),
            }
        )
        sanity_rows.append(
            {
                "relation": relation,
                "requested_k": int(relation_k[relation]),
                "selected_edges": num_edges,
                "d1_top20_edges": int(len(d1_pairs)),
                "selected_is_subset_of_d1_top20": bool(new_pairs.issubset(d1_pairs)),
                "new_edges_outside_d1_top20": int(len(new_pairs - d1_pairs)),
                "ranking_source": "original_d1_edge_file_order",
            }
        )

    pd.DataFrame(quality_rows).to_csv(metrics_dir / "topk_edge_quality_by_relation.csv", index=False)
    pd.DataFrame(rank_rows).to_csv(metrics_dir / "topk_rank_score_stats.csv", index=False)
    pd.DataFrame(degree_rows).to_csv(metrics_dir / "topk_relation_degree_stats.csv", index=False)
    pd.DataFrame(overlap_rows).to_csv(metrics_dir / "topk_selection_overlap.csv", index=False)
    pd.DataFrame(sanity_rows).to_csv(metrics_dir / "d1_fixed_prefix_sanity.csv", index=False)
    _save_json(
        edge_dir / "edge_build_config.json",
        {
            "experiment_name": experiment_name,
            "graph_mode": "current",
            "topk_mode": "d1_fixed_prefix",
            "relation_k": relation_k,
            "candidate_topM": {relation: 20 for relation in RELATIONS},
            "is_d1_fixed_prefix": True,
            "ranking_source": "original_d1_edge_file_order",
            "allows_new_edges_outside_d1_top20": False,
            "notes": "Select head(k) per src_user_id from D1 top20 edge files. No candidate-pool rebuild and no rerank.",
        },
    )
    return selected_frames


def _run_graph(
    *,
    exp_dir: Path,
    assets: dict,
    experiment_name: str,
    relation_k: dict[str, int],
    seed: int,
    max_epochs_override: int | None = None,
    patience_override: int | None = None,
    edge_only: bool = False,
) -> dict:
    exp_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = exp_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    relation_k = _validate_relation_k(relation_k)
    edge_frames = _build_d1_fixed_prefix_frames(
        assets=assets,
        output_dir=exp_dir,
        relation_k=relation_k,
        experiment_name=experiment_name,
    )
    compute_edge_stats(edge_frames=edge_frames, user_df=assets["user_df"], output_dir=exp_dir)

    if edge_only:
        row = {
            "edge_set": "Base_LogicAE_CB",
            "val_auc": np.nan,
            "val_ap": np.nan,
            "auc": np.nan,
            "ap": np.nan,
            "f1": np.nan,
            "recall": np.nan,
            "precision": np.nan,
            "threshold": np.nan,
            "num_edges": _count_relation_edges(exp_dir),
        }
    else:
        result_df = run_relation_aggregation_experiments(
            user_df=assets["user_df"],
            self_features=assets["self_features"],
            edge_frames=edge_frames,
            output_dir=metrics_dir,
            review_encoder_name="llm_masked_logic",
            model_kind="edge_aware_gat",
            seed=seed,
            backbone="current_egat",
            relation_model="edge_aware_gat",
            use_abnormal_edge_weight=False,
            use_abnormal_gate=False,
            use_abnormal_value_gate=False,
            use_abnormal_attention_bias=False,
            abnormal_score_source="auto",
            abnormal_edge_lambda=1.0,
            abnormal_edge_eta=0.5,
            abnormal_gate_eta=0.5,
            abnormal_pair_mode="both_high",
            abnormal_gate_learnable=False,
            abnormal_attention_gamma=1.0,
            review_scores_df=assets["review_scores_df"],
            selected_edge_set="Base_LogicAE_CB",
            relation_topk=None,
            use_node_gat=False,
            max_epochs_override=max_epochs_override,
            patience_override=patience_override,
        )
        row = result_df.loc[result_df["edge_set"] == "Base_LogicAE_CB"].iloc[0].to_dict()

    _save_json(
        exp_dir / "run_summary.json",
        {
            "experiment_name": experiment_name,
            "implementation": "routeK_d1_fixed_prefix",
            "best_graph_model": row,
            "strategy": "d1_fixed_prefix",
            "relation_k": relation_k,
            "semantic_contract": "top-k is head(k) of the original D1 top20 edge order per src_user_id.",
        },
    )
    _save_json(
        exp_dir / "config.json",
        {
            "experiment_name": experiment_name,
            "graph_mode": "current",
            "edge_set": "Base_LogicAE_CB",
            "model_backbone": "current_egat",
            "relation_model": "edge_aware_gat",
            "topk_mode": "d1_fixed_prefix",
            "relation_k": relation_k,
            "implementation": "routeK_d1_fixed_prefix",
            "semantic_contract": "No candidate-pool rebuild, no rerank, no edge outside D1 top20.",
        },
    )
    return row


def _run_or_resume_graph(*, resume: bool, exp_dir: Path, experiment_name: str, **kwargs) -> dict:
    if resume:
        row = _load_completed_row(exp_dir)
        if row is not None:
            print(f"[resume] skip completed {experiment_name}: {exp_dir}", flush=True)
            return row
        print(f"[resume] missing completed summary; running {experiment_name}: {exp_dir}", flush=True)
    return _run_graph(exp_dir=exp_dir, experiment_name=experiment_name, **kwargs)


def main() -> None:
    args = parse_args()
    cfg = json.loads(Path(args.config_path).read_text(encoding="utf-8"))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"routeK fixed-prefix output_root={output_root} resume={args.resume}", flush=True)
    assets = _load_assets()

    if args.smoke_edges_only:
        smoke_k = {key: int(value) for key, value in cfg["fixed_prefix_probe_relation_k"].items()}
        row = _run_graph(
            exp_dir=output_root / "SMOKE_D1_FixedPrefix",
            assets=assets,
            experiment_name="SMOKE_D1_FixedPrefix",
            relation_k=smoke_k,
            seed=args.seed,
            edge_only=True,
        )
        print(f"smoke_edges_only completed num_edges={row['num_edges']}", flush=True)
        return

    rows = [
        {
            "experiment_name": "D1_EGAT_Base_LogicAE_CB",
            "strategy": "reference_only",
            "selected_arm_id": "",
            "UPU_k": 20,
            "UTU_k": 20,
            "USU_k": 20,
            "LogicAE_CB_k": 20,
            "AUC": assets["d1_best"]["auc"],
            "AP": assets["d1_best"]["ap"],
            "F1": assets["d1_best"]["f1"],
            "Recall": assets["d1_best"]["recall"],
            "Precision": assets["d1_best"]["precision"],
            "test_threshold": assets["d1_best"]["threshold"],
            "notes": "reference row from D1 output",
        }
    ]

    probe_k = {key: int(value) for key, value in cfg["fixed_prefix_probe_relation_k"].items()}
    probe_dir = output_root / "D1_FixedPrefixK10"
    probe_row = _run_or_resume_graph(
        resume=args.resume,
        exp_dir=probe_dir,
        assets=assets,
        experiment_name="D1_FixedPrefixK10",
        relation_k=probe_k,
        seed=args.seed,
    )
    rows.append(
        {
            "experiment_name": "D1_FixedPrefixK10",
            "strategy": "fixed_prefix_probe",
            "selected_arm_id": "",
            "UPU_k": probe_k["UPU"],
            "UTU_k": probe_k["UTU"],
            "USU_k": probe_k["USU"],
            "LogicAE_CB_k": probe_k["LogicAE_CB"],
            "AUC": probe_row["auc"],
            "AP": probe_row["ap"],
            "F1": probe_row["f1"],
            "Recall": probe_row["recall"],
            "Precision": probe_row["precision"],
            "test_threshold": probe_row["threshold"],
            "notes": "head(k) from original D1 top20 order; no new edges",
        }
    )

    k4_dir = output_root / "D1_K4_FixedPrefixBandit"
    warm_rows = []
    for arm in cfg["k4_arms"]:
        arm_id = int(arm["arm_id"])
        relation_k = {key: int(value) for key, value in arm.items() if key != "arm_id"}
        arm_dir = k4_dir / f"arm_{arm_id:02d}" / "warmup"
        row = _run_or_resume_graph(
            resume=args.resume,
            exp_dir=arm_dir,
            assets=assets,
            experiment_name=f"D1_K4_fixed_prefix_arm_{arm_id:02d}_warmup",
            relation_k=relation_k,
            seed=args.seed,
            max_epochs_override=args.k4_warmup_epochs,
            patience_override=max(3, args.k4_warmup_epochs // 2),
        )
        warm_rows.append(
            {
                "arm_id": arm_id,
                "UPU_k": relation_k["UPU"],
                "UTU_k": relation_k["UTU"],
                "USU_k": relation_k["USU"],
                "LogicAE_CB_k": relation_k["LogicAE_CB"],
                "num_edges": _count_relation_edges(arm_dir),
                "warmup_val_auc": row.get("val_auc"),
                "warmup_val_ap": row.get("val_ap"),
            }
        )

    arm_df = pd.DataFrame(warm_rows)
    arm0_edges = max(float(arm_df.loc[arm_df["arm_id"] == 0, "num_edges"].iloc[0]), 1.0)
    arm_df["edge_density_norm"] = arm_df["num_edges"].astype(float) / arm0_edges
    arm_df["warmup_reward"] = arm_df.apply(
        lambda r: _reward(r["warmup_val_auc"], r["warmup_val_ap"], r["edge_density_norm"], args.bandit_lambda_density),
        axis=1,
    )
    top3 = arm_df.sort_values("warmup_reward", ascending=False).head(3)["arm_id"].astype(int).tolist()
    arm_df["selected_for_full_train"] = arm_df["arm_id"].isin(top3)

    full_rows = []
    for arm_id in top3:
        arm = next(item for item in cfg["k4_arms"] if int(item["arm_id"]) == arm_id)
        relation_k = {key: int(value) for key, value in arm.items() if key != "arm_id"}
        full_dir = k4_dir / f"arm_{arm_id:02d}" / "full"
        row = _run_or_resume_graph(
            resume=args.resume,
            exp_dir=full_dir,
            assets=assets,
            experiment_name=f"D1_K4_fixed_prefix_arm_{arm_id:02d}_full",
            relation_k=relation_k,
            seed=args.seed,
        )
        reward = _reward(
            float(row.get("val_auc", 0.0)),
            float(row.get("val_ap", 0.0)),
            float(arm_df.loc[arm_df["arm_id"] == arm_id, "edge_density_norm"].iloc[0]),
            args.bandit_lambda_density,
        )
        full_rows.append(
            {
                "arm_id": arm_id,
                "UPU_k": relation_k["UPU"],
                "UTU_k": relation_k["UTU"],
                "USU_k": relation_k["USU"],
                "LogicAE_CB_k": relation_k["LogicAE_CB"],
                "num_edges": int(arm_df.loc[arm_df["arm_id"] == arm_id, "num_edges"].iloc[0]),
                "edge_density_norm": float(arm_df.loc[arm_df["arm_id"] == arm_id, "edge_density_norm"].iloc[0]),
                "warmup_val_auc": float(arm_df.loc[arm_df["arm_id"] == arm_id, "warmup_val_auc"].iloc[0]),
                "warmup_val_ap": float(arm_df.loc[arm_df["arm_id"] == arm_id, "warmup_val_ap"].iloc[0]),
                "warmup_reward": float(arm_df.loc[arm_df["arm_id"] == arm_id, "warmup_reward"].iloc[0]),
                "selected_for_full_train": True,
                "full_train_val_auc": row.get("val_auc"),
                "full_train_val_ap": row.get("val_ap"),
                "full_train_reward": reward,
                "test_auc": row.get("auc"),
                "test_ap": row.get("ap"),
                "test_f1": row.get("f1"),
                "test_recall": row.get("recall"),
                "test_precision": row.get("precision"),
            }
        )

    full_df = pd.DataFrame(full_rows)
    selected = full_df.sort_values("full_train_reward", ascending=False).iloc[0].to_dict()
    (k4_dir / "metrics").mkdir(parents=True, exist_ok=True)
    arm_df.merge(
        full_df,
        on=[
            "arm_id",
            "UPU_k",
            "UTU_k",
            "USU_k",
            "LogicAE_CB_k",
            "num_edges",
            "edge_density_norm",
            "selected_for_full_train",
        ],
        how="left",
    ).to_csv(k4_dir / "metrics" / "bandit_arm_search.csv", index=False)
    _save_json(
        k4_dir / "run_summary.json",
        {
            "experiment_name": "D1_K4_FixedPrefixBandit",
            "implementation": "routeK_d1_fixed_prefix",
            "selected_arm_id": int(selected["arm_id"]),
            "selected_relation_k": {
                "UPU": int(selected["UPU_k"]),
                "UTU": int(selected["UTU_k"]),
                "USU": int(selected["USU_k"]),
                "LogicAE_CB": int(selected["LogicAE_CB_k"]),
            },
            "semantic_contract": "K4 searches relation-wise k, but every selected edge is head(k) from D1 top20.",
        },
    )
    rows.append(
        {
            "experiment_name": "D1_K4_FixedPrefixBandit",
            "strategy": "K4_fixed_prefix_bandit",
            "selected_arm_id": int(selected["arm_id"]),
            "UPU_k": int(selected["UPU_k"]),
            "UTU_k": int(selected["UTU_k"]),
            "USU_k": int(selected["USU_k"]),
            "LogicAE_CB_k": int(selected["LogicAE_CB_k"]),
            "AUC": selected["test_auc"],
            "AP": selected["test_ap"],
            "F1": selected["test_f1"],
            "Recall": selected["test_recall"],
            "Precision": selected["test_precision"],
            "test_threshold": "UNKNOWN_FROM_D1",
            "notes": "head(k) from original D1 top20 order; no new edges",
        }
    )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_root / "routeD1_kattach_fixed_prefix_summary.csv", index=False)
    (output_root / "routeD1_kattach_fixed_prefix_summary.md").write_text(summary_df.to_csv(index=False), encoding="utf-8")


if __name__ == "__main__":
    main()

