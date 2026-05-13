from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from graph.graph_pipeline import (
    _routek_pair_abnormal_lookup,
    _routek_tns_lookup,
    _undirected_pair_key,
    build_self_feature_matrix,
    compute_edge_stats,
)
from graph.relation_model import run_relation_aggregation_experiments
from graph.scripts.route_runner import _load_base_artifacts

MAIN_PROJECT_ROOT = Path("/home/xyz/HuChao (2)/Bert-TextClassification")
BASE_PROTOCOL_DIR = MAIN_PROJECT_ROOT / "graph" / "outputs" / "yelpzip_balanced_current_graph_no_reweight_20260502_160620"
REFERENCE_D1_DIR = MAIN_PROJECT_ROOT / "graph" / "outputs" / "routeD_tns_guided_logic_egat_20260504_200855" / "D1_EGAT_Base_LogicAE_CB"
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "d1_k2soft_faithful.json"
RELATIONS = ["UPU", "UTU", "USU", "LogicAE_CB"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--config_path", default=str(DEFAULT_CONFIG))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke_edges_only", action="store_true")
    return parser.parse_args()


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_completed_row(exp_dir: Path) -> dict[str, Any] | None:
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


def _load_assets() -> dict[str, Any]:
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


def _hash_directed_pairs(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    if frame.empty:
        return digest.hexdigest()
    for row in frame[["src_user_id", "dst_user_id"]].astype(str).itertuples(index=False):
        digest.update(f"{row.src_user_id}\t{row.dst_user_id}\n".encode("utf-8"))
    return digest.hexdigest()


def _pair_set(frame: pd.DataFrame) -> set[tuple[str, str]]:
    if frame.empty:
        return set()
    return set(zip(frame["src_user_id"].astype(str), frame["dst_user_id"].astype(str)))


def _build_soft_context(
    *,
    assets: dict[str, Any],
    abnormal_score_source: str,
    tns_phi_days: int,
) -> dict[str, Any]:
    _, pair_abnormal_fn = _routek_pair_abnormal_lookup(
        user_df=assets["user_df"],
        review_features=assets["review_scores_df"],
        abnormal_score_source=abnormal_score_source,
    )
    tns_lookup, temporal_lookup, tns_cache = _routek_tns_lookup(
        logic_edges=assets["d1_edge_frames"]["LogicAE_CB"].copy(),
        review_features=assets["review_scores_df"],
        user_df=assets["user_df"],
        user_abnormal_vectors=assets["user_abnormal_vectors"],
        tns_phi_days=tns_phi_days,
    )
    return {
        "pair_abnormal_fn": pair_abnormal_fn,
        "tns_lookup": tns_lookup,
        "temporal_pair_count": len(temporal_lookup),
        "tns_pair_count": len(tns_lookup),
        "tns_cache_keys": sorted(str(key) for key in tns_cache.keys()),
    }


def _soft_weight_factor(
    *,
    soft_mode: str,
    abnormal_pair: pd.Series,
    tns_score: pd.Series,
    alpha_abnormal: float,
    beta_tns: float,
    gamma_interaction: float,
) -> pd.Series:
    interaction = abnormal_pair * tns_score
    if soft_mode == "fixed_original":
        return pd.Series(np.ones(len(abnormal_pair), dtype=np.float32), index=abnormal_pair.index)
    if soft_mode == "abnormal_soft":
        return (1.0 + float(alpha_abnormal) * abnormal_pair).astype(np.float32)
    if soft_mode == "abnormal_tns_soft":
        return (
            (1.0 + float(alpha_abnormal) * abnormal_pair)
            * (1.0 + float(beta_tns) * tns_score)
            * (1.0 + float(gamma_interaction) * interaction)
        ).astype(np.float32)
    raise ValueError(f"Unsupported soft_mode={soft_mode!r}")


def _build_d1_soft_frames(
    *,
    assets: dict[str, Any],
    soft_context: dict[str, Any],
    output_dir: Path,
    experiment_name: str,
    soft_mode: str,
    alpha_abnormal: float,
    beta_tns: float,
    gamma_interaction: float,
    abnormal_score_source: str,
    tns_phi_days: int,
) -> dict[str, pd.DataFrame]:
    edge_dir = output_dir / "edges"
    metrics_dir = output_dir / "metrics"
    edge_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    user_df = assets["user_df"]
    user_label_map = user_df.set_index(user_df["user_id"].astype(str))["user_label"].astype(int).to_dict()
    all_user_ids = set(user_df["user_id"].astype(str))
    pair_abnormal_fn = soft_context["pair_abnormal_fn"]
    tns_lookup = soft_context["tns_lookup"]

    selected_frames: dict[str, pd.DataFrame] = {}
    quality_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    topology_rows: list[dict[str, Any]] = []
    annotation_sample_rows: list[pd.DataFrame] = []

    for relation in RELATIONS:
        d1_frame = assets["d1_edge_frames"][relation].copy()
        work = d1_frame.copy()
        if not work.empty:
            work["src_user_id"] = work["src_user_id"].astype(str)
            work["dst_user_id"] = work["dst_user_id"].astype(str)
            base_score = pd.to_numeric(work.get("edge_weight", 0.0), errors="coerce").fillna(0.0).astype(np.float32)
            abnormal_pair = work.apply(
                lambda row: pair_abnormal_fn(row["src_user_id"], row["dst_user_id"]),
                axis=1,
            ).astype(np.float32)
            tns_score = work.apply(
                lambda row: float(tns_lookup.get(_undirected_pair_key(row["src_user_id"], row["dst_user_id"]), 0.0)),
                axis=1,
            ).astype(np.float32)
            interaction = (abnormal_pair * tns_score).astype(np.float32)
            factor = _soft_weight_factor(
                soft_mode=soft_mode,
                abnormal_pair=abnormal_pair,
                tns_score=tns_score,
                alpha_abnormal=alpha_abnormal,
                beta_tns=beta_tns,
                gamma_interaction=gamma_interaction,
            )
            soft_weight = (base_score * factor).astype(np.float32)
            work["base_score"] = base_score
            work["abnormal_pair"] = abnormal_pair
            work["pair_abnormal_score"] = abnormal_pair
            work["tns_score"] = tns_score
            work["abnormal_tns_interaction"] = interaction
            work["soft_weight_factor"] = factor.astype(np.float32)
            work["rank_score"] = soft_weight
            work["reliability_score"] = soft_weight
            work["edge_weight_before"] = base_score
            work["edge_weight_after"] = soft_weight
            work["edge_weight"] = soft_weight

        # Training must see the exact D1 schema. Audit-only columns such as
        # pair_abnormal_score are valid EGAT edge features, so leaking them into
        # K0 would silently make K0 different from D1.
        model_frame = d1_frame.copy()
        if not model_frame.empty:
            model_frame["src_user_id"] = model_frame["src_user_id"].astype(str)
            model_frame["dst_user_id"] = model_frame["dst_user_id"].astype(str)
            if soft_mode != "fixed_original":
                model_frame["edge_weight"] = work["edge_weight"].to_numpy(dtype=np.float32)

        selected_frames[relation] = model_frame
        model_frame.to_csv(edge_dir / f"{relation}_edges.csv", index=False)
        if not work.empty:
            annotation_cols = [
                "src_user_id",
                "dst_user_id",
                "base_score",
                "abnormal_pair",
                "tns_score",
                "abnormal_tns_interaction",
                "soft_weight_factor",
                "rank_score",
            ]
            sample = work.reindex(columns=annotation_cols).head(200).copy()
            sample.insert(0, "relation", relation)
            annotation_sample_rows.append(sample)

        d1_pairs = _pair_set(d1_frame)
        new_pairs = _pair_set(model_frame)
        degree_counter = Counter(model_frame["src_user_id"].astype(str).tolist()) if not model_frame.empty else Counter()
        fake_fake = 0
        fake_real = 0
        real_real = 0
        for edge in model_frame.itertuples(index=False):
            src_label = int(user_label_map.get(str(edge.src_user_id), 0))
            dst_label = int(user_label_map.get(str(edge.dst_user_id), 0))
            if src_label == 1 and dst_label == 1:
                fake_fake += 1
            elif src_label == 0 and dst_label == 0:
                real_real += 1
            else:
                fake_real += 1

        num_edges = int(len(model_frame))
        base = pd.to_numeric(work.get("base_score", pd.Series(dtype=np.float32)), errors="coerce").fillna(0.0)
        abnormal = pd.to_numeric(work.get("abnormal_pair", pd.Series(dtype=np.float32)), errors="coerce").fillna(0.0)
        tns = pd.to_numeric(work.get("tns_score", pd.Series(dtype=np.float32)), errors="coerce").fillna(0.0)
        factor = pd.to_numeric(work.get("soft_weight_factor", pd.Series(dtype=np.float32)), errors="coerce").fillna(1.0)
        rank = pd.to_numeric(work.get("rank_score", pd.Series(dtype=np.float32)), errors="coerce").fillna(0.0)

        quality_rows.append(
            {
                "relation": relation,
                "num_edges": num_edges,
                "avg_degree": float(np.mean(list(degree_counter.values())) if degree_counter else 0.0),
                "isolated_user_count": int(len(all_user_ids - set(degree_counter.keys()))),
                "same_label_ratio": float((fake_fake + real_real) / max(num_edges, 1)),
                "fake_fake_ratio": float(fake_fake / max(num_edges, 1)),
                "fake_real_ratio": float(fake_real / max(num_edges, 1)),
                "real_real_ratio": float(real_real / max(num_edges, 1)),
                "avg_base_score": float(base.mean()) if num_edges else 0.0,
                "avg_abnormal_pair": float(abnormal.mean()) if num_edges else 0.0,
                "avg_tns_score": float(tns.mean()) if num_edges else 0.0,
                "avg_soft_weight_factor": float(factor.mean()) if num_edges else 1.0,
                "avg_rank_score": float(rank.mean()) if num_edges else 0.0,
            }
        )
        rank_rows.append(
            {
                "relation": relation,
                "min_base_score": float(base.min()) if num_edges else 0.0,
                "mean_base_score": float(base.mean()) if num_edges else 0.0,
                "max_base_score": float(base.max()) if num_edges else 0.0,
                "min_soft_weight_factor": float(factor.min()) if num_edges else 1.0,
                "mean_soft_weight_factor": float(factor.mean()) if num_edges else 1.0,
                "max_soft_weight_factor": float(factor.max()) if num_edges else 1.0,
                "min_rank_score": float(rank.min()) if num_edges else 0.0,
                "mean_rank_score": float(rank.mean()) if num_edges else 0.0,
                "max_rank_score": float(rank.max()) if num_edges else 0.0,
                "mean_abnormal_pair": float(abnormal.mean()) if num_edges else 0.0,
                "mean_tns_score": float(tns.mean()) if num_edges else 0.0,
            }
        )
        topology_rows.append(
            {
                "relation": relation,
                "d1_edges": int(len(d1_frame)),
                "soft_edges": num_edges,
                "directed_pair_overlap": float(len(d1_pairs & new_pairs) / max(len(d1_pairs), 1)),
                "new_edges_ratio": float(len(new_pairs - d1_pairs) / max(len(new_pairs), 1)),
                "dropped_edges_ratio": float(len(d1_pairs - new_pairs) / max(len(d1_pairs), 1)),
                "d1_pair_hash": _hash_directed_pairs(d1_frame),
                "soft_pair_hash": _hash_directed_pairs(model_frame),
                "topology_exact_match": bool(_hash_directed_pairs(d1_frame) == _hash_directed_pairs(model_frame)),
                "model_schema_exact_d1": bool(list(model_frame.columns) == list(d1_frame.columns)),
            }
        )

    pd.DataFrame(quality_rows).to_csv(metrics_dir / "d1_soft_edge_quality_by_relation.csv", index=False)
    pd.DataFrame(rank_rows).to_csv(metrics_dir / "d1_soft_weight_stats.csv", index=False)
    pd.DataFrame(topology_rows).to_csv(metrics_dir / "d1_soft_topology_sanity.csv", index=False)
    if annotation_sample_rows:
        pd.concat(annotation_sample_rows, ignore_index=True).to_csv(
            metrics_dir / "d1_soft_edge_annotation_sample.csv",
            index=False,
        )
    _save_json(
        edge_dir / "edge_build_config.json",
        {
            "experiment_name": experiment_name,
            "graph_mode": "current",
            "topk_mode": "d1_k2soft_faithful",
            "soft_mode": soft_mode,
            "alpha_abnormal": float(alpha_abnormal),
            "beta_tns": float(beta_tns),
            "gamma_interaction": float(gamma_interaction),
            "tns_phi_days": int(tns_phi_days),
            "abnormal_score_source": str(abnormal_score_source),
            "is_strict_d1_topology": True,
            "allows_new_edges_outside_d1": False,
            "selection_policy": "none; edge pairs and order are copied from the reference D1 edge files",
            "weight_policy": "model edge files keep the exact D1 schema; edge_weight is original D1 weight for K0, abnormal-soft for K1, abnormal+TNS-soft for K2",
            "audit_column_policy": "soft diagnostic columns are written only to metrics samples/stats and never passed to the model",
            "soft_context": {
                "tns_pair_count": int(soft_context["tns_pair_count"]),
                "temporal_pair_count": int(soft_context["temporal_pair_count"]),
                "tns_cache_keys": soft_context["tns_cache_keys"],
            },
        },
    )
    return selected_frames


def _count_relation_edges(exp_dir: Path) -> int:
    total = 0
    for relation in RELATIONS:
        path = exp_dir / "edges" / f"{relation}_edges.csv"
        if path.exists():
            total += len(pd.read_csv(path))
    return total


def _run_graph(
    *,
    exp_dir: Path,
    assets: dict[str, Any],
    soft_context: dict[str, Any],
    experiment_name: str,
    strategy: str,
    soft_mode: str,
    alpha_abnormal: float,
    beta_tns: float,
    gamma_interaction: float,
    abnormal_score_source: str,
    tns_phi_days: int,
    seed: int,
    edge_only: bool = False,
) -> dict[str, Any]:
    exp_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = exp_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    edge_frames = _build_d1_soft_frames(
        assets=assets,
        soft_context=soft_context,
        output_dir=exp_dir,
        experiment_name=experiment_name,
        soft_mode=soft_mode,
        alpha_abnormal=alpha_abnormal,
        beta_tns=beta_tns,
        gamma_interaction=gamma_interaction,
        abnormal_score_source=abnormal_score_source,
        tns_phi_days=tns_phi_days,
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
            abnormal_score_source=abnormal_score_source,
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
        )
        row = result_df.loc[result_df["edge_set"] == "Base_LogicAE_CB"].iloc[0].to_dict()

    _save_json(
        exp_dir / "run_summary.json",
        {
            "experiment_name": experiment_name,
            "strategy": strategy,
            "implementation": "routeK_d1_k2soft_faithful",
            "best_graph_model": row,
            "semantic_contract": "D1 topology is copied exactly; only edge_weight changes for K1/K2.",
            "soft_mode": soft_mode,
            "alpha_abnormal": float(alpha_abnormal),
            "beta_tns": float(beta_tns),
            "gamma_interaction": float(gamma_interaction),
            "tns_phi_days": int(tns_phi_days),
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
            "implementation": "routeK_d1_k2soft_faithful",
            "strategy": strategy,
            "soft_mode": soft_mode,
            "semantic_contract": "No candidate pool rebuild, no top-k selection, no rerank; D1 directed pairs and order are preserved.",
        },
    )
    return row


def _run_or_resume_graph(*, resume: bool, exp_dir: Path, experiment_name: str, **kwargs: Any) -> dict[str, Any]:
    if resume:
        row = _load_completed_row(exp_dir)
        if row is not None:
            print(f"[resume] skip completed {experiment_name}: {exp_dir}", flush=True)
            return row
        print(f"[resume] missing completed summary; running {experiment_name}: {exp_dir}", flush=True)
    return _run_graph(exp_dir=exp_dir, experiment_name=experiment_name, **kwargs)


def _summary_row(
    *,
    experiment_name: str,
    strategy: str,
    soft_mode: str,
    row: dict[str, Any],
    d1_best: dict[str, Any],
    notes: str,
) -> dict[str, Any]:
    auc = float(row.get("auc", np.nan))
    ap = float(row.get("ap", np.nan))
    f1 = float(row.get("f1", np.nan))
    return {
        "experiment_name": experiment_name,
        "strategy": strategy,
        "soft_mode": soft_mode,
        "val_auc": row.get("val_auc"),
        "val_ap": row.get("val_ap"),
        "AUC": row.get("auc"),
        "AP": row.get("ap"),
        "F1": row.get("f1"),
        "Recall": row.get("recall"),
        "Precision": row.get("precision"),
        "test_threshold": row.get("threshold"),
        "delta_auc_vs_d1": auc - float(d1_best["auc"]) if np.isfinite(auc) else np.nan,
        "delta_ap_vs_d1": ap - float(d1_best["ap"]) if np.isfinite(ap) else np.nan,
        "delta_f1_vs_d1": f1 - float(d1_best["f1"]) if np.isfinite(f1) else np.nan,
        "notes": notes,
    }


def main() -> None:
    args = parse_args()
    cfg = json.loads(Path(args.config_path).read_text(encoding="utf-8"))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
    alpha_abnormal = float(cfg.get("alpha_abnormal", 0.5))
    beta_tns = float(cfg.get("beta_tns", 0.2))
    gamma_interaction = float(cfg.get("gamma_interaction", 0.2))
    tns_phi_days = int(cfg.get("tns_phi_days", 5))
    abnormal_score_source = str(cfg.get("abnormal_score_source", "auto"))

    print(f"routeK k2soft-faithful output_root={output_root} resume={args.resume}", flush=True)
    print("semantic_contract=D1 topology fixed; only edge_weight changes for K1/K2", flush=True)
    assets = _load_assets()
    soft_context = _build_soft_context(
        assets=assets,
        abnormal_score_source=abnormal_score_source,
        tns_phi_days=tns_phi_days,
    )
    print(
        f"soft_context tns_pair_count={soft_context['tns_pair_count']} temporal_pair_count={soft_context['temporal_pair_count']}",
        flush=True,
    )

    rows: list[dict[str, Any]] = [
        _summary_row(
            experiment_name="D1_EGAT_Base_LogicAE_CB",
            strategy="reference_only",
            soft_mode="reference_only",
            row=assets["d1_best"],
            d1_best=assets["d1_best"],
            notes="reference row from D1 output",
        )
    ]

    if args.smoke_edges_only:
        variants = [item for item in cfg["variants"] if item["soft_mode"] == "abnormal_tns_soft"]
    else:
        variants = cfg["variants"]

    for variant in variants:
        experiment_name = str(variant["experiment_name"])
        strategy = str(variant["strategy"])
        soft_mode = str(variant["soft_mode"])
        exp_dir = output_root / experiment_name
        row = _run_or_resume_graph(
            resume=args.resume,
            exp_dir=exp_dir,
            assets=assets,
            soft_context=soft_context,
            experiment_name=experiment_name,
            strategy=strategy,
            soft_mode=soft_mode,
            alpha_abnormal=alpha_abnormal,
            beta_tns=beta_tns,
            gamma_interaction=gamma_interaction,
            abnormal_score_source=abnormal_score_source,
            tns_phi_days=tns_phi_days,
            seed=seed,
            edge_only=args.smoke_edges_only,
        )
        rows.append(
            _summary_row(
                experiment_name=experiment_name,
                strategy=strategy,
                soft_mode=soft_mode,
                row=row,
                d1_best=assets["d1_best"],
                notes=str(variant.get("notes", "")),
            )
        )
        print(
            f"completed {experiment_name} auc={row.get('auc')} ap={row.get('ap')} f1={row.get('f1')}",
            flush=True,
        )

    summary_df = pd.DataFrame(rows)
    summary_path = output_root / "routeD1_k2soft_faithful_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    _save_json(
        output_root / "run_summary.json",
        {
            "experiment_name": "routeD1_k2soft_faithful",
            "implementation": "routeK_d1_k2soft_faithful",
            "semantic_contract": "D1 topology is copied exactly for K0/K1/K2; K1/K2 only change edge_weight.",
            "config_path": str(args.config_path),
            "summary_csv": str(summary_path),
            "rows": rows,
        },
    )
    print(f"summary_path={summary_path}", flush=True)


if __name__ == "__main__":
    main()
