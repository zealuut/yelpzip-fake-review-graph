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

from graph.graph_pipeline import (  # noqa: E402
    _routek_pair_abnormal_lookup,
    build_edge_frames,
    build_self_feature_matrix,
    compute_edge_stats,
)
from graph.relation_model import run_relation_aggregation_experiments  # noqa: E402
from graph.scripts.route_runner import _load_base_artifacts  # noqa: E402

MAIN_PROJECT_ROOT = Path("/home/xyz/HuChao (2)/Bert-TextClassification")
BASE_PROTOCOL_DIR = MAIN_PROJECT_ROOT / "graph" / "outputs" / "yelpzip_balanced_current_graph_no_reweight_20260502_160620"
REFERENCE_D1_DIR = MAIN_PROJECT_ROOT / "graph" / "outputs" / "routeD_tns_guided_logic_egat_20260504_200855" / "D1_EGAT_Base_LogicAE_CB"
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "d2_abnormal_edge_gate.json"
RELATIONS = ["UPU", "UTU", "USU", "LogicAE_CB"]
MODEL_EXTRA_COLUMNS = ["pair_abnormal_score", "abnormal_score_src", "abnormal_score_dst"]


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
        base_rows = result_df.loc[result_df.get("edge_set", "") == "Base_LogicAE_CB"]
        return (base_rows.iloc[-1] if not base_rows.empty else result_df.iloc[-1]).to_dict()
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
    user_score_map, pair_abnormal_fn = _routek_pair_abnormal_lookup(
        user_df=user_df,
        review_features=review_scores_df,
        abnormal_score_source="auto",
    )
    return {
        "user_df": user_df,
        "review_scores_df": review_scores_df,
        "user_text_vectors": artifacts["user_text_vectors"],
        "user_abnormal_vectors": artifacts["user_abnormal_vectors"],
        "self_features": self_features,
        "d1_edge_frames": d1_edge_frames,
        "d1_best": d1_summary["best_graph_model"],
        "user_score_map": user_score_map,
        "pair_abnormal_fn": pair_abnormal_fn,
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


def _sorted_model_columns(frame: pd.DataFrame, base_columns: list[str]) -> list[str]:
    columns = [column for column in base_columns if column in frame.columns]
    columns.extend(column for column in MODEL_EXTRA_COLUMNS if column in frame.columns and column not in columns)
    return columns


def _as_model_frame(frame: pd.DataFrame, base_columns: list[str]) -> pd.DataFrame:
    model_frame = frame.reindex(columns=_sorted_model_columns(frame, base_columns)).copy()
    if not model_frame.empty:
        model_frame["src_user_id"] = model_frame["src_user_id"].astype(str)
        model_frame["dst_user_id"] = model_frame["dst_user_id"].astype(str)
    return model_frame


def _annotate_frame(
    frame: pd.DataFrame,
    *,
    assets: dict[str, Any],
    relation: str,
    d1_positions: dict[tuple[str, str], int] | None = None,
) -> pd.DataFrame:
    work = frame.copy()
    if work.empty:
        return work
    work["src_user_id"] = work["src_user_id"].astype(str)
    work["dst_user_id"] = work["dst_user_id"].astype(str)
    base_score = pd.to_numeric(work.get("edge_weight", 0.0), errors="coerce").fillna(0.0).astype(np.float32)
    work["base_score"] = base_score
    score_map = assets["user_score_map"]
    src_scores = work["src_user_id"].map(score_map).fillna(0.0).astype(np.float32)
    dst_scores = work["dst_user_id"].map(score_map).fillna(0.0).astype(np.float32)
    pair_scores = np.sqrt(src_scores * dst_scores) * (1.0 - np.abs(src_scores - dst_scores))
    work["pair_abnormal_score"] = np.clip(pair_scores, 0.0, 1.0).astype(np.float32)
    work["abnormal_score_src"] = src_scores.astype(np.float32)
    work["abnormal_score_dst"] = dst_scores.astype(np.float32)
    work["relation"] = relation
    work["base_rank_score"] = _base_rank_score(work)
    if d1_positions is not None:
        work["is_d1_edge"] = [
            (src, dst) in d1_positions
            for src, dst in zip(work["src_user_id"].astype(str), work["dst_user_id"].astype(str))
        ]
        work["d1_order"] = [
            d1_positions.get((src, dst), 10**9)
            for src, dst in zip(work["src_user_id"].astype(str), work["dst_user_id"].astype(str))
        ]
    return work


def _base_rank_score(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=np.float32)
    rank = frame.groupby("src_user_id")["base_score"].rank(method="first", ascending=False)
    size = frame.groupby("src_user_id")["base_score"].transform("size").astype(float)
    denom = (size - 1.0).clip(lower=1.0)
    score = 1.0 - ((rank.astype(float) - 1.0) / denom)
    score = score.where(size > 1.0, 1.0)
    return score.astype(np.float32)


def _edge_label_counts(frame: pd.DataFrame, user_label_map: dict[str, int]) -> dict[str, int]:
    counts = Counter()
    for row in frame[["src_user_id", "dst_user_id"]].astype(str).itertuples(index=False):
        src_label = int(user_label_map.get(row.src_user_id, 0))
        dst_label = int(user_label_map.get(row.dst_user_id, 0))
        if src_label == 1 and dst_label == 1:
            counts["fake_fake"] += 1
        elif src_label == 0 and dst_label == 0:
            counts["real_real"] += 1
        else:
            counts["fake_real"] += 1
    return counts


def _numeric_mean(frame: pd.DataFrame, column: str, default: float = 0.0) -> float:
    if frame.empty or column not in frame.columns:
        return float(default)
    values = pd.to_numeric(frame[column], errors="coerce").fillna(default)
    return float(values.mean()) if len(values) else float(default)


def _write_edge_audit(
    *,
    exp_dir: Path,
    selected_frames: dict[str, pd.DataFrame],
    work_frames: dict[str, pd.DataFrame],
    assets: dict[str, Any],
    selection_policy: str,
) -> None:
    metrics_dir = exp_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    user_df = assets["user_df"]
    user_label_map = user_df.set_index(user_df["user_id"].astype(str))["user_label"].astype(int).to_dict()
    all_users = set(user_df["user_id"].astype(str))
    quality_rows: list[dict[str, Any]] = []
    topology_rows: list[dict[str, Any]] = []
    samples: list[pd.DataFrame] = []
    for relation in RELATIONS:
        selected = selected_frames[relation]
        work = work_frames[relation]
        d1_frame = assets["d1_edge_frames"][relation]
        d1_pairs = _pair_set(d1_frame)
        selected_pairs = _pair_set(selected)
        counts = _edge_label_counts(selected, user_label_map)
        degree_counter = Counter(selected["src_user_id"].astype(str).tolist()) if not selected.empty else Counter()
        num_edges = int(len(selected))
        same_label = counts["fake_fake"] + counts["real_real"]
        quality_rows.append(
            {
                "relation": relation,
                "num_edges": num_edges,
                "avg_degree": float(np.mean(list(degree_counter.values())) if degree_counter else 0.0),
                "isolated_user_count": int(len(all_users - set(degree_counter.keys()))),
                "same_label_ratio": float(same_label / max(num_edges, 1)),
                "fake_fake_ratio": float(counts["fake_fake"] / max(num_edges, 1)),
                "fake_real_ratio": float(counts["fake_real"] / max(num_edges, 1)),
                "real_real_ratio": float(counts["real_real"] / max(num_edges, 1)),
                "avg_edge_weight": _numeric_mean(selected, "edge_weight"),
                "avg_pair_abnormal_score": _numeric_mean(selected, "pair_abnormal_score"),
                "avg_base_rank_score": _numeric_mean(work, "base_rank_score"),
                "avg_diagnostic_gate_score": _numeric_mean(work, "diagnostic_gate_score"),
            }
        )
        topology_rows.append(
            {
                "relation": relation,
                "selection_policy": selection_policy,
                "d1_edges": int(len(d1_frame)),
                "selected_edges": num_edges,
                "directed_pair_overlap": float(len(d1_pairs & selected_pairs) / max(len(d1_pairs), 1)),
                "new_edges_ratio": float(len(selected_pairs - d1_pairs) / max(len(selected_pairs), 1)),
                "dropped_edges_ratio": float(len(d1_pairs - selected_pairs) / max(len(d1_pairs), 1)),
                "d1_pair_hash": _hash_directed_pairs(d1_frame),
                "selected_pair_hash": _hash_directed_pairs(selected),
                "topology_exact_match": bool(_hash_directed_pairs(d1_frame) == _hash_directed_pairs(selected)),
            }
        )
        audit_cols = [
            "relation",
            "src_user_id",
            "dst_user_id",
            "edge_weight",
            "base_score",
            "pair_abnormal_score",
            "abnormal_score_src",
            "abnormal_score_dst",
            "base_rank_score",
            "u_shape_abnormal_score",
            "diagnostic_gate_score",
            "is_d1_edge",
            "d1_order",
        ]
        samples.append(work.reindex(columns=[col for col in audit_cols if col in work.columns]).head(250))
    pd.DataFrame(quality_rows).to_csv(metrics_dir / "d2_edge_quality_by_relation.csv", index=False)
    pd.DataFrame(topology_rows).to_csv(metrics_dir / "d2_topology_sanity.csv", index=False)
    if samples:
        pd.concat(samples, ignore_index=True).to_csv(metrics_dir / "d2_edge_selection_sample.csv", index=False)


def _build_fixed_d1_frames(*, assets: dict[str, Any], exp_dir: Path) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    edge_dir = exp_dir / "edges"
    edge_dir.mkdir(parents=True, exist_ok=True)
    selected_frames: dict[str, pd.DataFrame] = {}
    work_frames: dict[str, pd.DataFrame] = {}
    for relation in RELATIONS:
        d1_frame = assets["d1_edge_frames"][relation].copy()
        base_columns = list(d1_frame.columns)
        work = _annotate_frame(d1_frame, assets=assets, relation=relation)
        model_frame = _as_model_frame(work, base_columns)
        model_frame.to_csv(edge_dir / f"{relation}_edges.csv", index=False)
        selected_frames[relation] = model_frame
        work_frames[relation] = work
    return selected_frames, work_frames


def _build_candidate_pool(*, assets: dict[str, Any], output_root: Path, cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    pool_top_k = int(cfg.get("pool_top_k", 60))
    source_dir_value = str(cfg.get("candidate_pool_source_dir", "")).strip()
    if source_dir_value:
        source_dir = Path(source_dir_value)
        edge_dir = source_dir / "edges" if (source_dir / "edges").is_dir() else source_dir
        if edge_dir.is_dir() and all((edge_dir / f"{relation}_edges.csv").exists() for relation in RELATIONS):
            print(f"reusing candidate pool source={edge_dir}", flush=True)
            _save_json(
                output_root / "candidate_pool_source.json",
                {
                    "source": str(edge_dir),
                    "pool_top_k": pool_top_k,
                    "policy": "reuse existing no-TNS topK candidate edge files; rebuild only when the source is unavailable",
                },
            )
            return {relation: pd.read_csv(edge_dir / f"{relation}_edges.csv") for relation in RELATIONS}
        print(f"candidate pool source unavailable; rebuilding source={edge_dir}", flush=True)
    pool_dir = output_root / f"_candidate_pool_top{pool_top_k}"
    print(f"building candidate pool top_k={pool_top_k} dir={pool_dir}", flush=True)
    frames = build_edge_frames(
        user_df=assets["user_df"],
        user_text_vectors=assets["user_text_vectors"],
        user_abnormal_vectors=assets["user_abnormal_vectors"],
        output_dir=pool_dir,
        top_k=pool_top_k,
        review_features=assets["review_scores_df"],
        logic_threshold_mode="quantile",
        logic_threshold_quantile=0.60,
        logic_threshold_value=0.30,
        graph_mode="current",
        use_tns_guided_logic=False,
    )
    return {relation: frames[relation].copy() for relation in RELATIONS}


def _score_candidate_pool(
    *,
    assets: dict[str, Any],
    pool_frames: dict[str, pd.DataFrame],
    cfg: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    user_split_map = assets["user_df"].set_index(assets["user_df"]["user_id"].astype(str))["split"].astype(str).to_dict()
    weights = cfg.get("diagnostic_score_weights", {})
    base_w = float(weights.get("base_rank", 0.55))
    u_w = float(weights.get("u_shape", 0.30))
    pair_w = float(weights.get("pair_abnormal", 0.15))
    scored: dict[str, pd.DataFrame] = {}
    param_rows: list[dict[str, Any]] = []
    for relation in RELATIONS:
        d1_frame = assets["d1_edge_frames"][relation]
        d1_positions = {
            (str(row.src_user_id), str(row.dst_user_id)): idx
            for idx, row in enumerate(d1_frame.itertuples(index=False))
        }
        work = _annotate_frame(
            pool_frames[relation],
            assets=assets,
            relation=relation,
            d1_positions=d1_positions,
        )
        if work.empty:
            scored[relation] = work
            continue
        src_split = work["src_user_id"].astype(str).map(user_split_map).fillna("unknown")
        train_scores = pd.to_numeric(work.loc[src_split == "train", "pair_abnormal_score"], errors="coerce").dropna()
        all_scores = pd.to_numeric(work["pair_abnormal_score"], errors="coerce").dropna()
        reference_scores = train_scores if not train_scores.empty else all_scores
        median = float(reference_scores.median()) if not reference_scores.empty else 0.0
        q05 = float(reference_scores.quantile(0.05)) if not reference_scores.empty else 0.0
        q95 = float(reference_scores.quantile(0.95)) if not reference_scores.empty else 1.0
        denom = max(abs(q95 - median), abs(q05 - median), 1e-6)
        u_shape = (pd.to_numeric(work["pair_abnormal_score"], errors="coerce").fillna(0.0) - median).abs() / denom
        work["u_shape_abnormal_score"] = u_shape.clip(0.0, 1.0).astype(np.float32)
        pair_score = pd.to_numeric(work["pair_abnormal_score"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        work["diagnostic_gate_score"] = (
            base_w * pd.to_numeric(work["base_rank_score"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
            + u_w * work["u_shape_abnormal_score"]
            + pair_w * pair_score
        ).astype(np.float32)
        param_rows.append(
            {
                "relation": relation,
                "pool_edges": int(len(work)),
                "train_edges_for_params": int(len(train_scores)),
                "pair_abnormal_median": median,
                "pair_abnormal_q05": q05,
                "pair_abnormal_q95": q95,
                "u_shape_denominator": denom,
                "base_rank_weight": base_w,
                "u_shape_weight": u_w,
                "pair_abnormal_weight": pair_w,
            }
        )
        scored[relation] = work
    return scored, param_rows


def _select_topk(frame: pd.DataFrame, *, rank_col: str, k: int) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    sort_cols = ["src_user_id", rank_col, "base_rank_score", "base_score", "dst_user_id"]
    ascending = [True, False, False, False, True]
    selected = (
        frame.sort_values(sort_cols, ascending=ascending)
        .groupby("src_user_id", as_index=False, group_keys=False)
        .head(k)
        .copy()
    )
    return selected


def _select_topk_with_d1_reserve(
    frame: pd.DataFrame,
    *,
    rank_col: str,
    k: int,
    reserve_ratio: float,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    reserve_k = int(round(float(k) * float(reserve_ratio)))
    reserve_k = max(0, min(k, reserve_k))
    pieces: list[pd.DataFrame] = []
    for _, group in frame.groupby("src_user_id", sort=False):
        reserved = (
            group.loc[group.get("is_d1_edge", False).astype(bool)]
            .sort_values(["d1_order", "base_rank_score"], ascending=[True, False])
            .head(reserve_k)
        )
        reserved_keys = set(zip(reserved["src_user_id"].astype(str), reserved["dst_user_id"].astype(str)))
        fill_mask = [
            (src, dst) not in reserved_keys
            for src, dst in zip(group["src_user_id"].astype(str), group["dst_user_id"].astype(str))
        ]
        fill_pool = group.loc[fill_mask]
        fill = fill_pool.sort_values(
            [rank_col, "base_rank_score", "base_score", "dst_user_id"],
            ascending=[False, False, False, True],
        ).head(max(k - len(reserved), 0))
        pieces.append(pd.concat([reserved, fill], ignore_index=True))
    if not pieces:
        return frame.iloc[0:0].copy()
    return pd.concat(pieces, ignore_index=True)


def _build_pool_reselect_frames(
    *,
    assets: dict[str, Any],
    exp_dir: Path,
    scored_pool: dict[str, pd.DataFrame],
    selection_score: str,
    selected_top_k: int,
    d1_reserve_ratio: float | None,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    edge_dir = exp_dir / "edges"
    edge_dir.mkdir(parents=True, exist_ok=True)
    selected_frames: dict[str, pd.DataFrame] = {}
    work_frames: dict[str, pd.DataFrame] = {}
    for relation in RELATIONS:
        candidate = scored_pool[relation].copy()
        if d1_reserve_ratio is None:
            selected = _select_topk(candidate, rank_col=selection_score, k=selected_top_k)
        else:
            selected = _select_topk_with_d1_reserve(
                candidate,
                rank_col=selection_score,
                k=selected_top_k,
                reserve_ratio=d1_reserve_ratio,
            )
        base_columns = list(assets["d1_edge_frames"][relation].columns)
        model_frame = _as_model_frame(selected, base_columns)
        model_frame.to_csv(edge_dir / f"{relation}_edges.csv", index=False)
        selected_frames[relation] = model_frame
        work_frames[relation] = selected
    return selected_frames, work_frames


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
    edge_frames: dict[str, pd.DataFrame],
    gate_eta: float,
    seed: int,
    edge_only: bool,
) -> dict[str, Any]:
    metrics_dir = exp_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    compute_edge_stats(edge_frames=edge_frames, user_df=assets["user_df"], output_dir=exp_dir)
    if edge_only:
        return {
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
        use_abnormal_gate=True,
        use_abnormal_value_gate=False,
        use_abnormal_attention_bias=False,
        abnormal_score_source="auto",
        abnormal_edge_lambda=1.0,
        abnormal_edge_eta=0.5,
        abnormal_gate_eta=gate_eta,
        abnormal_pair_mode="both_high",
        abnormal_gate_learnable=True,
        abnormal_attention_gamma=1.0,
        review_scores_df=assets["review_scores_df"],
        selected_edge_set="Base_LogicAE_CB",
        relation_topk=None,
        use_node_gat=False,
    )
    return result_df.loc[result_df["edge_set"] == "Base_LogicAE_CB"].iloc[0].to_dict()


def _run_variant(
    *,
    exp_dir: Path,
    variant: dict[str, Any],
    cfg: dict[str, Any],
    assets: dict[str, Any],
    scored_pool: dict[str, pd.DataFrame] | None,
    seed: int,
    resume: bool,
    edge_only: bool,
) -> dict[str, Any]:
    experiment_name = str(variant["experiment_name"])
    if resume:
        row = _load_completed_row(exp_dir)
        if row is not None:
            print(f"[resume] skip completed {experiment_name}: {exp_dir}", flush=True)
            return row

    exp_dir.mkdir(parents=True, exist_ok=True)
    topology_mode = str(variant["topology_mode"])
    gate_eta = float(variant.get("gate_eta", 0.5))
    selected_top_k = int(cfg.get("selected_top_k", 20))
    if topology_mode == "fixed_d1":
        edge_frames, work_frames = _build_fixed_d1_frames(assets=assets, exp_dir=exp_dir)
        selection_policy = "fixed_d1_topology"
    elif topology_mode in {"pool_reselect", "pool_reselect_reserve"}:
        if scored_pool is None:
            raise ValueError("scored_pool is required for pool reselection variants")
        selection_score = str(variant.get("selection_score", "pair_abnormal_score"))
        reserve_ratio = (
            float(variant.get("d1_reserve_ratio", 0.5))
            if topology_mode == "pool_reselect_reserve"
            else None
        )
        edge_frames, work_frames = _build_pool_reselect_frames(
            assets=assets,
            exp_dir=exp_dir,
            scored_pool=scored_pool,
            selection_score=selection_score,
            selected_top_k=selected_top_k,
            d1_reserve_ratio=reserve_ratio,
        )
        selection_policy = (
            f"pool_top{int(cfg.get('pool_top_k', 60))}_select_top{selected_top_k}_by_{selection_score}"
            if reserve_ratio is None
            else f"pool_top{int(cfg.get('pool_top_k', 60))}_reserve_{reserve_ratio}_select_top{selected_top_k}_by_{selection_score}"
        )
    else:
        raise ValueError(f"Unsupported topology_mode={topology_mode!r}")

    _write_edge_audit(
        exp_dir=exp_dir,
        selected_frames=edge_frames,
        work_frames=work_frames,
        assets=assets,
        selection_policy=selection_policy,
    )
    row = _run_graph(
        exp_dir=exp_dir,
        assets=assets,
        edge_frames=edge_frames,
        gate_eta=gate_eta,
        seed=seed,
        edge_only=edge_only,
    )
    _save_json(
        exp_dir / "run_summary.json",
        {
            "experiment_name": experiment_name,
            "strategy": variant.get("strategy"),
            "implementation": "routeD2_abnormal_edge_gate",
            "best_graph_model": row,
            "topology_mode": topology_mode,
            "selection_policy": selection_policy,
            "gate_policy": "learnable per-edge message gate, no TNS feature, no manual TNS weight",
            "gate_eta": gate_eta,
            "notes": variant.get("notes", ""),
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
            "implementation": "routeD2_abnormal_edge_gate",
            "topology_mode": topology_mode,
            "selection_policy": selection_policy,
            "gate_eta": gate_eta,
            "no_tns": True,
        },
    )
    return row


def _summary_row(
    *,
    experiment_name: str,
    strategy: str,
    topology_mode: str,
    gate_eta: float | None,
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
        "topology_mode": topology_mode,
        "gate_eta": gate_eta,
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

    print(f"routeD2 abnormal-edge-gate output_root={output_root} resume={args.resume}", flush=True)
    print("semantic_contract=no TNS; D2A fixed D1 topology; D2B pool reselection by abnormal edge diagnostics", flush=True)
    assets = _load_assets()

    variants = cfg["variants"]
    needs_pool = any(str(item.get("topology_mode", "")).startswith("pool_reselect") for item in variants)
    scored_pool = None
    pool_param_rows: list[dict[str, Any]] = []
    if needs_pool:
        pool_frames = _build_candidate_pool(assets=assets, output_root=output_root, cfg=cfg)
        scored_pool, pool_param_rows = _score_candidate_pool(assets=assets, pool_frames=pool_frames, cfg=cfg)
        pd.DataFrame(pool_param_rows).to_csv(output_root / "d2_candidate_pool_score_params.csv", index=False)

    rows: list[dict[str, Any]] = [
        _summary_row(
            experiment_name="D1_EGAT_Base_LogicAE_CB",
            strategy="reference_only",
            topology_mode="reference_only",
            gate_eta=None,
            row=assets["d1_best"],
            d1_best=assets["d1_best"],
            notes="reference row from D1 output; not rerun",
        )
    ]
    run_variants = variants
    for variant in run_variants:
        experiment_name = str(variant["experiment_name"])
        exp_dir = output_root / experiment_name
        row = _run_variant(
            exp_dir=exp_dir,
            variant=variant,
            cfg=cfg,
            assets=assets,
            scored_pool=scored_pool,
            seed=seed,
            resume=args.resume,
            edge_only=args.smoke_edges_only,
        )
        rows.append(
            _summary_row(
                experiment_name=experiment_name,
                strategy=str(variant.get("strategy", "")),
                topology_mode=str(variant.get("topology_mode", "")),
                gate_eta=float(variant.get("gate_eta", np.nan)),
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
    summary_path = output_root / "routeD2_abnormal_edge_gate_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    _save_json(
        output_root / "run_summary.json",
        {
            "experiment_name": "routeD2_abnormal_edge_gate",
            "implementation": "routeD2_abnormal_edge_gate",
            "semantic_contract": "No TNS. D2A fixes D1 topology and learns per-edge message gates. D2B rebuilds top20 edges from pool60 by abnormal edge diagnostics.",
            "config_path": str(args.config_path),
            "summary_csv": str(summary_path),
            "candidate_pool_score_params": pool_param_rows,
            "rows": rows,
        },
    )
    print(f"summary_path={summary_path}", flush=True)


if __name__ == "__main__":
    main()
