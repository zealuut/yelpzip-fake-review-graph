from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ISS_COLUMNS = ["RD", "AD", "EXR", "MRO", "ATR"]
GROUP_COLUMNS = [
    "group_id",
    "burst_start",
    "burst_end",
    "burst_days",
    "group_size_raw",
    "group_size_purified",
    "num_businesses",
    "num_reviews",
    "GRT",
    "GS",
    "GRD",
    "GOR",
    "GER",
    "GCAR",
    "GSS",
    "group_rank",
]
MEMBERSHIP_COLUMNS = [
    "user_id",
    "group_id",
    "membership_type",
    "is_seed_user",
    "num_reviews_in_group",
    "num_businesses_in_group",
    "member_temporal_span",
    "group_GSS",
    "group_rank",
]
NODE_FEATURE_COLUMNS = [
    "user_id",
    "tnsgd_in_any_group",
    "tnsgd_raw_group_count",
    "tnsgd_core_group_count",
    "tnsgd_max_group_GSS",
    "tnsgd_mean_group_GSS",
    "tnsgd_min_group_rank",
    "tnsgd_max_group_size",
    "tnsgd_total_burst_days",
]


@dataclass(frozen=True)
class TNSGDConfig:
    experiment_name: str = "TNSGD-GroupFirst-phi5"
    phi_days: int = 5
    delta_I: float = 0.5
    merge_jaccard: float = 0.8
    top_sequence_pool: int = 300
    strategy_top_n: int = 30
    strategy_last_n: int = 0
    min_raw_group_size: int = 3
    min_core_group_size: int = 2
    group_size_norm: float = 10.0
    asset_project_root: Path = Path("/home/xyz/HuChao (2)/Bert-TextClassification")
    base_protocol_dir: Path = Path(
        "/home/xyz/HuChao (2)/Bert-TextClassification/graph/outputs/yelpzip_balanced_current_graph_no_reweight_20260502_160620"
    )


def _normalize_user_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_auc_ap(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    finite = np.isfinite(scores)
    labels = labels[finite]
    scores = scores[finite]
    if len(labels) == 0 or len(np.unique(labels)) < 2 or len(np.unique(scores)) < 2:
        return 0.5, float(labels.mean()) if len(labels) else 0.0
    return float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores))


def _load_assets(cfg: TNSGDConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    user_df = pd.read_csv(cfg.base_protocol_dir / "user_scores_enriched.csv")
    review_df = pd.read_csv(cfg.base_protocol_dir / "prepared_data" / "reviews_canonical.csv")
    user_df["user_id"] = user_df["user_id"].map(_normalize_user_id)
    review_df["user_id"] = review_df["user_id"].map(_normalize_user_id)
    review_df["product_id"] = review_df["product_id"].astype(str)
    review_df["review_node_id"] = review_df["review_node_id"].astype(str)
    review_df["review_datetime"] = pd.to_datetime(review_df["review_datetime"], errors="coerce")
    review_df = review_df.dropna(subset=["review_datetime"]).reset_index(drop=True)
    return user_df, review_df


def _attach_iss(user_df: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in ISS_COLUMNS if column not in user_df.columns]
    if missing:
        raise ValueError(f"Missing ISS columns: {missing}")
    frame = user_df.copy()
    indicators = frame[ISS_COLUMNS].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    frame["ISS"] = indicators.clip(lower=0.0, upper=1.0).mean(axis=1)
    return frame


def _event_row(seed: pd.Series, other: pd.Series, product_id: str) -> dict[str, Any]:
    seed_time = pd.Timestamp(seed["review_datetime"])
    other_time = pd.Timestamp(other["review_datetime"])
    return {
        "seed_user": str(seed["user_id"]),
        "neighbor_user": str(other["user_id"]),
        "business_id": str(product_id),
        "event_time": other_time,
        "seed_review_id": str(seed["review_node_id"]),
        "neighbor_review_id": str(other["review_node_id"]),
        "seed_rating": float(seed["rating"]),
        "neighbor_rating": float(other["rating"]),
        "time_gap_days": float(abs((other_time - seed_time).total_seconds()) / 86400.0),
    }


def _build_temporal_neighbor_events(review_df: pd.DataFrame, high_suspicious_users: set[str], phi_days: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    window = pd.Timedelta(days=int(phi_days))
    cols = ["review_node_id", "user_id", "product_id", "rating", "review_datetime"]
    for product_id, pdf in review_df[cols].groupby("product_id", sort=False):
        pdf = pdf.sort_values("review_datetime").reset_index(drop=True)
        times = pdf["review_datetime"].tolist()
        for i, row in pdf.iterrows():
            seed_user = str(row["user_id"])
            if seed_user not in high_suspicious_users:
                continue
            seed_time = row["review_datetime"]
            j = i - 1
            while j >= 0 and (seed_time - times[j]) <= window:
                other = pdf.iloc[j]
                if str(other["user_id"]) != seed_user:
                    rows.append(_event_row(row, other, product_id))
                j -= 1
            j = i + 1
            while j < len(pdf) and (times[j] - seed_time) <= window:
                other = pdf.iloc[j]
                if str(other["user_id"]) != seed_user:
                    rows.append(_event_row(row, other, product_id))
                j += 1
    if not rows:
        return pd.DataFrame(
            columns=[
                "seed_user",
                "neighbor_user",
                "business_id",
                "event_time",
                "seed_review_id",
                "neighbor_review_id",
                "seed_rating",
                "neighbor_rating",
                "time_gap_days",
            ]
        )
    return (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["seed_user", "neighbor_user", "business_id", "seed_review_id", "neighbor_review_id"])
        .sort_values(["seed_user", "event_time", "business_id"])
        .reset_index(drop=True)
    )


def _select_seed_users(events_df: pd.DataFrame, user_df: pd.DataFrame, cfg: TNSGDConfig) -> pd.DataFrame:
    high = user_df[user_df["ISS"] >= cfg.delta_I][["user_id", "ISS"]].copy()
    lengths = events_df.groupby("seed_user", sort=False).size().rename("tns_sequence_length").reset_index()
    pool = high.merge(lengths, left_on="user_id", right_on="seed_user", how="inner").drop(columns=["seed_user"])
    if pool.empty:
        return pd.DataFrame(columns=["user_id", "ISS", "tns_sequence_length", "selection_bucket"])
    pool = pool.sort_values(["tns_sequence_length", "ISS", "user_id"], ascending=[False, False, True]).head(cfg.top_sequence_pool)
    selected_parts: list[pd.DataFrame] = []
    if cfg.strategy_top_n > 0:
        top = pool.sort_values(["ISS", "tns_sequence_length", "user_id"], ascending=[False, False, True]).head(cfg.strategy_top_n).copy()
        top["selection_bucket"] = "top"
        selected_parts.append(top)
    if cfg.strategy_last_n > 0:
        last = pool.sort_values(["ISS", "tns_sequence_length", "user_id"], ascending=[True, False, True]).head(cfg.strategy_last_n).copy()
        last["selection_bucket"] = "last"
        selected_parts.append(last)
    if not selected_parts:
        return pd.DataFrame(columns=["user_id", "ISS", "tns_sequence_length", "selection_bucket"])
    return pd.concat(selected_parts, ignore_index=True).drop_duplicates(subset=["user_id"]).reset_index(drop=True)


def _candidate_groups(events_df: pd.DataFrame, seed_users: set[str], cfg: TNSGDConfig) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    selected_events = events_df[events_df["seed_user"].isin(seed_users)]
    for seed_user, seq in selected_events.groupby("seed_user", sort=False):
        seq = seq.sort_values("event_time").reset_index(drop=True)
        if seq.empty:
            continue
        split_points = [0]
        times = seq["event_time"].tolist()
        for idx in range(1, len(seq)):
            gap_days = float((times[idx] - times[idx - 1]).total_seconds() / 86400.0)
            if gap_days > cfg.phi_days:
                split_points.append(idx)
        split_points.append(len(seq))
        for start_idx, end_idx in zip(split_points[:-1], split_points[1:]):
            session = seq.iloc[start_idx:end_idx]
            raw_members = set(session["neighbor_user"].astype(str).tolist()) | {str(seed_user)}
            if len(raw_members) < cfg.min_raw_group_size:
                continue
            candidates.append(
                {
                    "burst_start": pd.Timestamp(session["event_time"].min()).normalize(),
                    "burst_end": pd.Timestamp(session["event_time"].max()).normalize(),
                    "raw_members": raw_members,
                    "seed_users": {str(seed_user)},
                    "business_ids": set(session["business_id"].astype(str).tolist()),
                    "review_ids": set(session["seed_review_id"].astype(str).tolist()) | set(session["neighbor_review_id"].astype(str).tolist()),
                    "source_session_count": 1,
                }
            )
    return candidates


def _merge_candidates(candidates: list[dict[str, Any]], merge_jaccard: float) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for cand in sorted(candidates, key=lambda c: (c["burst_start"], c["burst_end"], sorted(c["raw_members"])[0])):
        target = None
        for existing in merged:
            if cand["burst_start"] != existing["burst_start"] or cand["burst_end"] != existing["burst_end"]:
                continue
            union = cand["raw_members"] | existing["raw_members"]
            inter = cand["raw_members"] & existing["raw_members"]
            if len(inter) / max(len(union), 1) >= merge_jaccard:
                target = existing
                break
        if target is None:
            merged.append({k: (set(v) if isinstance(v, set) else v) for k, v in cand.items()})
        else:
            target["raw_members"].update(cand["raw_members"])
            target["seed_users"].update(cand["seed_users"])
            target["business_ids"].update(cand["business_ids"])
            target["review_ids"].update(cand["review_ids"])
            target["source_session_count"] += int(cand.get("source_session_count", 1))
    return merged


def _group_review_tightness(group_reviews: pd.DataFrame, group_size: int, num_businesses: int, burst_days: float) -> float:
    if group_reviews.empty or group_size < 2:
        return 0.0
    possible_pairs = max(group_size * (group_size - 1) / 2.0, 1.0)
    pair_hits = 0
    for _, pdf in group_reviews.groupby("product_id", sort=False):
        users = set(pdf["user_id"].astype(str).tolist())
        if len(users) >= 2:
            pair_hits += len(list(combinations(users, 2)))
    density = pair_hits / max(possible_pairs * max(num_businesses, 1), 1.0)
    return float(np.clip(density / max(float(burst_days), 1.0), 0.0, 1.0))


def _group_rating_deviation(group_reviews: pd.DataFrame, product_avg: dict[str, float]) -> float:
    if group_reviews.empty:
        return 0.0
    diffs = [abs(float(row.rating) - float(product_avg.get(str(row.product_id), row.rating))) / 4.0 for row in group_reviews.itertuples(index=False)]
    return float(np.clip(np.mean(diffs), 0.0, 1.0)) if diffs else 0.0


def _group_one_day_reviews(group_reviews: pd.DataFrame) -> float:
    if group_reviews.empty:
        return 0.0
    return float(np.clip(group_reviews["review_datetime"].dt.date.value_counts().max() / max(len(group_reviews), 1), 0.0, 1.0))


def _group_extreme_rating_ratio(group_reviews: pd.DataFrame, user_df: pd.DataFrame, members: set[str]) -> float:
    if not group_reviews.empty:
        ratings = group_reviews["rating"].astype(float)
        return float(((ratings <= 1.0) | (ratings >= 5.0)).mean())
    exr = user_df[user_df["user_id"].isin(members)]["EXR"].astype(float)
    return float(np.clip(exr.mean(), 0.0, 1.0)) if len(exr) else 0.0


def _group_coactive_review_ratio(group_reviews: pd.DataFrame, user_total_reviews: dict[str, int], members: set[str]) -> float:
    if group_reviews.empty:
        return 0.0
    total_member_reviews = sum(int(user_total_reviews.get(user_id, 0)) for user_id in members)
    return float(np.clip(len(group_reviews) / max(total_member_reviews, 1), 0.0, 1.0))


def _score_groups(groups: list[dict[str, Any]], user_df: pd.DataFrame, review_df: pd.DataFrame, cfg: TNSGDConfig) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    iss_map = user_df.set_index("user_id")["ISS"].to_dict()
    product_avg = review_df.groupby("product_id")["rating"].mean().to_dict()
    user_total_reviews = review_df.groupby("user_id")["review_node_id"].nunique().to_dict()
    rows: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    for group in groups:
        raw_members = set(group["raw_members"])
        core_members = {user_id for user_id in raw_members if float(iss_map.get(user_id, 0.0)) >= cfg.delta_I}
        if len(raw_members) < cfg.min_raw_group_size or len(core_members) < cfg.min_core_group_size:
            continue
        start = pd.Timestamp(group["burst_start"])
        end = pd.Timestamp(group["burst_end"])
        burst_days = float((end - start).days + 1)
        member_basis = core_members if core_members else raw_members
        mask = (
            review_df["user_id"].isin(member_basis)
            & (review_df["review_datetime"] >= start)
            & (review_df["review_datetime"] < end + pd.Timedelta(days=1))
        )
        if group["business_ids"]:
            mask &= review_df["product_id"].isin(group["business_ids"])
        group_reviews = review_df[mask].copy()
        if group_reviews.empty:
            fallback = review_df["user_id"].isin(member_basis) & review_df["review_node_id"].isin(group["review_ids"])
            group_reviews = review_df[fallback].copy()
        num_reviews = int(len(group_reviews))
        num_businesses = int(group_reviews["product_id"].nunique()) if num_reviews else int(len(group["business_ids"]))
        group_size_purified = int(len(core_members))
        grt = _group_review_tightness(group_reviews, group_size_purified, max(num_businesses, 1), burst_days)
        gs = float(min(group_size_purified / max(cfg.group_size_norm, 1.0), 1.0))
        grd = _group_rating_deviation(group_reviews, product_avg)
        gor = _group_one_day_reviews(group_reviews)
        ger = _group_extreme_rating_ratio(group_reviews, user_df, member_basis)
        gcar = _group_coactive_review_ratio(group_reviews, user_total_reviews, member_basis)
        gss = float(np.mean([grt, gs, grd, gor, ger, gcar]))
        row = {
            "burst_start": start.date().isoformat(),
            "burst_end": end.date().isoformat(),
            "burst_days": burst_days,
            "group_size_raw": int(len(raw_members)),
            "group_size_purified": group_size_purified,
            "num_businesses": num_businesses,
            "num_reviews": num_reviews,
            "GRT": grt,
            "GS": gs,
            "GRD": grd,
            "GOR": gor,
            "GER": ger,
            "GCAR": gcar,
            "GSS": gss,
        }
        rows.append(row)
        states.append({**group, "core_members": core_members, "metrics": row})
    if not rows:
        return pd.DataFrame(columns=GROUP_COLUMNS), []
    order = np.argsort([-row["GSS"] for row in rows])
    ranked_rows: list[dict[str, Any]] = []
    ranked_states: list[dict[str, Any]] = []
    for rank, old_idx in enumerate(order, start=1):
        group_id = f"tnsgd_g{rank:06d}"
        row = {"group_id": group_id, **rows[old_idx], "group_rank": rank}
        state = states[old_idx]
        state["group_id"] = group_id
        state["group_rank"] = rank
        state["metrics"] = row
        ranked_rows.append(row)
        ranked_states.append(state)
    return pd.DataFrame(ranked_rows, columns=GROUP_COLUMNS), ranked_states


def _build_membership(groups: list[dict[str, Any]], review_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group in groups:
        raw_members = set(group["raw_members"])
        core_members = set(group["core_members"])
        start = pd.Timestamp(group["burst_start"])
        end = pd.Timestamp(group["burst_end"])
        business_ids = set(group["business_ids"])
        for user_id in sorted(raw_members | core_members):
            if user_id in raw_members and user_id in core_members:
                membership_type = "both"
            elif user_id in core_members:
                membership_type = "core"
            else:
                membership_type = "raw"
            mask = (
                (review_df["user_id"] == user_id)
                & (review_df["review_datetime"] >= start)
                & (review_df["review_datetime"] < end + pd.Timedelta(days=1))
            )
            if business_ids:
                mask &= review_df["product_id"].isin(business_ids)
            user_reviews = review_df[mask]
            span = 0.0 if user_reviews.empty else float((user_reviews["review_datetime"].max() - user_reviews["review_datetime"].min()).total_seconds() / 86400.0)
            rows.append(
                {
                    "user_id": user_id,
                    "group_id": group["group_id"],
                    "membership_type": membership_type,
                    "is_seed_user": int(user_id in group["seed_users"]),
                    "num_reviews_in_group": int(len(user_reviews)),
                    "num_businesses_in_group": int(user_reviews["product_id"].nunique()) if len(user_reviews) else 0,
                    "member_temporal_span": span,
                    "group_GSS": float(group["metrics"]["GSS"]),
                    "group_rank": int(group["group_rank"]),
                }
            )
    return pd.DataFrame(rows, columns=MEMBERSHIP_COLUMNS)


def _build_node_features(user_df: pd.DataFrame, groups_df: pd.DataFrame, membership_df: pd.DataFrame) -> pd.DataFrame:
    users = user_df[["user_id"]].drop_duplicates().copy()
    if membership_df.empty:
        for column in NODE_FEATURE_COLUMNS[1:]:
            users[column] = 0.0
        return users[NODE_FEATURE_COLUMNS]
    frame = membership_df.copy()
    frame["is_raw_member"] = frame["membership_type"].isin(["raw", "both"]).astype(int)
    frame["is_core_member"] = frame["membership_type"].isin(["core", "both"]).astype(int)
    frame["group_size_purified"] = frame["group_id"].map(groups_df.set_index("group_id")["group_size_purified"]).fillna(0.0)
    frame["burst_days"] = frame["group_id"].map(groups_df.set_index("group_id")["burst_days"]).fillna(0.0)
    agg = (
        frame.groupby("user_id", sort=False)
        .agg(
            tnsgd_raw_group_count=("is_raw_member", "sum"),
            tnsgd_core_group_count=("is_core_member", "sum"),
            tnsgd_max_group_GSS=("group_GSS", "max"),
            tnsgd_mean_group_GSS=("group_GSS", "mean"),
            tnsgd_min_group_rank=("group_rank", "min"),
            tnsgd_max_group_size=("group_size_purified", "max"),
            tnsgd_total_burst_days=("burst_days", "sum"),
        )
        .reset_index()
    )
    agg["tnsgd_in_any_group"] = 1
    result = users.merge(agg, on="user_id", how="left").fillna(0.0)
    for column in ["tnsgd_in_any_group", "tnsgd_raw_group_count", "tnsgd_core_group_count", "tnsgd_min_group_rank", "tnsgd_max_group_size"]:
        result[column] = result[column].astype(int)
    return result[NODE_FEATURE_COLUMNS]


def _label_diagnostics(groups: list[dict[str, Any]], user_df: pd.DataFrame) -> pd.DataFrame:
    label_map = user_df.set_index("user_id")["user_label"].to_dict()
    global_fake_rate = float(user_df["user_label"].mean())
    rows: list[dict[str, Any]] = []
    for group in groups:
        raw_labels = [int(label_map.get(user_id, 0)) for user_id in group["raw_members"]]
        core_labels = [int(label_map.get(user_id, 0)) for user_id in group["core_members"]]
        seed_labels = [int(label_map.get(user_id, 0)) for user_id in group["seed_users"]]
        raw_fake_rate = float(np.mean(raw_labels)) if raw_labels else 0.0
        core_fake_rate = float(np.mean(core_labels)) if core_labels else 0.0
        rows.append(
            {
                "group_id": group["group_id"],
                "group_rank": int(group["group_rank"]),
                "GSS": float(group["metrics"]["GSS"]),
                "raw_num_users": int(len(raw_labels)),
                "raw_num_fake": int(sum(raw_labels)),
                "raw_fake_rate": raw_fake_rate,
                "core_num_users": int(len(core_labels)),
                "core_num_fake": int(sum(core_labels)),
                "core_fake_rate": core_fake_rate,
                "seed_num_users": int(len(seed_labels)),
                "seed_fake_rate": float(np.mean(seed_labels)) if seed_labels else 0.0,
                "global_fake_rate": global_fake_rate,
                "raw_lift_vs_global": raw_fake_rate - global_fake_rate,
                "core_lift_vs_global": core_fake_rate - global_fake_rate,
            }
        )
    return pd.DataFrame(rows)


def run_tnsgd_group_first(cfg: TNSGDConfig, output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    user_df, review_df = _load_assets(cfg)
    user_df = _attach_iss(user_df)
    high_users = set(user_df.loc[user_df["ISS"] >= cfg.delta_I, "user_id"].astype(str).tolist())
    events_df = _build_temporal_neighbor_events(review_df, high_users, cfg.phi_days)
    selected_seeds = _select_seed_users(events_df, user_df, cfg)
    candidates = _candidate_groups(events_df, set(selected_seeds["user_id"].astype(str).tolist()), cfg)
    merged = _merge_candidates(candidates, cfg.merge_jaccard)
    groups_df, group_state = _score_groups(merged, user_df, review_df, cfg)
    membership_df = _build_membership(group_state, review_df)
    node_features_df = _build_node_features(user_df, groups_df, membership_df)
    label_diag_df = _label_diagnostics(group_state, user_df)

    groups_path = output_root / "tnsgd_groups.csv"
    membership_path = output_root / "tnsgd_user_group_membership.csv"
    node_features_path = output_root / "tnsgd_user_node_features.parquet"
    label_diag_path = output_root / "tnsgd_group_label_diagnostics.csv"
    events_path = output_root / "tnsgd_temporal_neighbor_events.csv"
    seeds_path = output_root / "tnsgd_selected_seed_users.csv"
    config_path = output_root / "run_config.json"
    summary_path = output_root / "tnsgd_summary.json"

    groups_df.to_csv(groups_path, index=False)
    membership_df.to_csv(membership_path, index=False)
    node_features_df.to_parquet(node_features_path, index=False)
    label_diag_df.to_csv(label_diag_path, index=False)
    events_df.to_csv(events_path, index=False)
    selected_seeds.to_csv(seeds_path, index=False)
    config_payload = {**asdict(cfg), "asset_project_root": str(cfg.asset_project_root), "base_protocol_dir": str(cfg.base_protocol_dir)}
    config_path.write_text(json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    feature_auc, feature_ap = _safe_auc_ap(user_df["user_label"].to_numpy(), node_features_df["tnsgd_max_group_GSS"].to_numpy(dtype=float))
    any_members = node_features_df[node_features_df["tnsgd_in_any_group"] > 0][["user_id"]].merge(user_df[["user_id", "user_label"]], on="user_id", how="left")
    core_members = node_features_df[node_features_df["tnsgd_core_group_count"] > 0][["user_id"]].merge(user_df[["user_id", "user_label"]], on="user_id", how="left")
    summary = {
        "experiment_name": cfg.experiment_name,
        "output_root": str(output_root.resolve()),
        "phi_days": int(cfg.phi_days),
        "delta_I": float(cfg.delta_I),
        "merge_jaccard": float(cfg.merge_jaccard),
        "top_sequence_pool": int(cfg.top_sequence_pool),
        "strategy": f"Top-{cfg.strategy_top_n}&Last-{cfg.strategy_last_n}",
        "num_users": int(len(user_df)),
        "num_reviews": int(len(review_df)),
        "num_high_iss_users": int(len(high_users)),
        "num_temporal_neighbor_events": int(len(events_df)),
        "num_selected_seed_users": int(len(selected_seeds)),
        "num_candidate_groups_raw": int(len(candidates)),
        "num_candidate_groups_merged": int(len(merged)),
        "num_output_groups": int(len(groups_df)),
        "num_memberships": int(len(membership_df)),
        "num_users_in_any_group": int((node_features_df["tnsgd_in_any_group"] > 0).sum()),
        "num_users_in_core_group": int((node_features_df["tnsgd_core_group_count"] > 0).sum()),
        "global_fake_rate": float(user_df["user_label"].mean()),
        "any_group_fake_rate": float(any_members["user_label"].mean()) if len(any_members) else 0.0,
        "core_group_fake_rate": float(core_members["user_label"].mean()) if len(core_members) else 0.0,
        "max_group_GSS_user_auc": feature_auc,
        "max_group_GSS_user_ap": feature_ap,
        "groups_path": str(groups_path.resolve()),
        "membership_path": str(membership_path.resolve()),
        "node_features_path": str(node_features_path.resolve()),
        "label_diagnostics_path": str(label_diag_path.resolve()),
        "events_path": str(events_path.resolve()),
        "seeds_path": str(seeds_path.resolve()),
        "input_user_scores_sha256": _sha256_file(cfg.base_protocol_dir / "user_scores_enriched.csv"),
        "input_reviews_sha256": _sha256_file(cfg.base_protocol_dir / "prepared_data" / "reviews_canonical.csv"),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
