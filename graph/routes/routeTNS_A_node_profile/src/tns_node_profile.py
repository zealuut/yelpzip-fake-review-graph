from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _normalize_user_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _pair_key(u: str, v: str, product_id: str, review_u: str, review_v: str) -> tuple[str, str, str, str, str]:
    a, b = sorted([u, v])
    r1, r2 = sorted([review_u, review_v])
    return a, b, product_id, r1, r2


@dataclass
class TNSConfig:
    delta_days: int = 3
    session_threshold_days: int = 3
    min_group_size: int = 3
    max_group_duration_days: int = 3


def build_tns_events(review_df: pd.DataFrame, cfg: TNSConfig) -> pd.DataFrame:
    frame = review_df.copy()
    frame["user_id"] = frame["user_id"].map(_normalize_user_id)
    frame["product_id"] = frame["product_id"].astype(str)
    frame["review_datetime"] = pd.to_datetime(frame["review_datetime"], errors="coerce")
    frame = frame.dropna(subset=["review_datetime"]).reset_index(drop=True)
    frame["review_node_id"] = frame["review_node_id"].astype(str)

    rows: list[dict[str, Any]] = []
    delta = pd.Timedelta(days=int(cfg.delta_days))
    for product_id, pdf in frame.groupby("product_id", sort=False):
        pdf = pdf.sort_values("review_datetime").reset_index(drop=True)
        times = pdf["review_datetime"].tolist()
        for i in range(len(pdf)):
            j = i + 1
            while j < len(pdf) and (times[j] - times[i]) <= delta:
                ui = pdf.at[i, "user_id"]
                uj = pdf.at[j, "user_id"]
                if ui != uj:
                    rows.append(
                        {
                            "u": ui,
                            "v": uj,
                            "business_id": str(product_id),
                            "time_u": pdf.at[i, "review_datetime"],
                            "time_v": pdf.at[j, "review_datetime"],
                            "time_gap_days": float(abs((times[j] - times[i]).total_seconds()) / 86400.0),
                            "rating_gap": float(abs(float(pdf.at[i, "rating"]) - float(pdf.at[j, "rating"]))),
                            "review_id_u": str(pdf.at[i, "review_node_id"]),
                            "review_id_v": str(pdf.at[j, "review_node_id"]),
                        }
                    )
                j += 1
    if not rows:
        return pd.DataFrame(
            columns=[
                "u",
                "v",
                "business_id",
                "time_u",
                "time_v",
                "time_gap_days",
                "rating_gap",
                "review_id_u",
                "review_id_v",
            ]
        )
    events = pd.DataFrame(rows).drop_duplicates(
        subset=["u", "v", "business_id", "review_id_u", "review_id_v"]
    )
    return events.reset_index(drop=True)


def build_tns_groups(review_df: pd.DataFrame, events_df: pd.DataFrame, cfg: TNSConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    if events_df.empty:
        empty_groups = pd.DataFrame(
            columns=[
                "group_id",
                "source_user",
                "business_id",
                "group_size",
                "duration_days",
                "time_tightness",
                "co_review_density",
                "rating_consistency",
                "extreme_rating_ratio",
                "business_concentration",
                "member_activity_overlap",
                "tns_group_score",
                "member_user_ids",
            ]
        )
        return empty_groups, pd.DataFrame(columns=["group_id", "user_id", "role"])

    review_meta = review_df.copy()
    review_meta["user_id"] = review_meta["user_id"].map(_normalize_user_id)
    review_meta["product_id"] = review_meta["product_id"].astype(str)
    review_meta["review_datetime"] = pd.to_datetime(review_meta["review_datetime"], errors="coerce")
    review_meta["review_node_id"] = review_meta["review_node_id"].astype(str)

    business_review_counts = review_meta.groupby("product_id")["review_node_id"].nunique().to_dict()
    user_total_reviews = review_meta.groupby("user_id")["review_node_id"].nunique().to_dict()

    group_rows: list[dict[str, Any]] = []
    member_rows: list[dict[str, Any]] = []
    session_gap = pd.Timedelta(days=int(cfg.session_threshold_days))
    max_duration = float(cfg.max_group_duration_days)
    group_id = 0

    for source_user, src_events in events_df.groupby("u", sort=False):
        src_events = src_events.sort_values("time_u").reset_index(drop=True)
        current: list[pd.Series] = []
        prev_time = None
        for row in src_events.itertuples(index=False):
            row_time = pd.Timestamp(row.time_u)
            if prev_time is not None and (row_time - prev_time) > session_gap and current:
                group_id = _flush_group(
                    current=current,
                    group_id=group_id,
                    cfg=cfg,
                    max_duration=max_duration,
                    business_review_counts=business_review_counts,
                    user_total_reviews=user_total_reviews,
                    group_rows=group_rows,
                    member_rows=member_rows,
                )
                current = []
            current.append(pd.Series(row._asdict()))
            prev_time = row_time
        if current:
            group_id = _flush_group(
                current=current,
                group_id=group_id,
                cfg=cfg,
                max_duration=max_duration,
                business_review_counts=business_review_counts,
                user_total_reviews=user_total_reviews,
                group_rows=group_rows,
                member_rows=member_rows,
            )

    return pd.DataFrame(group_rows), pd.DataFrame(member_rows)


def _flush_group(
    current: list[pd.Series],
    group_id: int,
    cfg: TNSConfig,
    max_duration: float,
    business_review_counts: dict[str, int],
    user_total_reviews: dict[str, int],
    group_rows: list[dict[str, Any]],
    member_rows: list[dict[str, Any]],
) -> int:
    if not current:
        return group_id
    session_df = pd.DataFrame(current)
    members = sorted(set(session_df["u"].astype(str).tolist()) | set(session_df["v"].astype(str).tolist()))
    duration_days = float((session_df["time_v"].max() - session_df["time_u"].min()).total_seconds() / 86400.0)
    duration_days = max(duration_days, 0.0)
    if len(members) < int(cfg.min_group_size) or duration_days > max_duration:
        return group_id

    business_id = str(session_df["business_id"].mode().iloc[0])
    pair_count = float(len(session_df))
    possible_pairs = max(len(members) * (len(members) - 1) / 2.0, 1.0)
    co_review_density = pair_count / possible_pairs
    mean_gap = float(session_df["time_gap_days"].mean()) if pair_count > 0 else float(cfg.delta_days)
    time_tightness = 1.0 / (1.0 + mean_gap)
    rating_gap_mean = float(session_df["rating_gap"].mean()) if pair_count > 0 else 0.0
    rating_consistency = 1.0 / (1.0 + rating_gap_mean)
    extreme_rating_ratio = float((session_df["rating_gap"] >= 3.0).mean()) if pair_count > 0 else 0.0
    business_concentration = 1.0 / max(float(business_review_counts.get(business_id, 1)), 1e-6)
    member_activity_overlap = float(
        np.mean([1.0 / max(float(user_total_reviews.get(user_id, 1)), 1.0) for user_id in members])
    )
    group_size = float(len(members))
    group_size_norm = min(group_size / 10.0, 1.0)
    duration_norm = 1.0 - min(duration_days / max(float(cfg.max_group_duration_days), 1.0), 1.0)
    group_score = float(
        np.mean(
            [
                group_size_norm,
                duration_norm,
                time_tightness,
                min(co_review_density, 1.0),
                rating_consistency,
                min(extreme_rating_ratio, 1.0),
                min(business_concentration * 50.0, 1.0),
                min(member_activity_overlap * 20.0, 1.0),
            ]
        )
    )
    gid = f"g_{group_id:06d}"
    group_rows.append(
        {
            "group_id": gid,
            "source_user": str(session_df["u"].iloc[0]),
            "business_id": business_id,
            "group_size": int(group_size),
            "duration_days": duration_days,
            "time_tightness": time_tightness,
            "co_review_density": co_review_density,
            "rating_consistency": rating_consistency,
            "extreme_rating_ratio": extreme_rating_ratio,
            "business_concentration": business_concentration,
            "member_activity_overlap": member_activity_overlap,
            "tns_group_score": group_score,
            "member_user_ids": "|".join(members),
        }
    )
    source_user = str(session_df["u"].iloc[0])
    for user_id in members:
        member_rows.append(
            {
                "group_id": gid,
                "user_id": user_id,
                "role": "source" if user_id == source_user else "neighbor",
            }
        )
    return group_id + 1


def build_tns_user_profile(user_df: pd.DataFrame, groups_df: pd.DataFrame, members_df: pd.DataFrame) -> pd.DataFrame:
    users = user_df[["user_id"]].drop_duplicates().copy()
    users["user_id"] = users["user_id"].map(_normalize_user_id)
    if groups_df.empty or members_df.empty:
        for column in [
            "tns_group_count",
            "tns_high_score_group_count",
            "tns_max_group_score",
            "tns_mean_group_score",
            "tns_top3_group_score_mean",
            "tns_max_group_size",
            "tns_mean_group_size",
            "tns_min_duration",
            "tns_mean_time_tightness",
            "tns_max_co_review_density",
            "tns_unique_tns_partners",
            "tns_unique_businesses",
            "tns_source_role_count",
        ]:
            users[column] = 0.0
        return users

    merged = members_df.merge(groups_df, on="group_id", how="left")
    merged["high_score_flag"] = (merged["tns_group_score"] >= 0.6).astype(int)

    partner_map: dict[str, set[str]] = {}
    for row in groups_df.itertuples(index=False):
        members = [m for m in str(row.member_user_ids).split("|") if m]
        for user_id in members:
            partner_map.setdefault(user_id, set()).update({m for m in members if m != user_id})

    agg = (
        merged.groupby("user_id", sort=False)
        .agg(
            tns_group_count=("group_id", "nunique"),
            tns_high_score_group_count=("high_score_flag", "sum"),
            tns_max_group_score=("tns_group_score", "max"),
            tns_mean_group_score=("tns_group_score", "mean"),
            tns_max_group_size=("group_size", "max"),
            tns_mean_group_size=("group_size", "mean"),
            tns_min_duration=("duration_days", "min"),
            tns_mean_time_tightness=("time_tightness", "mean"),
            tns_max_co_review_density=("co_review_density", "max"),
            tns_unique_businesses=("business_id", "nunique"),
            tns_source_role_count=("role", lambda s: int((s == "source").sum())),
        )
        .reset_index()
    )

    top3 = (
        merged.sort_values(["user_id", "tns_group_score"], ascending=[True, False])
        .groupby("user_id", sort=False)
        .head(3)
        .groupby("user_id", sort=False)["tns_group_score"]
        .mean()
        .rename("tns_top3_group_score_mean")
        .reset_index()
    )
    agg = agg.merge(top3, on="user_id", how="left")
    agg["tns_unique_tns_partners"] = agg["user_id"].map(lambda u: float(len(partner_map.get(u, set()))))

    result = users.merge(agg, on="user_id", how="left").fillna(0.0)
    return result


def build_tns_group_stats(groups_df: pd.DataFrame, members_df: pd.DataFrame, user_df: pd.DataFrame) -> pd.DataFrame:
    if groups_df.empty:
        return pd.DataFrame(
            [
                {
                    "num_groups": 0,
                    "num_users_with_tns": 0,
                    "avg_group_size": 0.0,
                    "avg_duration_days": 0.0,
                    "avg_group_score": 0.0,
                    "fake_member_ratio": 0.0,
                    "real_member_ratio": 0.0,
                }
            ]
        )
    labels = user_df[["user_id", "user_label"]].drop_duplicates().copy()
    labels["user_id"] = labels["user_id"].map(_normalize_user_id)
    merged = members_df.merge(labels, on="user_id", how="left")
    total_members = max(len(merged), 1)
    return pd.DataFrame(
        [
            {
                "num_groups": int(groups_df["group_id"].nunique()),
                "num_users_with_tns": int(members_df["user_id"].nunique()),
                "avg_group_size": float(groups_df["group_size"].mean()),
                "avg_duration_days": float(groups_df["duration_days"].mean()),
                "avg_group_score": float(groups_df["tns_group_score"].mean()),
                "fake_member_ratio": float((merged["user_label"] == 1).sum() / total_members),
                "real_member_ratio": float((merged["user_label"] == 0).sum() / total_members),
            }
        ]
    )


def append_tns_profile(self_features: np.ndarray, user_df: pd.DataFrame, profile_df: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    merged = user_df[["user_id"]].drop_duplicates().copy()
    merged["user_id"] = merged["user_id"].map(_normalize_user_id)
    merged = merged.merge(profile_df, on="user_id", how="left").fillna(0.0)
    tns_columns = [column for column in merged.columns if column != "user_id"]
    tns_matrix = merged[tns_columns].to_numpy(dtype=np.float32)
    if tns_matrix.size == 0:
        tns_matrix = np.zeros((len(merged), 0), dtype=np.float32)
    stats = {
        "feature_dim_before": int(self_features.shape[1]),
        "feature_dim_after": int(self_features.shape[1] + tns_matrix.shape[1]),
        "tns_feature_dim": int(tns_matrix.shape[1]),
        "tns_feature_hash": _sha256_array(tns_matrix),
    }
    return np.concatenate([self_features.astype(np.float32), tns_matrix], axis=1), stats


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
