from __future__ import annotations

import ast
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - local fallback
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []


EDGE_COLUMNS = ["src_user_id", "dst_user_id", "edge_type", "edge_weight"]
DEFAULT_GRAPH_SUPPORT_BETAS = {
    "UPU": 0.20,
    "UTU": 0.15,
    "USU": 0.15,
    "CB": 0.20,
    "LogicAE_CB": 0.30,
}
DEFAULT_LOGIC_THRESHOLD_MODE = "quantile"
DEFAULT_LOGIC_THRESHOLD_QUANTILE = 0.60
DEFAULT_LOGIC_THRESHOLD_VALUE = 0.30


def _empty_edge_frame(extra_columns: list[str] | None = None) -> pd.DataFrame:
    columns = EDGE_COLUMNS + list(extra_columns or [])
    return pd.DataFrame(columns=columns)


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-8, a_max=None)
    return vectors / norms


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _as_string_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, set):
        return {str(item) for item in value}
    if isinstance(value, (list, tuple)):
        return {str(item) for item in value}
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return set()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, (list, tuple, set)):
                    return {str(item) for item in parsed}
            except Exception:
                pass
        return {stripped}
    return {str(value)}


def _resolve_time_bucket(timestamp: pd.Timestamp, bucket_type: str) -> str:
    if pd.isna(timestamp):
        return "unknown"
    if bucket_type == "week":
        iso_year, iso_week, _ = timestamp.isocalendar()
        return f"{int(iso_year):04d}-W{int(iso_week):02d}"
    return f"{timestamp.year:04d}-{timestamp.month:02d}"


def _rating_tendency(avg_rating: float) -> str:
    if avg_rating >= 4.0:
        return "high"
    if avg_rating <= 2.0:
        return "low"
    return "mid"


def _rating_entropy(ratings: pd.Series) -> float:
    counts = ratings.round().astype(int).value_counts(normalize=True)
    if counts.empty:
        return 0.0
    return float(-(counts * np.log(counts.clip(lower=1e-12))).sum())


def _day_gap_stats(review_dates: pd.Series) -> tuple[float, float, float]:
    valid_dates = pd.to_datetime(review_dates, errors="coerce").dropna().sort_values()
    if valid_dates.empty:
        return 0.0, 0.0, 0.0
    tenure_days = float((valid_dates.iloc[-1] - valid_dates.iloc[0]).days)
    if len(valid_dates) <= 1:
        return 0.0, 0.0, max(tenure_days, 0.0)
    gaps = valid_dates.diff().dropna().dt.total_seconds().to_numpy(dtype=np.float64) / 86400.0
    return float(np.mean(gaps)), float(np.std(gaps)), max(tenure_days, 0.0)


def _review_day_number(value: Any) -> float:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return float("nan")
    return float(timestamp.toordinal())


def _rating_consistency(src_rating: float, dst_rating: float) -> float:
    rating_gap = abs(float(src_rating) - float(dst_rating))
    return _clip01(1.0 - rating_gap / 4.0)


def _time_proximity(src_day: float, dst_day: float) -> float:
    if not np.isfinite(src_day) or not np.isfinite(dst_day):
        return 0.5
    day_gap = abs(float(src_day) - float(dst_day))
    return _clip01(1.0 / (1.0 + day_gap / 30.0))


def _normalized_idf(total_users: int, degree: int) -> float:
    if degree <= 0 or total_users <= 1:
        return 0.0
    raw = np.log1p(total_users / max(float(degree), 1.0))
    max_raw = np.log1p(float(total_users))
    return _clip01(raw / max(max_raw, 1e-8))


def _percentile_scores(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if len(numeric) <= 1:
        return np.zeros(len(numeric), dtype=np.float32)
    return numeric.rank(method="average", pct=True).to_numpy(dtype=np.float32)


def _add_behavior_anomaly_indicators(user_df: pd.DataFrame) -> pd.DataFrame:
    user_df = user_df.copy()
    user_df["RD"] = (pd.to_numeric(user_df["rating_deviation_avg"], errors="coerce").fillna(0.0) / 4.0).clip(0.0, 1.0)
    user_df["EXR"] = pd.to_numeric(user_df["extreme_rating_ratio"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    user_df["MRO"] = _percentile_scores(user_df["max_daily_reviews"])
    user_df["AD"] = 1.0 - _percentile_scores(user_df["user_tenure_days"])
    user_df["ATR"] = 1.0 - _percentile_scores(user_df["avg_review_gap_days"])
    user_df["behavior_anomaly_score"] = user_df[["RD", "EXR", "MRO", "AD", "ATR"]].mean(axis=1).clip(0.0, 1.0)
    return user_df


def _resolve_logic_threshold(
    candidate_edges: pd.DataFrame,
    mode: str,
    quantile: float,
    threshold_value: float,
) -> float | None:
    mode = str(mode or DEFAULT_LOGIC_THRESHOLD_MODE).lower()
    threshold_value = _clip01(threshold_value)
    if mode == "none":
        return None
    if mode == "fixed":
        return threshold_value
    if mode != "quantile":
        raise ValueError(f"Unsupported logic threshold mode: {mode}")
    if candidate_edges.empty or "edge_weight" not in candidate_edges.columns:
        return threshold_value
    scores = candidate_edges["edge_weight"].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if scores.empty:
        return threshold_value
    q = _clip01(quantile)
    return max(threshold_value, float(np.quantile(scores.to_numpy(dtype=np.float32), q)))


def build_review_and_user_artifacts(
    review_df: pd.DataFrame,
    llm_feature_df: pd.DataFrame,
    review_output_df: pd.DataFrame,
    review_vectors: np.ndarray,
    text_vectors: np.ndarray,
    output_dir: str | Path,
    top_m: int,
    time_bucket: str,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    output_dir = Path(output_dir)
    vector_dir = output_dir / "logic_vectors"
    vector_dir.mkdir(parents=True, exist_ok=True)

    review_features = review_output_df.merge(
        llm_feature_df[
            [
                "review_node_id",
                "specificity_score",
                "template_score",
                "exaggeration_score",
                "experience_detail_score",
                "num_abnormal_patterns",
                "matched_mask_token_count",
                "mask_source",
                "llm_feature_available",
            ]
        ],
        on="review_node_id",
        how="left",
    ).merge(
        review_df[["review_node_id", "review_datetime", "split"]],
        on="review_node_id",
        how="left",
    ).sort_values("review_node_id").reset_index(drop=True)
    llm_feature_available = review_features.get(
        "llm_feature_available",
        pd.Series(1.0, index=review_features.index),
    ).fillna(0.0).astype(float)
    llm_evidence_score = (
        0.2 * review_features["template_score"]
        + 0.2 * review_features["exaggeration_score"]
        + 0.1 * (1.0 - review_features["specificity_score"])
    )
    review_features["evidence_score"] = (
        0.5 * review_features["p_fake_review"]
        + llm_feature_available * llm_evidence_score
    ).clip(0.0, 1.0)
    review_features["time_bucket"] = review_features["review_datetime"].apply(
        lambda value: _resolve_time_bucket(value, time_bucket)
    )
    product_avg_rating = review_df.groupby("product_id")["rating"].mean()
    product_first_date = pd.to_datetime(review_df["review_datetime"], errors="coerce").groupby(review_df["product_id"]).min()

    np.save(vector_dir / "review_abnormal_vectors.npy", review_vectors)
    np.save(vector_dir / "review_text_vectors.npy", text_vectors)
    review_features[
        [
            "review_node_id",
            "user_id",
            "product_id",
            "rating",
            "review_date",
            "review_label",
            "p_fake_review",
            "evidence_score",
            "specificity_score",
            "template_score",
            "exaggeration_score",
            "experience_detail_score",
            "num_abnormal_patterns",
            "matched_mask_token_count",
            "mask_source",
            "llm_feature_available",
        ]
    ].to_csv(vector_dir / "review_abnormal_scores.csv", index=False)

    user_rows: list[dict[str, Any]] = []
    user_abnormal_by_id: dict[str, np.ndarray] = {}
    user_text_by_id: dict[str, np.ndarray] = {}

    for user_id, user_reviews in tqdm(review_features.groupby("user_id"), desc="Aggregating user vectors"):
        user_id = str(user_id)
        review_indices = user_reviews.index.to_numpy()
        ranked_reviews = user_reviews.sort_values("evidence_score", ascending=False)
        selected_indices = ranked_reviews.head(top_m).index.to_numpy()
        abnormal_vector = review_vectors[selected_indices].mean(axis=0)
        text_vector = text_vectors[review_indices].mean(axis=0)

        original_rows = review_df.loc[review_indices]
        review_dates = pd.to_datetime(original_rows["review_date"], errors="coerce")
        date_counts = original_rows.assign(day=review_dates.dt.date).groupby("day").size()
        positive_ratio = float((original_rows["rating"] >= 4).mean())
        negative_ratio = float((original_rows["rating"] <= 2).mean())
        extreme_rating_ratio = float(original_rows["rating"].isin([1.0, 5.0]).mean())
        max_daily_reviews = int(date_counts.max() if not date_counts.empty else len(original_rows))
        avg_gap_days, std_gap_days, tenure_days = _day_gap_stats(original_rows["review_date"])

        product_means = original_rows["product_id"].map(product_avg_rating).astype(float)
        rating_deviation = (original_rows["rating"].astype(float) - product_means).abs().fillna(0.0)
        product_start_dates = original_rows["product_id"].map(product_first_date)
        review_time_lags = (review_dates - pd.to_datetime(product_start_dates, errors="coerce")).dt.total_seconds() / 86400.0
        review_time_lags = review_time_lags.replace([np.inf, -np.inf], np.nan).dropna()

        user_rows.append(
            {
                "user_id": user_id,
                "user_label": int(original_rows["review_label"].max()),
                "split": str(original_rows["split"].iloc[0]),
                "total_reviews": int(len(original_rows)),
                "avg_rating": float(original_rows["rating"].mean()),
                "rating_std": float(original_rows["rating"].std(ddof=0) if len(original_rows) > 1 else 0.0),
                "rating_entropy": _rating_entropy(original_rows["rating"]),
                "rating_deviation_avg": float(rating_deviation.mean()),
                "rating_deviation_std": float(rating_deviation.std(ddof=0) if len(rating_deviation) > 1 else 0.0),
                "positive_ratio": positive_ratio,
                "negative_ratio": negative_ratio,
                "extreme_rating_ratio": extreme_rating_ratio,
                "max_daily_reviews": max_daily_reviews,
                "burst_ratio": float(max_daily_reviews / max(len(original_rows), 1)),
                "active_days": int(review_dates.dt.date.nunique(dropna=True) or 1),
                "user_tenure_days": tenure_days,
                "avg_review_gap_days": avg_gap_days,
                "std_review_gap_days": std_gap_days,
                "avg_review_time_lag_days": float(review_time_lags.mean() if not review_time_lags.empty else 0.0),
                "std_review_time_lag_days": float(review_time_lags.std(ddof=0) if len(review_time_lags) > 1 else 0.0),
                "avg_review_length": float(np.mean([len(text.split()) for text in original_rows["review_text"]])),
                "product_set": sorted(set(original_rows["product_id"].astype(str))),
                "time_bucket_set": sorted(set(user_reviews["time_bucket"].astype(str))),
            }
        )
        user_abnormal_by_id[user_id] = abnormal_vector.astype(np.float32)
        user_text_by_id[user_id] = text_vector.astype(np.float32)

    user_df = pd.DataFrame(user_rows).sort_values("user_id").reset_index(drop=True)
    user_df = _add_behavior_anomaly_indicators(user_df)
    ordered_user_ids = user_df["user_id"].astype(str).tolist()
    user_abnormal_matrix = np.asarray([user_abnormal_by_id[user_id] for user_id in ordered_user_ids], dtype=np.float32)
    user_text_matrix = np.asarray([user_text_by_id[user_id] for user_id in ordered_user_ids], dtype=np.float32)
    np.save(vector_dir / "user_abnormal_vectors_initial.npy", user_abnormal_matrix)
    np.save(vector_dir / "user_abnormal_vectors.npy", user_abnormal_matrix)
    np.save(vector_dir / "user_text_vectors.npy", user_text_matrix)
    user_df.to_csv(vector_dir / "user_summary.csv", index=False)

    return review_features, user_df, review_vectors, user_abnormal_matrix, user_text_matrix


def _build_shared_entity_edges(
    user_df: pd.DataFrame,
    entity_column: str,
    edge_type: str,
    top_k: int | None,
) -> pd.DataFrame:
    user_entities = {
        str(record["user_id"]): _as_string_set(record[entity_column])
        for record in user_df[["user_id", entity_column]].to_dict("records")
    }
    entity_to_users: dict[str, list[str]] = defaultdict(list)
    for user_id, entities in user_entities.items():
        for entity in entities:
            entity_to_users[str(entity)].append(user_id)

    edge_rows: list[dict[str, Any]] = []
    total_users = max(len(user_entities), 1)
    entity_idf = {
        entity: _normalized_idf(total_users, len(users))
        for entity, users in entity_to_users.items()
    }
    neighbor_limit = None if top_k is None else int(top_k)
    if neighbor_limit is not None and neighbor_limit <= 0:
        neighbor_limit = None

    for user_id, entities in tqdm(user_entities.items(), desc=f"Building {edge_type} edges"):
        counter: Counter[str] = Counter()
        score_counter: Counter[str] = Counter()
        for entity in entities:
            for neighbor_id in entity_to_users[str(entity)]:
                if neighbor_id != user_id:
                    counter[neighbor_id] += 1
                    score_counter[neighbor_id] += entity_idf[str(entity)]
        if not counter:
            continue

        scored_neighbors = []
        user_size = max(len(entities), 1)
        for neighbor_id, shared_count in counter.items():
            neighbor_size = max(len(user_entities[neighbor_id]), 1)
            weight = float(score_counter[neighbor_id]) / np.sqrt(user_size * neighbor_size)
            scored_neighbors.append((neighbor_id, _clip01(weight), int(shared_count)))
        scored_neighbors.sort(key=lambda item: item[1], reverse=True)

        # Senior-mode entity graphs should retain the full shared-entity
        # neighborhood instead of silently falling back to top-k truncation.
        selected_neighbors = scored_neighbors if neighbor_limit is None else scored_neighbors[:neighbor_limit]
        for neighbor_id, weight, shared_count in selected_neighbors:
            edge_rows.append(
                {
                    "src_user_id": user_id,
                    "dst_user_id": neighbor_id,
                    "edge_type": edge_type,
                    "edge_weight": weight,
                    "shared_entity_count": shared_count,
                }
            )
    return pd.DataFrame(edge_rows, columns=EDGE_COLUMNS + ["shared_entity_count"])


def _build_upu_edges(review_features: pd.DataFrame, user_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    if review_features is None or review_features.empty:
        return _build_shared_entity_edges(user_df, entity_column="product_set", edge_type="UPU", top_k=top_k)

    work = review_features[["user_id", "product_id", "rating", "review_datetime"]].copy()
    work["user_id"] = work["user_id"].astype(str)
    work["product_id"] = work["product_id"].astype(str)
    work["review_day"] = work["review_datetime"].apply(_review_day_number)
    summary = work.groupby(["user_id", "product_id"], as_index=False).agg(
        avg_rating=("rating", "mean"),
        review_day=("review_day", "mean"),
    )
    if summary.empty:
        return _empty_edge_frame(["shared_product_count", "S_product_idf", "S_rating", "S_time"])

    product_to_users = summary.groupby("product_id")["user_id"].apply(list).to_dict()
    user_products = summary.groupby("user_id")["product_id"].apply(list).to_dict()
    record_lookup = {
        (str(row.user_id), str(row.product_id)): (float(row.avg_rating), float(row.review_day))
        for row in summary.itertuples(index=False)
    }
    total_users = max(int(user_df["user_id"].nunique()), 1)
    product_idf = {
        product_id: _normalized_idf(total_users, len(set(users)))
        for product_id, users in product_to_users.items()
    }

    rows: list[dict[str, Any]] = []
    for user_id, products in tqdm(user_products.items(), desc="Building UPU edges"):
        score_by_neighbor: defaultdict[str, float] = defaultdict(float)
        count_by_neighbor: Counter[str] = Counter()
        rating_by_neighbor: defaultdict[str, float] = defaultdict(float)
        time_by_neighbor: defaultdict[str, float] = defaultdict(float)
        idf_by_neighbor: defaultdict[str, float] = defaultdict(float)

        for product_id in products:
            src_rating, src_day = record_lookup[(user_id, product_id)]
            idf_score = product_idf.get(product_id, 0.0)
            for neighbor_id in product_to_users.get(product_id, []):
                neighbor_id = str(neighbor_id)
                if neighbor_id == user_id:
                    continue
                dst_rating, dst_day = record_lookup[(neighbor_id, product_id)]
                rating_score = _rating_consistency(src_rating, dst_rating)
                time_score = _time_proximity(src_day, dst_day)
                contribution = idf_score * rating_score * time_score
                score_by_neighbor[neighbor_id] += contribution
                count_by_neighbor[neighbor_id] += 1
                rating_by_neighbor[neighbor_id] += rating_score
                time_by_neighbor[neighbor_id] += time_score
                idf_by_neighbor[neighbor_id] += idf_score

        scored_neighbors: list[dict[str, Any]] = []
        src_size = max(len(products), 1)
        for neighbor_id, raw_score in score_by_neighbor.items():
            shared_count = max(count_by_neighbor[neighbor_id], 1)
            dst_size = max(len(user_products.get(neighbor_id, [])), 1)
            normalized_score = raw_score / np.sqrt(src_size * dst_size)
            scored_neighbors.append(
                {
                    "src_user_id": user_id,
                    "dst_user_id": neighbor_id,
                    "edge_type": "UPU",
                    "edge_weight": _clip01(normalized_score),
                    "shared_product_count": int(shared_count),
                    "S_product_idf": float(idf_by_neighbor[neighbor_id] / shared_count),
                    "S_rating": float(rating_by_neighbor[neighbor_id] / shared_count),
                    "S_time": float(time_by_neighbor[neighbor_id] / shared_count),
                }
            )

        scored_neighbors.sort(key=lambda item: item["edge_weight"], reverse=True)
        rows.extend(scored_neighbors[:top_k])

    return pd.DataFrame(
        rows,
        columns=EDGE_COLUMNS + ["shared_product_count", "S_product_idf", "S_rating", "S_time"],
    )


def _rating_direction_consistency(src_row: pd.Series, dst_row: pd.Series) -> float:
    src_rating = float(src_row["avg_rating"])
    dst_rating = float(dst_row["avg_rating"])
    if _rating_tendency(src_rating) == _rating_tendency(dst_rating):
        return 1.0
    if abs(src_rating - dst_rating) <= 1.0:
        return 0.7
    return 0.25


def _build_utu_edges(user_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    if user_df.empty:
        return _empty_edge_frame(["shared_time_count", "S_time_idf", "S_rating", "S_product"])

    user_lookup_df = user_df.copy()
    user_lookup_df["user_id"] = user_lookup_df["user_id"].astype(str)
    user_lookup: dict[str, dict[str, Any]] = {}
    user_buckets: dict[str, set[str]] = {}
    bucket_to_users: dict[str, list[str]] = defaultdict(list)
    for record in user_lookup_df.to_dict("records"):
        user_id = str(record["user_id"])
        buckets = _as_string_set(record["time_bucket_set"])
        products = _as_string_set(record["product_set"])
        avg_rating = float(record["avg_rating"])
        tendency = _rating_tendency(avg_rating)
        user_lookup[user_id] = {
            "avg_rating": avg_rating,
            "rating_tendency": tendency,
            "product_set": products,
        }
        user_buckets[user_id] = buckets
        for bucket in buckets:
            bucket_to_users[bucket].append(user_id)

    total_users = max(len(user_buckets), 1)
    bucket_idf = {bucket: _normalized_idf(total_users, len(set(users))) for bucket, users in bucket_to_users.items()}

    # UTU is intentionally conservative: broad time buckets are noisy, so each
    # user only checks a small rating-neighborhood inside the same time bucket.
    candidate_window = max(top_k * 4, 60)
    bucket_rating_groups: dict[tuple[str, str], tuple[np.ndarray, list[str]]] = {}
    for bucket, users in bucket_to_users.items():
        tendency_groups: defaultdict[str, list[str]] = defaultdict(list)
        for user_id in users:
            tendency_groups[user_lookup[user_id]["rating_tendency"]].append(user_id)
        for tendency, tendency_users in tendency_groups.items():
            tendency_users.sort(key=lambda uid: user_lookup[uid]["avg_rating"])
            ratings = np.asarray([user_lookup[uid]["avg_rating"] for uid in tendency_users], dtype=np.float32)
            bucket_rating_groups[(bucket, tendency)] = (ratings, tendency_users)

    rows: list[dict[str, Any]] = []
    for user_id, buckets in tqdm(user_buckets.items(), desc="Building UTU edges"):
        score_by_neighbor: defaultdict[str, float] = defaultdict(float)
        count_by_neighbor: Counter[str] = Counter()
        rating_by_neighbor: defaultdict[str, float] = defaultdict(float)
        product_by_neighbor: defaultdict[str, float] = defaultdict(float)
        idf_by_neighbor: defaultdict[str, float] = defaultdict(float)
        src_record = user_lookup[user_id]
        src_products = src_record["product_set"]
        src_rating = float(src_record["avg_rating"])
        src_tendency = str(src_record["rating_tendency"])

        for bucket in buckets:
            idf_score = bucket_idf.get(bucket, 0.0)
            ratings, tendency_users = bucket_rating_groups.get((bucket, src_tendency), (np.asarray([], dtype=np.float32), []))
            if len(tendency_users) == 0:
                continue
            center = int(np.searchsorted(ratings, src_rating, side="left"))
            left = max(0, center - candidate_window)
            right = min(len(tendency_users), center + candidate_window + 1)
            for neighbor_id in tendency_users[left:right]:
                neighbor_id = str(neighbor_id)
                if neighbor_id == user_id:
                    continue
                dst_record = user_lookup[neighbor_id]
                rating_score = _rating_consistency(src_rating, float(dst_record["avg_rating"]))
                dst_products = dst_record["product_set"]
                product_score = 1.0 if src_products & dst_products else 0.5
                contribution = idf_score * rating_score * product_score
                score_by_neighbor[neighbor_id] += contribution
                count_by_neighbor[neighbor_id] += 1
                rating_by_neighbor[neighbor_id] += rating_score
                product_by_neighbor[neighbor_id] += product_score
                idf_by_neighbor[neighbor_id] += idf_score

        scored_neighbors: list[dict[str, Any]] = []
        src_size = max(len(buckets), 1)
        for neighbor_id, raw_score in score_by_neighbor.items():
            shared_count = max(count_by_neighbor[neighbor_id], 1)
            dst_size = max(len(user_buckets.get(neighbor_id, [])), 1)
            normalized_score = raw_score / np.sqrt(src_size * dst_size)
            scored_neighbors.append(
                {
                    "src_user_id": user_id,
                    "dst_user_id": neighbor_id,
                    "edge_type": "UTU",
                    "edge_weight": _clip01(normalized_score),
                    "shared_time_count": int(shared_count),
                    "S_time_idf": float(idf_by_neighbor[neighbor_id] / shared_count),
                    "S_rating": float(rating_by_neighbor[neighbor_id] / shared_count),
                    "S_product": float(product_by_neighbor[neighbor_id] / shared_count),
                }
            )

        scored_neighbors.sort(key=lambda item: item["edge_weight"], reverse=True)
        rows.extend(scored_neighbors[:top_k])

    return pd.DataFrame(
        rows,
        columns=EDGE_COLUMNS + ["shared_time_count", "S_time_idf", "S_rating", "S_product"],
    )


def _build_usu_edges(user_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    if len(user_df) <= 1:
        return _empty_edge_frame(["S_burst", "S_density"])

    work_df = user_df[["user_id", "max_daily_reviews", "active_days", "total_reviews"]].copy()
    work_df["user_id"] = work_df["user_id"].astype(str)
    work_df["burst_signal"] = np.log1p(work_df["max_daily_reviews"].astype(float))
    work_df["density_signal"] = work_df["total_reviews"].astype(float) / work_df["active_days"].clip(lower=1).astype(float)
    work_df["bucket"] = pd.qcut(
        work_df["burst_signal"].rank(method="first"),
        q=min(10, len(work_df)),
        labels=False,
        duplicates="drop",
    )

    rows: list[dict[str, Any]] = []
    bucket_groups = {
        int(bucket): frame.reset_index(drop=True)
        for bucket, frame in work_df.groupby("bucket")
        if pd.notna(bucket)
    }
    for row in tqdm(work_df.itertuples(index=False), total=len(work_df), desc="Building USU edges"):
        if pd.isna(row.bucket):
            continue
        candidate_frames = [bucket_groups.get(int(row.bucket), pd.DataFrame())]
        candidate_frames.append(bucket_groups.get(int(row.bucket) - 1, pd.DataFrame()))
        candidate_frames.append(bucket_groups.get(int(row.bucket) + 1, pd.DataFrame()))
        candidates = pd.concat(candidate_frames, ignore_index=True).drop_duplicates(subset=["user_id"])
        candidates = candidates[candidates["user_id"] != row.user_id]
        if candidates.empty:
            continue

        scores = []
        for candidate in candidates.itertuples(index=False):
            burst_gap = abs(float(row.burst_signal) - float(candidate.burst_signal))
            density_gap = abs(float(row.density_signal) - float(candidate.density_signal))
            burst_score = 1.0 / (1.0 + burst_gap)
            density_score = 1.0 / (1.0 + density_gap)
            weight = 0.5 * burst_score + 0.5 * density_score
            scores.append((candidate.user_id, _clip01(weight), float(burst_score), float(density_score)))
        scores.sort(key=lambda item: item[1], reverse=True)
        for neighbor_id, weight, burst_score, density_score in scores[:top_k]:
            rows.append(
                {
                    "src_user_id": row.user_id,
                    "dst_user_id": neighbor_id,
                    "edge_type": "USU",
                    "edge_weight": weight,
                    "S_burst": burst_score,
                    "S_density": density_score,
                }
            )
    return pd.DataFrame(rows, columns=EDGE_COLUMNS + ["S_burst", "S_density"])


def _build_senior_usu_edges(user_df: pd.DataFrame, senior_usu_ratio: float) -> pd.DataFrame:
    if len(user_df) <= 1:
        return _empty_edge_frame(["S_burst", "S_density"])

    work_df = user_df[["user_id", "burst_ratio", "max_daily_reviews", "active_days", "total_reviews"]].copy()
    work_df["user_id"] = work_df["user_id"].astype(str)
    work_df["burst_ratio"] = pd.to_numeric(work_df["burst_ratio"], errors="coerce").fillna(0.0)
    work_df["density_signal"] = (
        pd.to_numeric(work_df["total_reviews"], errors="coerce").fillna(0.0)
        / pd.to_numeric(work_df["active_days"], errors="coerce").replace(0, 1).fillna(1.0)
    )
    ratio = float(np.clip(senior_usu_ratio, 0.0, 1.0))
    selected_count = max(2, int(np.ceil(len(work_df) * ratio)))
    selected = work_df.sort_values(["burst_ratio", "max_daily_reviews"], ascending=False).head(selected_count).reset_index(drop=True)
    if len(selected) <= 1:
        return _empty_edge_frame(["S_burst", "S_density"])

    rows: list[dict[str, Any]] = []
    for src in selected.itertuples(index=False):
        for dst in selected.itertuples(index=False):
            if src.user_id == dst.user_id:
                continue
            burst_score = _clip01(1.0 - abs(float(src.burst_ratio) - float(dst.burst_ratio)))
            density_score = _clip01(1.0 / (1.0 + abs(float(src.density_signal) - float(dst.density_signal))))
            rows.append(
                {
                    "src_user_id": str(src.user_id),
                    "dst_user_id": str(dst.user_id),
                    "edge_type": "USU",
                    "edge_weight": 1.0,
                    "S_burst": float(burst_score),
                    "S_density": float(density_score),
                }
            )
    return pd.DataFrame(rows, columns=EDGE_COLUMNS + ["S_burst", "S_density"])


def _build_knn_edges(user_ids: list[str], vectors: np.ndarray, edge_type: str, top_k: int) -> pd.DataFrame:
    if len(user_ids) <= 1:
        return _empty_edge_frame()

    normalized = _normalize_vectors(vectors.astype(np.float32))
    nn_model = NearestNeighbors(n_neighbors=min(top_k + 1, len(user_ids)), metric="cosine", algorithm="brute")
    nn_model.fit(normalized)
    distances, indices = nn_model.kneighbors(normalized)

    rows: list[dict[str, Any]] = []
    for src_index, src_user_id in enumerate(user_ids):
        for distance, neighbor_index in zip(distances[src_index][1:], indices[src_index][1:]):
            similarity = _clip01(1.0 - float(distance))
            rows.append(
                {
                    "src_user_id": str(src_user_id),
                    "dst_user_id": str(user_ids[int(neighbor_index)]),
                    "edge_type": edge_type,
                    "edge_weight": similarity,
                }
            )
    return pd.DataFrame(rows, columns=EDGE_COLUMNS)


def _behavior_scores(src_row: pd.Series, dst_row: pd.Series) -> tuple[float, float, float]:
    time_overlap = float(bool(_as_string_set(src_row["time_bucket_set"]) & _as_string_set(dst_row["time_bucket_set"])))
    product_overlap = float(bool(_as_string_set(src_row["product_set"]) & _as_string_set(dst_row["product_set"])))
    rating_gap = abs(float(src_row["avg_rating"]) - float(dst_row["avg_rating"]))
    rating_match = float(
        rating_gap <= 1.0
        or _rating_tendency(float(src_row["avg_rating"])) == _rating_tendency(float(dst_row["avg_rating"]))
    )
    return time_overlap, rating_match, product_overlap


def _build_cb_like_edges(
    user_df: pd.DataFrame,
    candidate_edges: pd.DataFrame,
    vector_score_name: str,
    edge_type: str,
    top_k: int,
    min_vector_score: float | None = None,
    threshold_column: str | None = None,
) -> pd.DataFrame:
    extra_columns = [vector_score_name, "S_time", "S_rating", "S_product"]
    if threshold_column:
        extra_columns.append(threshold_column)
    if candidate_edges.empty:
        return _empty_edge_frame(extra_columns)

    user_lookup = user_df.copy()
    user_lookup["user_id"] = user_lookup["user_id"].astype(str)
    user_lookup = user_lookup.set_index("user_id")
    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in candidate_edges.itertuples(index=False):
        src_user_id = str(row.src_user_id)
        dst_user_id = str(row.dst_user_id)
        if src_user_id not in user_lookup.index or dst_user_id not in user_lookup.index:
            continue
        src_row = user_lookup.loc[src_user_id]
        dst_row = user_lookup.loc[dst_user_id]
        s_time, s_rating, s_product = _behavior_scores(src_row, dst_row)
        vector_score = _clip01(float(row.edge_weight))
        if min_vector_score is not None and vector_score < float(min_vector_score):
            continue
        combined_score = _clip01(0.4 * vector_score + 0.2 * s_time + 0.2 * s_rating + 0.2 * s_product)
        edge_row = {
            "src_user_id": src_user_id,
            "dst_user_id": dst_user_id,
            "edge_type": edge_type,
            "edge_weight": combined_score,
            vector_score_name: vector_score,
            "S_time": s_time,
            "S_rating": s_rating,
            "S_product": s_product,
        }
        if threshold_column:
            edge_row[threshold_column] = float(min_vector_score or 0.0)
        grouped_rows[src_user_id].append(edge_row)

    edge_rows: list[dict[str, Any]] = []
    for rows in grouped_rows.values():
        rows.sort(key=lambda item: item["edge_weight"], reverse=True)
        edge_rows.extend(rows[:top_k])
    base_columns = EDGE_COLUMNS + extra_columns
    return pd.DataFrame(edge_rows, columns=base_columns)


def build_edge_frames(
    user_df: pd.DataFrame,
    user_text_vectors: np.ndarray,
    user_abnormal_vectors: np.ndarray,
    output_dir: str | Path,
    top_k: int,
    review_features: pd.DataFrame | None = None,
    logic_threshold_mode: str = DEFAULT_LOGIC_THRESHOLD_MODE,
    logic_threshold_quantile: float = DEFAULT_LOGIC_THRESHOLD_QUANTILE,
    logic_threshold_value: float = DEFAULT_LOGIC_THRESHOLD_VALUE,
    graph_mode: str = "current",
    senior_usu_ratio: float = 0.10,
) -> dict[str, pd.DataFrame]:
    output_dir = Path(output_dir)
    edge_dir = output_dir / "edges"
    edge_dir.mkdir(parents=True, exist_ok=True)

    user_ids = user_df["user_id"].astype(str).tolist()
    graph_mode = str(graph_mode or "current").lower()

    if graph_mode in {"senior", "senior_enhanced"}:
        upu_edges = _build_shared_entity_edges(
            user_df,
            entity_column="product_set",
            edge_type="UPU",
            top_k=None,
        )
        utu_edges = _build_shared_entity_edges(
            user_df,
            entity_column="time_bucket_set",
            edge_type="UTU",
            top_k=None,
        )
        usu_edges = _build_senior_usu_edges(user_df, senior_usu_ratio=senior_usu_ratio)
    else:
        upu_edges = _build_upu_edges(review_features, user_df, top_k=top_k) if review_features is not None else _build_shared_entity_edges(
            user_df,
            entity_column="product_set",
            edge_type="UPU",
            top_k=top_k,
        )
        utu_edges = _build_utu_edges(user_df, top_k=top_k)
        usu_edges = _build_usu_edges(user_df, top_k=top_k)
    textsim_edges = _build_knn_edges(user_ids, user_text_vectors, edge_type="TextSim", top_k=top_k)
    logic_knn_edges = _build_knn_edges(user_ids, user_abnormal_vectors, edge_type="LogicKNN", top_k=max(top_k * 5, top_k + 10))
    tau_logic = _resolve_logic_threshold(
        candidate_edges=logic_knn_edges,
        mode=logic_threshold_mode,
        quantile=logic_threshold_quantile,
        threshold_value=logic_threshold_value,
    )

    cb_edges = _build_cb_like_edges(
        user_df=user_df,
        candidate_edges=textsim_edges,
        vector_score_name="S_text",
        edge_type="CB",
        top_k=top_k,
    )
    logicae_cb_edges = _build_cb_like_edges(
        user_df=user_df,
        candidate_edges=logic_knn_edges,
        vector_score_name="S_logic",
        edge_type="LogicAE_CB",
        top_k=top_k,
        min_vector_score=tau_logic,
        threshold_column="tau_logic",
    )

    edge_frames = {
        "UPU": upu_edges,
        "UTU": utu_edges,
        "USU": usu_edges,
        "TextSim": textsim_edges,
        "CB": cb_edges,
        "LogicAE_CB": logicae_cb_edges,
    }
    for edge_name, frame in edge_frames.items():
        frame.to_csv(edge_dir / f"{edge_name}_edges.csv", index=False)
    edge_config = {
        "graph_mode": graph_mode,
        "top_k": int(top_k),
        "senior_full_entity_graphs": bool(graph_mode in {"senior", "senior_enhanced"}),
        "senior_usu_ratio": float(senior_usu_ratio),
        "logic_candidate_top_k": int(max(top_k * 5, top_k + 10)),
        "logic_threshold_mode": str(logic_threshold_mode),
        "logic_threshold_quantile": float(logic_threshold_quantile),
        "logic_threshold_value": float(logic_threshold_value),
        "resolved_tau_logic": None if tau_logic is None else float(tau_logic),
        "logic_candidate_edges": int(len(logic_knn_edges)),
        "logicae_cb_edges_after_threshold": int(len(logicae_cb_edges)),
        "senior_undirected_pair_estimates": {
            "UPU": int(len(upu_edges) // 2),
            "UTU": int(len(utu_edges) // 2),
            "USU": int(len(usu_edges) // 2),
        } if graph_mode in {"senior", "senior_enhanced"} else None,
    }
    (edge_dir / "edge_build_config.json").write_text(
        json.dumps(edge_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return edge_frames


def build_relation_aware_support_edges(
    edge_frames: dict[str, pd.DataFrame],
    top_k: int,
    relation_betas: dict[str, float] | None = None,
) -> pd.DataFrame:
    relation_betas = relation_betas or DEFAULT_GRAPH_SUPPORT_BETAS
    combined_scores: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    relation_hits: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)

    for relation_name, beta in relation_betas.items():
        if beta <= 0:
            continue
        frame = edge_frames.get(relation_name, _empty_edge_frame())
        if frame.empty:
            continue
        for row in frame.itertuples(index=False):
            src_user_id = str(row.src_user_id)
            dst_user_id = str(row.dst_user_id)
            contribution = float(beta) * _clip01(float(row.edge_weight))
            combined_scores[src_user_id][dst_user_id] += contribution
            relation_hits[(src_user_id, dst_user_id)][relation_name] = contribution

    rows: list[dict[str, Any]] = []
    relation_columns = [f"beta_{relation_name}" for relation_name in relation_betas]
    for src_user_id, neighbor_scores in combined_scores.items():
        scored_neighbors = sorted(neighbor_scores.items(), key=lambda item: item[1], reverse=True)
        for dst_user_id, score in scored_neighbors[:top_k]:
            row = {
                "src_user_id": src_user_id,
                "dst_user_id": dst_user_id,
                "edge_type": "GraphSupport",
                "edge_weight": _clip01(score),
            }
            hits = relation_hits[(src_user_id, dst_user_id)]
            for relation_name in relation_betas:
                row[f"beta_{relation_name}"] = float(hits.get(relation_name, 0.0))
            rows.append(row)

    return pd.DataFrame(rows, columns=EDGE_COLUMNS + relation_columns)


def apply_graph_guided_evidence_reweighting(
    review_features: pd.DataFrame,
    user_df: pd.DataFrame,
    review_vectors: np.ndarray,
    edge_frames: dict[str, pd.DataFrame],
    output_dir: str | Path,
    top_m: int,
    graph_top_k: int,
    alpha: float = 0.7,
    neighbor_review_cap: int = 20,
    relation_betas: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Use fixed user graph support to reweight review-level abnormal evidence.

    The graph only changes review aggregation weights. It does not rewrite token masks,
    LLM spans, or graph edges, which keeps the feedback path conservative.
    """
    output_dir = Path(output_dir)
    vector_dir = output_dir / "logic_vectors"
    edge_dir = output_dir / "edges"
    vector_dir.mkdir(parents=True, exist_ok=True)
    edge_dir.mkdir(parents=True, exist_ok=True)

    alpha = _clip01(alpha)
    graph_support_edges = build_relation_aware_support_edges(
        edge_frames=edge_frames,
        top_k=graph_top_k,
        relation_betas=relation_betas,
    )
    graph_support_edges.to_csv(edge_dir / "GraphSupport_edges.csv", index=False)

    working_reviews = review_features.sort_values("review_node_id").reset_index(drop=True).copy()
    if len(working_reviews) != review_vectors.shape[0]:
        raise ValueError("Review features and review vector row counts do not match.")

    review_user_ids = working_reviews["user_id"].astype(str).to_numpy()
    evidence_scores = working_reviews["evidence_score"].fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=np.float32)
    normalized_review_vectors = _normalize_vectors(review_vectors.astype(np.float32))

    user_to_review_indices: dict[str, np.ndarray] = {
        user_id: frame.index.to_numpy(dtype=np.int64)
        for user_id, frame in working_reviews.groupby(review_user_ids, sort=False)
    }
    capped_neighbor_indices: dict[str, np.ndarray] = {}
    for user_id, indices in user_to_review_indices.items():
        ranked = indices[np.argsort(-evidence_scores[indices])]
        if neighbor_review_cap > 0:
            ranked = ranked[:neighbor_review_cap]
        capped_neighbor_indices[user_id] = ranked

    neighbor_map: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for row in graph_support_edges.itertuples(index=False):
        weight = _clip01(float(row.edge_weight))
        if weight <= 0:
            continue
        neighbor_map[str(row.src_user_id)].append((str(row.dst_user_id), weight))

    graph_support = np.zeros(len(working_reviews), dtype=np.float32)
    support_weight_sum = np.zeros(len(working_reviews), dtype=np.float32)
    support_neighbor_count = np.zeros(len(working_reviews), dtype=np.int32)

    for user_id, src_indices in tqdm(user_to_review_indices.items(), desc="Graph-guided review support"):
        neighbors = neighbor_map.get(user_id, [])
        if not neighbors:
            continue

        src_vectors = normalized_review_vectors[src_indices]
        for dst_user_id, edge_weight in neighbors:
            dst_indices = capped_neighbor_indices.get(dst_user_id)
            if dst_indices is None or len(dst_indices) == 0:
                continue
            dst_vectors = normalized_review_vectors[dst_indices]
            similarities = np.matmul(src_vectors, dst_vectors.T)
            max_similarities = np.clip(similarities.max(axis=1), 0.0, 1.0)
            graph_support[src_indices] += edge_weight * max_similarities.astype(np.float32)
            support_weight_sum[src_indices] += edge_weight
            support_neighbor_count[src_indices] += 1

    has_support = support_weight_sum > 0
    graph_support[has_support] = graph_support[has_support] / support_weight_sum[has_support]
    effective_graph_support = evidence_scores.copy()
    effective_graph_support[has_support] = graph_support[has_support]
    corrected_scores = np.clip(alpha * evidence_scores + (1.0 - alpha) * effective_graph_support, 0.0, 1.0)

    working_reviews["graph_support_score"] = graph_support
    working_reviews["graph_support_has_neighbor"] = has_support.astype(int)
    working_reviews["graph_support_neighbor_count"] = support_neighbor_count
    working_reviews["corrected_evidence_score"] = corrected_scores

    user_df_out = user_df.copy()
    user_df_out["user_id"] = user_df_out["user_id"].astype(str)
    reweighted_by_user: dict[str, np.ndarray] = {}
    user_support_rows: list[dict[str, Any]] = []

    for user_id in user_df_out["user_id"].tolist():
        indices = user_to_review_indices.get(user_id)
        if indices is None or len(indices) == 0:
            reweighted_by_user[user_id] = np.zeros(review_vectors.shape[1], dtype=np.float32)
            continue

        ranked = indices[np.argsort(-corrected_scores[indices])]
        selected = ranked[:top_m]
        weights = corrected_scores[selected].astype(np.float32)
        if float(weights.sum()) <= 1e-8:
            weights = np.ones_like(weights, dtype=np.float32)
        weights = weights / np.clip(weights.sum(), a_min=1e-8, a_max=None)
        reweighted_by_user[user_id] = np.average(review_vectors[selected], axis=0, weights=weights).astype(np.float32)
        user_support_rows.append(
            {
                "user_id": user_id,
                "graph_support_mean": float(graph_support[indices].mean()),
                "graph_support_coverage": float(has_support[indices].mean()),
                "corrected_evidence_mean": float(corrected_scores[indices].mean()),
                "selected_corrected_evidence_mean": float(corrected_scores[selected].mean()),
            }
        )

    support_summary = pd.DataFrame(user_support_rows)
    if not support_summary.empty:
        user_df_out = user_df_out.merge(support_summary, on="user_id", how="left")
    for column in [
        "graph_support_mean",
        "graph_support_coverage",
        "corrected_evidence_mean",
        "selected_corrected_evidence_mean",
    ]:
        if column not in user_df_out:
            user_df_out[column] = 0.0
        user_df_out[column] = user_df_out[column].fillna(0.0)

    ordered_user_ids = user_df_out["user_id"].astype(str).tolist()
    reweighted_matrix = np.asarray([reweighted_by_user[user_id] for user_id in ordered_user_ids], dtype=np.float32)
    np.save(vector_dir / "user_abnormal_vectors_graph_reweighted.npy", reweighted_matrix)
    working_reviews.to_csv(vector_dir / "review_graph_reweight_scores.csv", index=False)
    user_df_out.to_csv(vector_dir / "user_summary_graph_reweighted.csv", index=False)

    return working_reviews, user_df_out, reweighted_matrix, graph_support_edges


def build_self_feature_matrix(user_df: pd.DataFrame, user_abnormal_vectors: np.ndarray) -> np.ndarray:
    feature_columns = [
        "total_reviews",
        "avg_rating",
        "rating_std",
        "rating_entropy",
        "rating_deviation_avg",
        "rating_deviation_std",
        "positive_ratio",
        "negative_ratio",
        "extreme_rating_ratio",
        "max_daily_reviews",
        "burst_ratio",
        "active_days",
        "user_tenure_days",
        "avg_review_gap_days",
        "std_review_gap_days",
        "avg_review_time_lag_days",
        "std_review_time_lag_days",
        "avg_review_length",
        "RD",
        "EXR",
        "MRO",
        "AD",
        "ATR",
        "behavior_anomaly_score",
    ]
    dense_features = user_df.reindex(columns=feature_columns, fill_value=0.0).fillna(0.0).to_numpy(dtype=np.float32)
    return np.concatenate([dense_features, user_abnormal_vectors.astype(np.float32)], axis=1)


def compute_edge_stats(
    edge_frames: dict[str, pd.DataFrame],
    user_df: pd.DataFrame,
    output_dir: str | Path,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    user_label_map = user_df.set_index(user_df["user_id"].astype(str))["user_label"].astype(int).to_dict()
    stats_rows: list[dict[str, Any]] = []

    for edge_type, frame in edge_frames.items():
        if frame.empty or "src_user_id" not in frame.columns:
            degree_counter: Counter[str] = Counter()
            frame_iterable = []
        else:
            degree_counter = Counter(frame["src_user_id"].astype(str).tolist())
            frame_iterable = frame.itertuples(index=False)
        isolated_count = int(user_df[~user_df["user_id"].astype(str).isin(degree_counter.keys())].shape[0])

        fake_fake = 0
        fake_real = 0
        real_real = 0
        for row in frame_iterable:
            src_label = int(user_label_map.get(str(row.src_user_id), 0))
            dst_label = int(user_label_map.get(str(row.dst_user_id), 0))
            if src_label == 1 and dst_label == 1:
                fake_fake += 1
            elif src_label == 0 and dst_label == 0:
                real_real += 1
            else:
                fake_real += 1

        num_edges = int(len(frame))
        stats_rows.append(
            {
                "edge_type": edge_type,
                "num_edges": num_edges,
                "avg_degree": float(np.mean(list(degree_counter.values())) if degree_counter else 0.0),
                "isolated_user_count": isolated_count,
                "fake_fake_edges": fake_fake,
                "fake_real_edges": fake_real,
                "real_real_edges": real_real,
                "fake_fake_ratio": float(fake_fake / max(num_edges, 1)),
                "fake_real_ratio": float(fake_real / max(num_edges, 1)),
                "real_real_ratio": float(real_real / max(num_edges, 1)),
            }
        )

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(metrics_dir / "edge_stats.csv", index=False)
    return stats_df
