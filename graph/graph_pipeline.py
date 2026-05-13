from __future__ import annotations

import ast
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from sklearn.metrics import accuracy_score, average_precision_score, f1_score, fbeta_score, precision_score, recall_score, roc_auc_score

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
    "TNSGuided_LogicAE_CB": 0.30,
}
DEFAULT_LOGIC_THRESHOLD_MODE = "quantile"
DEFAULT_LOGIC_THRESHOLD_QUANTILE = 0.60
DEFAULT_LOGIC_THRESHOLD_VALUE = 0.30
TNS_HEAVY_CACHE_VERSION = "v2_sparse_safe"


def _empty_edge_frame(extra_columns: list[str] | None = None) -> pd.DataFrame:
    columns = EDGE_COLUMNS + list(extra_columns or [])
    return pd.DataFrame(columns=columns)


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-8, a_max=None)
    return vectors / norms


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(numeric):
        return float(default)
    return float(numeric)


def _normalize_user_id_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            return ""
        if float(value).is_integer():
            return str(int(value))
        return str(value)
    text = str(value).strip()
    if not text:
        return ""
    if re.fullmatch(r"-?\d+(?:\.0+)?", text):
        try:
            return str(int(float(text)))
        except Exception:
            return text
    return text


def _undirected_pair_key(src_user_id: Any, dst_user_id: Any) -> str:
    lhs = _normalize_user_id_value(src_user_id)
    rhs = _normalize_user_id_value(dst_user_id)
    return f"{min(lhs, rhs)}|||{max(lhs, rhs)}"


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


def _simple_sentence_split(text: str) -> list[str]:
    parts = [part.strip() for part in re.split(r"[.!?]+", text or "") if part.strip()]
    return parts or [str(text or "").strip()]


def _simple_word_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", str(text or "").lower())


_FIRST_PERSON_PRONOUNS = {
    "i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves",
}
_SECOND_PERSON_PRONOUNS = {"you", "your", "yours", "yourself", "yourselves"}
_THIRD_PERSON_PRONOUNS = {"he", "she", "it", "they", "them", "his", "her", "their", "theirs", "him", "hers"}
_AFFECTIVE_WORDS = {
    "love", "hate", "amazing", "awful", "terrible", "wonderful", "horrible", "great", "bad", "good", "best", "worst",
    "delicious", "disgusting", "sad", "happy", "angry", "friendly", "rude", "excellent", "poor",
}
_COGNITIVE_WORDS = {
    "think", "know", "believe", "consider", "understand", "realize", "remember", "seem", "guess", "assume",
}
_PERCEPTUAL_WORDS = {
    "see", "hear", "feel", "taste", "smell", "look", "watch", "notice", "sound", "touch",
}


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
        tokenized_reviews = [(_simple_sentence_split(text), _simple_word_tokens(text)) for text in original_rows["review_text"]]
        sentence_lengths = [len(_simple_word_tokens(sentence)) for sentences, _ in tokenized_reviews for sentence in sentences if sentence]
        word_tokens = [token for _, tokens in tokenized_reviews for token in tokens]
        unique_words = len(set(word_tokens))
        total_words = max(len(word_tokens), 1)
        avg_sentence_length = float(np.mean(sentence_lengths) if sentence_lengths else 0.0)
        lexical_diversity = float(unique_words / total_words)
        pronoun_count = float(
            sum(
                1
                for token in word_tokens
                if token in _FIRST_PERSON_PRONOUNS | _SECOND_PERSON_PRONOUNS | _THIRD_PERSON_PRONOUNS
            )
        )
        self_reference_diversity = float(
            (sum(1 for token in word_tokens if token in _FIRST_PERSON_PRONOUNS) / max(total_words, 1))
        )
        affective_tokens = [token for token in word_tokens if token in _AFFECTIVE_WORDS]
        cognitive_tokens = [token for token in word_tokens if token in _COGNITIVE_WORDS]
        perceptual_tokens = [token for token in word_tokens if token in _PERCEPTUAL_WORDS]
        affective_diversity = float(len(set(affective_tokens)) / max(len(affective_tokens), 1))
        cognitive_diversity = float(len(set(cognitive_tokens)) / max(len(cognitive_tokens), 1))
        perceptual_diversity = float(len(set(perceptual_tokens)) / max(len(perceptual_tokens), 1))
        review_texts = original_rows["review_text"].tolist()
        if len(review_texts) > 1:
            reference = review_texts[0]
            ref_tokens = set(_simple_word_tokens(reference))
            similarities = []
            for review_text in review_texts[1:]:
                tokens = set(_simple_word_tokens(review_text))
                union = len(tokens | ref_tokens)
                similarities.append(len(tokens & ref_tokens) / union if union > 0 else 0.0)
            text_similarity = float(np.mean(similarities) if similarities else 0.0)
        else:
            text_similarity = 0.0

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
                "avg_sentence_length": avg_sentence_length,
                "lexical_diversity": lexical_diversity,
                "text_similarity": text_similarity,
                "pronoun_count": pronoun_count,
                "self_reference_diversity": self_reference_diversity,
                "affective_diversity": affective_diversity,
                "cognitive_diversity": cognitive_diversity,
                "perceptual_diversity": perceptual_diversity,
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


def _build_temporal_event_features(
    review_features: pd.DataFrame | None,
    phi_days: int,
) -> pd.DataFrame:
    columns = [
        "src_user_id",
        "dst_user_id",
        "same_product_near_time_count",
        "same_product_same_day_count",
        "same_product_within_phi_count",
        "min_time_gap_days",
        "mean_time_gap_days",
        "burst_session_count",
        "co_burst_group_size_mean",
        "co_burst_group_size_max",
        "temporal_score",
    ]
    if review_features is None or review_features.empty:
        return pd.DataFrame(columns=columns)

    phi_days = max(int(phi_days), 1)
    work = review_features[["user_id", "product_id", "review_datetime"]].copy()
    work["user_id"] = work["user_id"].astype(str)
    work["product_id"] = work["product_id"].astype(str)
    work["review_ts"] = pd.to_datetime(work["review_datetime"], errors="coerce")
    work = work.dropna(subset=["review_ts"])
    if work.empty:
        return pd.DataFrame(columns=columns)

    pair_stats: dict[tuple[str, str], dict[str, Any]] = {}
    near_time_days = max(phi_days * 2, phi_days + 1)

    for product_id, product_reviews in work.groupby("product_id", sort=False):
        ordered = product_reviews.sort_values("review_ts").reset_index(drop=True)
        if len(ordered) <= 1:
            continue
        rows = list(ordered.itertuples(index=False))
        for i, src in enumerate(rows):
            for j in range(i + 1, len(rows)):
                dst = rows[j]
                day_gap = abs((dst.review_ts - src.review_ts).total_seconds()) / 86400.0
                if day_gap > near_time_days:
                    break
                if src.user_id == dst.user_id:
                    continue
                key = tuple(sorted((str(src.user_id), str(dst.user_id))))
                stats = pair_stats.setdefault(
                    key,
                    {
                        "same_product_near_time_count": 0,
                        "same_product_same_day_count": 0,
                        "same_product_within_phi_count": 0,
                        "time_gaps": [],
                        "burst_group_sizes": [],
                    },
                )
                stats["same_product_near_time_count"] += 1
                if day_gap <= 0.0:
                    stats["same_product_same_day_count"] += 1
                if day_gap <= float(phi_days):
                    stats["same_product_within_phi_count"] += 1
                    stats["burst_group_sizes"].append(2)
                stats["time_gaps"].append(day_gap)

    if not pair_stats:
        return pd.DataFrame(columns=columns)

    max_within_phi = max(float(stats["same_product_within_phi_count"]) for stats in pair_stats.values())
    rows: list[dict[str, Any]] = []
    for (src_user_id, dst_user_id), stats in pair_stats.items():
        time_gaps = np.asarray(stats["time_gaps"], dtype=np.float32) if stats["time_gaps"] else np.asarray([], dtype=np.float32)
        min_gap = float(np.min(time_gaps)) if time_gaps.size else float(phi_days + 1)
        mean_gap = float(np.mean(time_gaps)) if time_gaps.size else float(phi_days + 1)
        within_phi = float(stats["same_product_within_phi_count"])
        burst_count = int(within_phi)
        closeness = _clip01(1.0 / (1.0 + min_gap))
        within_phi_norm = _clip01(within_phi / max(max_within_phi, 1.0))
        temporal_score = _clip01(within_phi_norm * closeness)
        burst_sizes = stats["burst_group_sizes"] or [0]
        rows.append(
            {
                "src_user_id": src_user_id,
                "dst_user_id": dst_user_id,
                "same_product_near_time_count": int(stats["same_product_near_time_count"]),
                "same_product_same_day_count": int(stats["same_product_same_day_count"]),
                "same_product_within_phi_count": int(stats["same_product_within_phi_count"]),
                "min_time_gap_days": min_gap,
                "mean_time_gap_days": mean_gap,
                "burst_session_count": burst_count,
                "co_burst_group_size_mean": float(np.mean(burst_sizes)),
                "co_burst_group_size_max": int(np.max(burst_sizes)),
                "temporal_score": temporal_score,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _tns_heavy_cache_dir(phi_days: int) -> Path:
    cache_dir = Path(__file__).resolve().parent / "outputs" / "cache" / f"tns_heavy_features_phi{int(phi_days)}_{TNS_HEAVY_CACHE_VERSION}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _minmax_normalize(values: pd.Series) -> tuple[pd.Series, dict[str, float]]:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0).astype(float)
    min_value = float(numeric.min()) if not numeric.empty else 0.0
    max_value = float(numeric.max()) if not numeric.empty else 0.0
    if max_value - min_value <= 1e-8:
        normalized = pd.Series(np.zeros(len(numeric), dtype=np.float32), index=numeric.index)
    else:
        normalized = ((numeric - min_value) / (max_value - min_value)).clip(0.0, 1.0).astype(np.float32)
    return normalized, {"min": min_value, "max": max_value}


def _build_tns_heavy_feature_cache(
    logic_edges: pd.DataFrame,
    review_features: pd.DataFrame | None,
    user_df: pd.DataFrame,
    user_abnormal_vectors: np.ndarray,
    phi_days: int,
) -> dict[str, Any]:
    cache_dir = _tns_heavy_cache_dir(phi_days)
    pair_path = cache_dir / "pair_tns_heavy_features.csv"
    session_path = cache_dir / "session_features.csv"
    config_path = cache_dir / "tns_feature_config.json"
    stats_path = cache_dir / "tns_feature_stats.json"

    if pair_path.exists() and session_path.exists() and config_path.exists() and stats_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        expected_edge_count = int(len(logic_edges) if logic_edges is not None else 0)
        cache_matches = int(config.get("phi_days", -1)) == int(phi_days) and int(config.get("logic_candidate_edges", -1)) == expected_edge_count
        if cache_matches:
            pair_df = pd.read_csv(pair_path)
            session_df = pd.read_csv(session_path)
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            return {
                "cache_dir": cache_dir,
                "pair_df": pair_df,
                "session_df": session_df,
                "config": config,
                "stats": stats,
                "pair_path": pair_path,
                "session_path": session_path,
                "config_path": config_path,
                "stats_path": stats_path,
            }

    columns = [
        "src_user_id",
        "dst_user_id",
        "logic_score",
        "same_burst_session_count",
        "pair_repeated_temporal_contacts",
        "min_time_gap_days",
        "mean_time_gap_days",
        "temporal_closeness",
        "max_session_group_size",
        "mean_session_group_size",
        "max_session_density",
        "mean_session_density",
        "max_session_fake_prior",
        "mean_session_fake_prior",
        "max_session_logic_consistency",
        "mean_session_logic_consistency",
        "repeated_group_count",
        "group_jaccard_overlap_max",
        "group_jaccard_overlap_mean",
    ]
    if logic_edges is None or logic_edges.empty or review_features is None or review_features.empty:
        pair_df = pd.DataFrame(columns=columns)
        session_df = pd.DataFrame(
            columns=[
                "session_id",
                "product_id",
                "start_time",
                "end_time",
                "duration_days",
                "session_size",
                "session_review_count",
                "session_density",
                "session_unique_products",
                "session_rating_mean",
                "session_rating_std",
                "session_extreme_rating_ratio",
                "session_rating_deviation_mean",
                "session_fake_prior_mean",
                "session_fake_prior_max",
                "session_fake_prior_topk_mean",
                "session_logic_consistency",
                "member_count",
            ]
        )
        config = {
            "phi_days": int(phi_days),
            "logic_candidate_edges": int(len(logic_edges) if logic_edges is not None else 0),
            "pair_feature_count": 0,
            "session_count": 0,
        }
        stats = {"normalization": {}, "notes": "empty_tns_heavy_cache"}
        pair_df.to_csv(pair_path, index=False)
        session_df.to_csv(session_path, index=False)
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
        stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
        return {
            "cache_dir": cache_dir,
            "pair_df": pair_df,
            "session_df": session_df,
            "config": config,
            "stats": stats,
            "pair_path": pair_path,
            "session_path": session_path,
            "config_path": config_path,
            "stats_path": stats_path,
        }

    logic_frame = logic_edges.copy()
    candidate_logic_scores: dict[tuple[str, str], float] = {}
    candidate_neighbors: dict[str, set[str]] = defaultdict(set)
    for row in logic_frame.itertuples(index=False):
        src_user_id = _normalize_user_id_value(row.src_user_id)
        dst_user_id = _normalize_user_id_value(row.dst_user_id)
        key = tuple(sorted((src_user_id, dst_user_id)))
        score = _safe_float(getattr(row, "S_logic", getattr(row, "edge_weight", 0.0)))
        candidate_logic_scores[key] = max(candidate_logic_scores.get(key, 0.0), score)
        candidate_neighbors[src_user_id].add(dst_user_id)
        candidate_neighbors[dst_user_id].add(src_user_id)

    work = review_features.copy()
    work["user_id"] = work["user_id"].apply(_normalize_user_id_value)
    work["product_id"] = work["product_id"].apply(_normalize_user_id_value)
    work["review_ts"] = pd.to_datetime(work.get("review_datetime", work.get("review_date")), errors="coerce")
    work = work.dropna(subset=["review_ts"])
    work["rating"] = pd.to_numeric(work.get("rating", 0.0), errors="coerce").fillna(0.0)
    if "evidence_score" in work.columns:
        prior = pd.to_numeric(work["evidence_score"], errors="coerce").fillna(0.0)
    elif "p_fake_review" in work.columns:
        prior = pd.to_numeric(work["p_fake_review"], errors="coerce").fillna(0.0)
    else:
        prior = pd.Series(np.zeros(len(work), dtype=np.float32), index=work.index)
    if "p_fake_review" in work.columns:
        prior = np.maximum(prior.to_numpy(dtype=np.float32), pd.to_numeric(work["p_fake_review"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32))
        prior = pd.Series(prior, index=work.index, dtype=np.float32)
    work["review_fake_prior"] = pd.to_numeric(prior, errors="coerce").fillna(0.0).clip(0.0, 1.0)
    product_rating_mean = work.groupby("product_id")["rating"].mean().to_dict()

    ordered_user_ids = user_df["user_id"].apply(_normalize_user_id_value).tolist()
    user_index = {user_id: idx for idx, user_id in enumerate(ordered_user_ids)}
    user_abnormal_map = {
        user_id: user_abnormal_vectors[idx].astype(np.float32)
        for idx, user_id in enumerate(ordered_user_ids)
        if idx < len(user_abnormal_vectors)
    }

    pair_stats: dict[tuple[str, str], dict[str, Any]] = {}
    session_rows: list[dict[str, Any]] = []
    session_id = 0
    phi_days = max(int(phi_days), 1)

    def _session_logic_consistency(members: list[str], member_prior_map: dict[str, float]) -> float:
        eligible = [user_id for user_id in members if user_id in user_abnormal_map]
        if len(eligible) <= 1:
            return 0.0
        eligible.sort(key=lambda user_id: member_prior_map.get(user_id, 0.0), reverse=True)
        eligible = eligible[:12]
        vectors = np.asarray([user_abnormal_map[user_id] for user_id in eligible], dtype=np.float32)
        if len(vectors) <= 1:
            return 0.0
        vectors = _normalize_vectors(vectors)
        sim = vectors @ vectors.T
        upper = sim[np.triu_indices(len(vectors), k=1)]
        return float(np.clip(np.mean(upper) if upper.size else 0.0, 0.0, 1.0))

    def _process_session(session_df: pd.DataFrame) -> None:
        nonlocal session_id
        if session_df.empty:
            return
        members = sorted(session_df["user_id"].astype(str).unique().tolist())
        member_first_ts = session_df.groupby("user_id")["review_ts"].min().to_dict()
        member_prior_map = session_df.groupby("user_id")["review_fake_prior"].mean().astype(float).to_dict()
        duration_days = float((session_df["review_ts"].max() - session_df["review_ts"].min()).total_seconds() / 86400.0) if len(session_df) > 1 else 0.0
        session_review_count = int(len(session_df))
        session_size = int(len(members))
        density = float(session_review_count / max(duration_days, 1.0))
        extreme_ratio = float(session_df["rating"].isin([1.0, 5.0]).mean()) if session_review_count else 0.0
        rating_mean = float(session_df["rating"].mean()) if session_review_count else 0.0
        rating_std = float(session_df["rating"].std(ddof=0)) if session_review_count > 1 else 0.0
        rating_dev_mean = float(
            np.mean(
                np.abs(
                    session_df["rating"].to_numpy(dtype=np.float32)
                    - np.asarray([product_rating_mean.get(product_id, rating_mean) for product_id in session_df["product_id"].astype(str)], dtype=np.float32)
                )
            )
        ) if session_review_count else 0.0
        fake_prior_values = session_df["review_fake_prior"].to_numpy(dtype=np.float32)
        topk_fake_prior = np.sort(fake_prior_values)[-min(3, len(fake_prior_values)):] if len(fake_prior_values) else np.asarray([0.0], dtype=np.float32)
        session_logic_consistency = _session_logic_consistency(members, member_prior_map)

        session_rows.append(
            {
                "session_id": session_id,
                "product_id": str(session_df["product_id"].iloc[0]),
                "start_time": str(session_df["review_ts"].min()),
                "end_time": str(session_df["review_ts"].max()),
                "duration_days": duration_days,
                "session_size": session_size,
                "session_review_count": session_review_count,
                "session_density": density,
                "session_unique_products": int(session_df["product_id"].astype(str).nunique()),
                "session_rating_mean": rating_mean,
                "session_rating_std": rating_std,
                "session_extreme_rating_ratio": extreme_ratio,
                "session_rating_deviation_mean": rating_dev_mean,
                "session_fake_prior_mean": float(np.mean(fake_prior_values) if len(fake_prior_values) else 0.0),
                "session_fake_prior_max": float(np.max(fake_prior_values) if len(fake_prior_values) else 0.0),
                "session_fake_prior_topk_mean": float(np.mean(topk_fake_prior)),
                "session_logic_consistency": session_logic_consistency,
                "member_count": session_size,
            }
        )

        if session_size <= 1:
            session_id += 1
            return

        member_set = set(members)
        for src_user_id in members:
            for dst_user_id in candidate_neighbors.get(src_user_id, set()) & member_set:
                if src_user_id >= dst_user_id:
                    continue
                key = tuple(sorted((src_user_id, dst_user_id)))
                if key not in candidate_logic_scores:
                    continue
                time_gap_days = abs((member_first_ts[src_user_id] - member_first_ts[dst_user_id]).total_seconds()) / 86400.0
                stats = pair_stats.setdefault(
                    key,
                    {
                        "same_burst_session_count": 0,
                        "pair_repeated_temporal_contacts": 0,
                        "time_gaps": [],
                        "session_group_sizes": [],
                        "session_densities": [],
                        "session_fake_priors": [],
                        "session_logic_consistencies": [],
                        "session_member_sets": [],
                    },
                )
                stats["same_burst_session_count"] += 1
                stats["pair_repeated_temporal_contacts"] += 1
                stats["time_gaps"].append(float(time_gap_days))
                stats["session_group_sizes"].append(float(session_size))
                stats["session_densities"].append(float(density))
                stats["session_fake_priors"].append(float(np.mean(fake_prior_values) if len(fake_prior_values) else 0.0))
                stats["session_logic_consistencies"].append(float(session_logic_consistency))
                if len(stats["session_member_sets"]) < 8:
                    stats["session_member_sets"].append(set(members))
        session_id += 1

    for _product_id, product_reviews in work.groupby("product_id", sort=False):
        ordered = product_reviews.sort_values("review_ts").reset_index(drop=True)
        if ordered.empty:
            continue
        session_start = 0
        for idx in range(1, len(ordered)):
            gap_days = float((ordered.loc[idx, "review_ts"] - ordered.loc[idx - 1, "review_ts"]).total_seconds() / 86400.0)
            if gap_days > float(phi_days):
                _process_session(ordered.iloc[session_start:idx].copy())
                session_start = idx
        _process_session(ordered.iloc[session_start:].copy())

    pair_rows: list[dict[str, Any]] = []
    for (src_user_id, dst_user_id), stats in pair_stats.items():
        time_gaps = np.asarray(stats["time_gaps"], dtype=np.float32) if stats["time_gaps"] else np.asarray([phi_days + 1.0], dtype=np.float32)
        session_sets = stats["session_member_sets"]
        jaccards: list[float] = []
        if len(session_sets) > 1:
            limit = min(len(session_sets), 6)
            for i in range(limit):
                for j in range(i + 1, limit):
                    lhs = session_sets[i]
                    rhs = session_sets[j]
                    union_size = len(lhs | rhs)
                    if union_size <= 0:
                        continue
                    jaccards.append(float(len(lhs & rhs) / union_size))
        pair_rows.append(
            {
                "src_user_id": src_user_id,
                "dst_user_id": dst_user_id,
                "logic_score": float(candidate_logic_scores.get((src_user_id, dst_user_id), 0.0)),
                "same_burst_session_count": int(stats["same_burst_session_count"]),
                "pair_repeated_temporal_contacts": int(stats["pair_repeated_temporal_contacts"]),
                "min_time_gap_days": float(np.min(time_gaps)),
                "mean_time_gap_days": float(np.mean(time_gaps)),
                "temporal_closeness": _clip01(1.0 / (1.0 + float(np.min(time_gaps)))),
                "max_session_group_size": float(np.max(stats["session_group_sizes"]) if stats["session_group_sizes"] else 0.0),
                "mean_session_group_size": float(np.mean(stats["session_group_sizes"]) if stats["session_group_sizes"] else 0.0),
                "max_session_density": float(np.max(stats["session_densities"]) if stats["session_densities"] else 0.0),
                "mean_session_density": float(np.mean(stats["session_densities"]) if stats["session_densities"] else 0.0),
                "max_session_fake_prior": float(np.max(stats["session_fake_priors"]) if stats["session_fake_priors"] else 0.0),
                "mean_session_fake_prior": float(np.mean(stats["session_fake_priors"]) if stats["session_fake_priors"] else 0.0),
                "max_session_logic_consistency": float(np.max(stats["session_logic_consistencies"]) if stats["session_logic_consistencies"] else 0.0),
                "mean_session_logic_consistency": float(np.mean(stats["session_logic_consistencies"]) if stats["session_logic_consistencies"] else 0.0),
                "repeated_group_count": int(max(len(session_sets) - 1, 0)),
                "group_jaccard_overlap_max": float(np.max(jaccards) if jaccards else 0.0),
                "group_jaccard_overlap_mean": float(np.mean(jaccards) if jaccards else 0.0),
            }
        )

    pair_df = pd.DataFrame(pair_rows, columns=columns)
    normalization_stats: dict[str, dict[str, float]] = {}
    norm_columns = [
        "same_burst_session_count",
        "pair_repeated_temporal_contacts",
        "temporal_closeness",
        "mean_session_fake_prior",
        "mean_session_logic_consistency",
        "group_jaccard_overlap_max",
        "repeated_group_count",
    ]
    for column in norm_columns:
        normalized, stats = _minmax_normalize(pair_df[column] if column in pair_df.columns else pd.Series(dtype=float))
        pair_df[f"{column}_norm"] = normalized
        normalization_stats[column] = stats
    heavy_components = [
        "same_burst_session_count_norm",
        "temporal_closeness_norm",
        "mean_session_fake_prior_norm",
        "mean_session_logic_consistency_norm",
        "group_jaccard_overlap_max_norm",
        "repeated_group_count_norm",
    ]
    pair_df["tns_heavy_score"] = pair_df.reindex(columns=heavy_components, fill_value=0.0).mean(axis=1).astype(np.float32)
    pair_df["has_tns_heavy_evidence"] = (pair_df["same_burst_session_count"] > 0).astype(int)

    session_df = pd.DataFrame(session_rows)
    config = {
        "phi_days": int(phi_days),
        "logic_candidate_edges": int(len(logic_edges)),
        "pair_feature_count": int(len(pair_df)),
        "session_count": int(len(session_df)),
        "notes": "TNS-heavy features are accumulated only for LogicAE candidate pairs to keep runtime bounded.",
    }
    stats = {
        "normalization": normalization_stats,
        "pair_rows": int(len(pair_df)),
        "session_rows": int(len(session_df)),
        "logic_candidate_edges": int(len(logic_edges)),
    }
    pair_df.to_csv(pair_path, index=False)
    session_df.to_csv(session_path, index=False)
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "cache_dir": cache_dir,
        "pair_df": pair_df,
        "session_df": session_df,
        "config": config,
        "stats": stats,
        "pair_path": pair_path,
        "session_path": session_path,
        "config_path": config_path,
        "stats_path": stats_path,
    }




def _build_tns_guided_logicae_edges(
    logic_edges: pd.DataFrame,
    review_features: pd.DataFrame | None,
    top_k: int,
    phi_days: int,
    logic_mode: str,
    logic_lambda: float,
    logic_tns_topk: int,
) -> pd.DataFrame:
    extra_columns = [
        "S_logic",
        "temporal_score",
        "interaction_score",
        "same_product_near_time_count",
        "same_product_same_day_count",
        "same_product_within_phi_count",
        "min_time_gap_days",
        "mean_time_gap_days",
        "burst_session_count",
        "co_burst_group_size_mean",
        "co_burst_group_size_max",
        "tns_phi_days",
        "tns_logic_lambda",
    ]
    if logic_edges is None or logic_edges.empty:
        return _empty_edge_frame(extra_columns)

    temporal_df = _build_temporal_event_features(review_features, phi_days=phi_days)
    if temporal_df.empty:
        empty_temporal = pd.DataFrame(columns=["src_user_id", "dst_user_id", "temporal_score"])
        temporal_df = empty_temporal

    logic_mode = str(logic_mode or "boost").lower()
    logic_lambda = max(float(logic_lambda), 0.0)
    logic_topk = max(int(logic_tns_topk or top_k), 1)

    temporal_lookup = {
        tuple(sorted((str(row.src_user_id), str(row.dst_user_id)))): row._asdict()
        for row in temporal_df.itertuples(index=False)
    }

    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in logic_edges.itertuples(index=False):
        src_user_id = str(row.src_user_id)
        dst_user_id = str(row.dst_user_id)
        key = tuple(sorted((src_user_id, dst_user_id)))
        temporal = temporal_lookup.get(key, {})
        logic_score = _clip01(_safe_float(getattr(row, "S_logic", getattr(row, "edge_weight", 0.0))))
        temporal_score = _clip01(_safe_float(temporal.get("temporal_score", 0.0)))
        if logic_mode == "product":
            interaction_score = _clip01(logic_score * temporal_score)
        else:
            interaction_score = _clip01(logic_score * (1.0 + logic_lambda * temporal_score))
        edge_row = {
            "src_user_id": src_user_id,
            "dst_user_id": dst_user_id,
            "edge_type": "TNSGuided_LogicAE_CB",
            "edge_weight": interaction_score,
            "S_logic": logic_score,
            "temporal_score": temporal_score,
            "interaction_score": interaction_score,
            "same_product_near_time_count": int(temporal.get("same_product_near_time_count", 0)),
            "same_product_same_day_count": int(temporal.get("same_product_same_day_count", 0)),
            "same_product_within_phi_count": int(temporal.get("same_product_within_phi_count", 0)),
            "min_time_gap_days": _safe_float(temporal.get("min_time_gap_days", phi_days + 1)),
            "mean_time_gap_days": _safe_float(temporal.get("mean_time_gap_days", phi_days + 1)),
            "burst_session_count": int(temporal.get("burst_session_count", 0)),
            "co_burst_group_size_mean": _safe_float(temporal.get("co_burst_group_size_mean", 0.0)),
            "co_burst_group_size_max": int(temporal.get("co_burst_group_size_max", 0)),
            "tns_phi_days": int(phi_days),
            "tns_logic_lambda": float(logic_lambda),
        }
        grouped_rows[src_user_id].append(edge_row)

    edge_rows: list[dict[str, Any]] = []
    for rows in grouped_rows.values():
        rows.sort(key=lambda item: item["interaction_score"], reverse=True)
        edge_rows.extend(rows[:logic_topk])
    return pd.DataFrame(edge_rows, columns=EDGE_COLUMNS + extra_columns)


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
    use_tns_guided_logic: bool = False,
    tns_phi_days: int = 5,
    tns_logic_mode: str = "boost",
    tns_logic_lambda: float = 1.0,
    logic_tns_topk: int = 20,
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
    tns_guided_logicae_edges = _build_tns_guided_logicae_edges(
        logic_edges=logicae_cb_edges,
        review_features=review_features,
        top_k=top_k,
        phi_days=tns_phi_days,
        logic_mode=tns_logic_mode,
        logic_lambda=tns_logic_lambda,
        logic_tns_topk=logic_tns_topk,
    ) if use_tns_guided_logic else _empty_edge_frame(
        [
            "S_logic",
            "temporal_score",
            "interaction_score",
            "same_product_near_time_count",
            "same_product_same_day_count",
            "same_product_within_phi_count",
            "min_time_gap_days",
            "mean_time_gap_days",
            "burst_session_count",
            "co_burst_group_size_mean",
            "co_burst_group_size_max",
            "tns_phi_days",
            "tns_logic_lambda",
        ]
    )

    edge_frames = {
        "UPU": upu_edges,
        "UTU": utu_edges,
        "USU": usu_edges,
        "TextSim": textsim_edges,
        "CB": cb_edges,
        "LogicAE_CB": logicae_cb_edges,
        "TNSGuided_LogicAE_CB": tns_guided_logicae_edges,
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
        "use_tns_guided_logic": bool(use_tns_guided_logic),
        "tns_phi_days": int(tns_phi_days),
        "tns_logic_mode": str(tns_logic_mode),
        "tns_logic_lambda": float(tns_logic_lambda),
        "logic_tns_topk": int(logic_tns_topk),
        "tns_guided_logicae_edges": int(len(tns_guided_logicae_edges)),
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
        "avg_sentence_length",
        "lexical_diversity",
        "text_similarity",
        "pronoun_count",
        "self_reference_diversity",
        "affective_diversity",
        "cognitive_diversity",
        "perceptual_diversity",
    ]
    dense_features = user_df.reindex(columns=feature_columns, fill_value=0.0).fillna(0.0).to_numpy(dtype=np.float32)
    return np.concatenate([dense_features, user_abnormal_vectors.astype(np.float32)], axis=1)


def build_user_abnormal_score_vector(
    user_df: pd.DataFrame,
    review_scores_df: pd.DataFrame | None,
    source: str = "auto",
    aggregate: str = "mean",
    top_k: int = 3,
) -> np.ndarray:
    source = str(source or "auto").lower()
    aggregate = str(aggregate or "mean").lower()
    ordered_user_ids = user_df["user_id"].astype(str).tolist()

    if source in {"behavior", "behavior_anomaly_score", "user_abnormal_score"}:
        return (
            user_df.reindex(columns=["behavior_anomaly_score"], fill_value=0.0)
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
            .reshape(-1)
        )

    if review_scores_df is None or review_scores_df.empty:
        return (
            user_df.reindex(columns=["behavior_anomaly_score"], fill_value=0.0)
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
            .reshape(-1)
        )

    review_scores_df = review_scores_df.copy()
    review_scores_df["user_id"] = review_scores_df["user_id"].astype(str)
    has_col = review_scores_df.columns

    if source == "auto":
        if "evidence_score" in has_col:
            source = "logic_ae"
        elif "p_fake_review" in has_col:
            source = "review_fake_score"
        else:
            source = "behavior"

    if source == "review_fake_score":
        score_col = "p_fake_review"
    elif source == "logic_ae":
        score_col = "corrected_evidence_score" if "corrected_evidence_score" in has_col else "evidence_score"
    elif source == "llm_mask":
        score_col = "num_abnormal_patterns"
    elif source in {"evidence_score", "corrected_evidence_score"} and source in has_col:
        score_col = source
    else:
        score_col = "p_fake_review" if "p_fake_review" in has_col else "evidence_score"

    if score_col not in has_col:
        return (
            user_df.reindex(columns=["behavior_anomaly_score"], fill_value=0.0)
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
            .reshape(-1)
        )

    def _aggregate_group(values: pd.Series) -> float:
        numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=np.float32)
        if numeric.size == 0:
            return 0.0
        if aggregate == "max":
            return float(np.max(numeric))
        if aggregate in {"topk_mean", "top_k_mean"}:
            k = max(int(top_k), 1)
            top_values = np.sort(numeric)[-k:]
            return float(np.mean(top_values))
        return float(np.mean(numeric))

    user_scores = review_scores_df.groupby("user_id")[score_col].apply(_aggregate_group).to_dict()
    fallback = user_df.reindex(columns=["behavior_anomaly_score"], fill_value=0.0).fillna(0.0)["behavior_anomaly_score"].astype(float)
    score_vector = np.asarray([float(user_scores.get(user_id, fallback.iloc[idx])) for idx, user_id in enumerate(ordered_user_ids)], dtype=np.float32)
    return np.clip(score_vector, 0.0, 1.0)


def annotate_edges_with_pair_scores(
    edge_frames: dict[str, pd.DataFrame],
    user_df: pd.DataFrame,
    user_abnormal_scores: np.ndarray | None,
    pair_mode: str = "both_high",
) -> dict[str, pd.DataFrame]:
    if user_abnormal_scores is None:
        return edge_frames

    user_ids = user_df["user_id"].astype(str).tolist()
    score_map = {user_id: float(np.clip(score, 0.0, 1.0)) for user_id, score in zip(user_ids, user_abnormal_scores)}
    pair_mode = str(pair_mode or "both_high").lower()
    annotated_frames: dict[str, pd.DataFrame] = {}
    for edge_name, frame in edge_frames.items():
        if frame.empty:
            annotated_frames[edge_name] = frame.copy()
            continue
        work = frame.copy()
        src_scores = work["src_user_id"].astype(str).map(score_map).fillna(0.0).astype(float)
        dst_scores = work["dst_user_id"].astype(str).map(score_map).fillna(0.0).astype(float)
        if pair_mode == "both_high":
            pair_score = np.sqrt(src_scores * dst_scores) * (1.0 - np.abs(src_scores - dst_scores))
        else:
            pair_score = ((src_scores + dst_scores) / 2.0)
        work["pair_abnormal_score"] = np.clip(pair_score, 0.0, 1.0)
        work["abnormal_score_src"] = src_scores
        work["abnormal_score_dst"] = dst_scores
        annotated_frames[edge_name] = work
    return annotated_frames


def apply_abnormal_score_edge_transform(
    edge_frames: dict[str, pd.DataFrame],
    user_df: pd.DataFrame,
    user_abnormal_scores: np.ndarray | None,
    abnormal_edge_eta: float = 0.5,
    pair_mode: str = "both_high",
) -> dict[str, pd.DataFrame]:
    if user_abnormal_scores is None:
        return edge_frames

    abnormal_edge_eta = float(np.clip(abnormal_edge_eta, 0.0, 1.0))
    annotated_frames = annotate_edges_with_pair_scores(edge_frames, user_df, user_abnormal_scores, pair_mode=pair_mode)
    transformed_frames: dict[str, pd.DataFrame] = {}
    for edge_name, frame in annotated_frames.items():
        if frame.empty:
            transformed_frames[edge_name] = frame.copy()
            continue
        work = frame.copy()
        pair_score = work["pair_abnormal_score"].fillna(0.0).astype(float).clip(0.0, 1.0)
        scale = 1.0 + abnormal_edge_eta * (2.0 * pair_score - 1.0)
        scale = np.clip(scale, 1.0 - abnormal_edge_eta, 1.0 + abnormal_edge_eta)
        work["abnormal_gate"] = scale.astype(np.float32)
        work["edge_weight"] = (
            work["edge_weight"].astype(float).clip(lower=0.0) * scale
        ).clip(0.0, 1.0)
        transformed_frames[edge_name] = work
    return transformed_frames


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
                "same_label_ratio": float((fake_fake + real_real) / max(num_edges, 1)),
                "avg_weight": float(pd.to_numeric(frame.get("edge_weight", 0.0), errors="coerce").fillna(0.0).mean() if num_edges > 0 else 0.0),
            }
        )

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(metrics_dir / "edge_stats.csv", index=False)
    return stats_df


def _routek_pair_abnormal_lookup(
    user_df: pd.DataFrame,
    review_features: pd.DataFrame,
    abnormal_score_source: str,
) -> tuple[dict[str, float], callable]:
    user_ids = user_df["user_id"].astype(str).tolist()
    user_abnormal_scores = build_user_abnormal_score_vector(
        user_df=user_df,
        review_scores_df=review_features,
        source=abnormal_score_source,
        aggregate="mean",
        top_k=3,
    )
    user_score_map = {user_id: float(np.clip(score, 0.0, 1.0)) for user_id, score in zip(user_ids, user_abnormal_scores)}

    def _pair_abnormal_score(src_user_id: Any, dst_user_id: Any) -> float:
        src = float(user_score_map.get(_normalize_user_id_value(src_user_id), 0.0))
        dst = float(user_score_map.get(_normalize_user_id_value(dst_user_id), 0.0))
        return float(np.clip(np.sqrt(src * dst) * (1.0 - abs(src - dst)), 0.0, 1.0))

    return user_score_map, _pair_abnormal_score


def _routek_tns_lookup(
    logic_edges: pd.DataFrame,
    review_features: pd.DataFrame,
    user_df: pd.DataFrame,
    user_abnormal_vectors: np.ndarray,
    tns_phi_days: int,
) -> tuple[dict[str, float], dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    tns_heavy_cache = _build_tns_heavy_feature_cache(
        logic_edges=logic_edges,
        review_features=review_features,
        user_df=user_df,
        user_abnormal_vectors=user_abnormal_vectors,
        phi_days=tns_phi_days,
    )
    temporal_cache = _build_temporal_event_features(review_features, phi_days=tns_phi_days)
    temporal_lookup = {
        tuple(sorted((str(row.src_user_id), str(row.dst_user_id)))): row._asdict()
        for row in temporal_cache.itertuples(index=False)
    }
    tns_lookup: dict[str, float] = {}
    pair_df = tns_heavy_cache.get("pair_df", pd.DataFrame())
    if not pair_df.empty:
        for row in pair_df.itertuples(index=False):
            tns_lookup[_undirected_pair_key(row.src_user_id, row.dst_user_id)] = float(
                _safe_float(getattr(row, "tns_heavy_score", getattr(row, "temporal_score", 0.0)))
            )
    return tns_lookup, temporal_lookup, tns_heavy_cache


def _routek_select_topk_with_optional_reserve(
    frame: pd.DataFrame,
    *,
    src_col: str,
    rank_col: str,
    base_col: str,
    k: int,
    use_base_reserve: bool = False,
    base_reserve_ratio: float = 0.30,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    k = max(int(k), 1)
    work = frame.copy()
    work["is_base_reserve_edge"] = 0.0
    work["is_reliability_selected_edge"] = 0.0

    selected_groups: list[pd.DataFrame] = []
    for _src_user_id, group in work.groupby(src_col, sort=False):
        group = group.copy()
        group = group.sort_values([rank_col, "dst_user_id"], ascending=[False, True], kind="mergesort")
        if not use_base_reserve:
            chosen = group.head(k).copy()
            chosen["is_reliability_selected_edge"] = 1.0
            selected_groups.append(chosen)
            continue

        reliability_take = max(1, int(round(k * (1.0 - float(base_reserve_ratio)))))
        reliability_take = min(reliability_take, k)
        base_take = max(0, k - reliability_take)

        rel_sel = group.head(reliability_take).copy()
        rel_sel["is_reliability_selected_edge"] = 1.0

        base_sorted = group.sort_values([base_col, "dst_user_id"], ascending=[False, True], kind="mergesort")
        base_sel = base_sorted.head(base_take).copy()
        base_sel["is_base_reserve_edge"] = 1.0

        merged = pd.concat([rel_sel, base_sel], ignore_index=True)
        merged = merged.drop_duplicates(subset=["src_user_id", "dst_user_id"], keep="first")
        if len(merged) < k:
            filler = group[~group[["src_user_id", "dst_user_id"]].apply(tuple, axis=1).isin(merged[["src_user_id", "dst_user_id"]].apply(tuple, axis=1))]
            filler = filler.head(k - len(merged)).copy()
            filler["is_reliability_selected_edge"] = 1.0
            merged = pd.concat([merged, filler], ignore_index=True)
        selected_groups.append(merged.head(k))

    return pd.concat(selected_groups, ignore_index=True) if selected_groups else work.head(0).copy()


def _routek_compute_threshold_operating_points(
    val_predictions_path: str | Path,
    test_predictions_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    val_df = pd.read_csv(val_predictions_path)
    test_df = pd.read_csv(test_predictions_path)
    if val_df.empty or test_df.empty:
        out = pd.DataFrame(columns=["threshold_mode", "threshold", "AUC", "AP", "F1", "Recall", "Precision"])
        out.to_csv(output_path, index=False)
        return out

    val_labels = val_df["label"].to_numpy(dtype=np.int64)
    val_probs = val_df["prob"].to_numpy(dtype=np.float32)
    test_labels = test_df["label"].to_numpy(dtype=np.int64)
    test_probs = test_df["prob"].to_numpy(dtype=np.float32)

    thresholds = np.unique(np.clip(val_probs, 0.001, 0.999))
    thresholds = np.unique(np.concatenate([[0.5], thresholds]))

    def _metrics_at(th: float) -> dict[str, float]:
        return _routek_safe_binary_metrics(test_labels, test_probs, threshold=float(th))

    best_f1_th = 0.5
    best_f1 = -1.0
    best_f2_th = 0.5
    best_f2 = -1.0
    best_p70_th = 0.5
    best_p70_recall = -1.0
    for th in thresholds:
        val_preds = (val_probs >= th).astype(int)
        val_precision = float(precision_score(val_labels, val_preds, zero_division=0))
        val_recall = float(recall_score(val_labels, val_preds, zero_division=0))
        val_f1 = float(f1_score(val_labels, val_preds, zero_division=0))
        val_f2 = float(fbeta_score(val_labels, val_preds, beta=2.0, zero_division=0))
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_f1_th = float(th)
        if val_f2 > best_f2:
            best_f2 = val_f2
            best_f2_th = float(th)
        if val_precision >= 0.70 and val_recall > best_p70_recall:
            best_p70_recall = val_recall
            best_p70_th = float(th)

    rows = []
    for name, th in [("threshold_f1", best_f1_th), ("threshold_f2", best_f2_th), ("threshold_p70", best_p70_th)]:
        metrics = _metrics_at(float(th))
        rows.append(
            {
                "threshold_mode": name,
                "threshold": float(th),
                "AUC": metrics["auc"],
                "AP": metrics["ap"],
                "F1": metrics["f1"],
                "Recall": metrics["recall"],
                "Precision": metrics["precision"],
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(output_path, index=False)
    return out


def _routek_safe_binary_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, float]:
    preds = (probs >= threshold).astype(int)
    unique_labels = np.unique(labels)
    return {
        "auc": float(roc_auc_score(labels, probs)) if len(unique_labels) > 1 else 0.0,
        "ap": float(average_precision_score(labels, probs)) if len(unique_labels) > 1 else 0.0,
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "accuracy": float(accuracy_score(labels, preds)),
    }


def build_routek_d1main_rns_topk_graph_frames(
    *,
    user_df: pd.DataFrame,
    review_features: pd.DataFrame,
    user_text_vectors: np.ndarray,
    user_abnormal_vectors: np.ndarray,
    d1_edge_frames: dict[str, pd.DataFrame],
    output_dir: str | Path,
    topk_mode: str,
    relation_k: dict[str, int],
    abnormal_score_source: str = "auto",
    alpha_abnormal: float = 0.5,
    beta_tns: float = 0.2,
    gamma_interaction: float = 0.2,
    tns_phi_days: int = 5,
    logic_threshold_mode: str = DEFAULT_LOGIC_THRESHOLD_MODE,
    logic_threshold_quantile: float = DEFAULT_LOGIC_THRESHOLD_QUANTILE,
    logic_threshold_value: float = DEFAULT_LOGIC_THRESHOLD_VALUE,
    candidate_topm: dict[str, int] | None = None,
    use_relation_specific_denoise: bool = False,
    use_base_reserve: bool = False,
    base_reserve_ratio: float = 0.30,
    preserve_d1_for_fixed_k0: bool = False,
) -> dict[str, pd.DataFrame]:
    output_dir = Path(output_dir)
    metrics_dir = output_dir / "metrics"
    edge_dir = output_dir / "edges"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    edge_dir.mkdir(parents=True, exist_ok=True)

    relation_sources = {
        "UPU": "UPU",
        "UTU": "UTU",
        "USU": "USU",
        "LogicAE_CB": "LogicAE_CB",
    }
    relation_k = {key: int(value) for key, value in relation_k.items()}
    candidate_topm = candidate_topm or {}

    if preserve_d1_for_fixed_k0 and str(topk_mode) == "fixed_original":
        fixed_frames = {name: d1_edge_frames.get(name, pd.DataFrame()).copy() for name in relation_sources}
        quality_rows = []
        rank_rows = []
        degree_rows = []
        for relation_name, frame in fixed_frames.items():
            user_label_map = user_df.set_index(user_df["user_id"].astype(str))["user_label"].astype(int).to_dict()
            degree_counter = Counter(frame["src_user_id"].astype(str).tolist()) if not frame.empty else Counter()
            fake_fake = 0
            fake_real = 0
            real_real = 0
            for edge in frame.itertuples(index=False):
                src_label = int(user_label_map.get(str(edge.src_user_id), 0))
                dst_label = int(user_label_map.get(str(edge.dst_user_id), 0))
                if src_label == 1 and dst_label == 1:
                    fake_fake += 1
                elif src_label == 0 and dst_label == 0:
                    real_real += 1
                else:
                    fake_real += 1
            num_edges = int(len(frame))
            base_score = pd.to_numeric(frame.get("edge_weight", 0.0), errors="coerce").fillna(0.0)
            quality_rows.append(
                {
                    "relation": relation_name,
                    "k": int(relation_k.get(relation_name, 20)),
                    "num_edges": num_edges,
                    "avg_degree": float(np.mean(list(degree_counter.values())) if degree_counter else 0.0),
                    "isolated_user_count": int(user_df[~user_df["user_id"].astype(str).isin(degree_counter.keys())].shape[0]),
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
                    "relation": relation_name,
                    "k": int(relation_k.get(relation_name, 20)),
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
                    "relation": relation_name,
                    "k": int(relation_k.get(relation_name, 20)),
                    "num_edges": num_edges,
                    "avg_degree": float(np.mean(list(degree_counter.values())) if degree_counter else 0.0),
                    "isolated_user_count": int(user_df[~user_df["user_id"].astype(str).isin(degree_counter.keys())].shape[0]),
                }
            )
            fixed_frames[relation_name].to_csv(edge_dir / f"{relation_name}_edges.csv", index=False)
        pd.DataFrame(quality_rows).to_csv(metrics_dir / "topk_edge_quality_by_relation.csv", index=False)
        pd.DataFrame(rank_rows).to_csv(metrics_dir / "topk_rank_score_stats.csv", index=False)
        pd.DataFrame(degree_rows).to_csv(metrics_dir / "topk_relation_degree_stats.csv", index=False)
        edge_config = {
            "graph_mode": "current",
            "topk_mode": str(topk_mode),
            "relation_k": relation_k,
            "alpha_abnormal": float(alpha_abnormal),
            "beta_tns": float(beta_tns),
            "gamma_interaction": float(gamma_interaction),
            "tns_phi_days": int(tns_phi_days),
            "abnormal_score_source": str(abnormal_score_source),
            "is_strict_d1_edge_copy": True,
            "notes": "K0_D1Main strict D1 edge reproduction using D1 edge files.",
        }
        (edge_dir / "edge_build_config.json").write_text(json.dumps(edge_config, indent=2, ensure_ascii=False), encoding="utf-8")
        return fixed_frames

    _, pair_abnormal_fn = _routek_pair_abnormal_lookup(
        user_df=user_df,
        review_features=review_features,
        abnormal_score_source=abnormal_score_source,
    )
    base_edge_frames = build_edge_frames(
        user_df=user_df,
        user_text_vectors=user_text_vectors,
        user_abnormal_vectors=user_abnormal_vectors,
        output_dir=output_dir,
        top_k=max(max(candidate_topm.values(), default=20), max(relation_k.values(), default=20)),
        review_features=review_features,
        logic_threshold_mode=logic_threshold_mode,
        logic_threshold_quantile=logic_threshold_quantile,
        logic_threshold_value=logic_threshold_value,
        graph_mode="current",
        senior_usu_ratio=0.10,
        use_tns_guided_logic=False,
    )
    tns_lookup, temporal_lookup, _ = _routek_tns_lookup(
        logic_edges=base_edge_frames.get("LogicAE_CB", pd.DataFrame()),
        review_features=review_features,
        user_df=user_df,
        user_abnormal_vectors=user_abnormal_vectors,
        tns_phi_days=tns_phi_days,
    )

    overlap_rows = []
    relation_specific_rows = []
    quality_rows = []
    rank_rows = []
    degree_rows = []
    selected_frames: dict[str, pd.DataFrame] = {}
    user_label_map = user_df.set_index(user_df["user_id"].astype(str))["user_label"].astype(int).to_dict()

    for relation_name, source_name in relation_sources.items():
        frame = base_edge_frames.get(source_name, pd.DataFrame()).copy()
        d1_frame = d1_edge_frames.get(relation_name, pd.DataFrame()).copy()
        if frame.empty:
            selected_frames[relation_name] = frame
            continue
        frame["src_user_id"] = frame["src_user_id"].astype(str)
        frame["dst_user_id"] = frame["dst_user_id"].astype(str)
        frame["base_score"] = pd.to_numeric(frame.get("edge_weight", 0.0), errors="coerce").fillna(0.0).astype(np.float32)
        frame["abnormal_pair"] = frame.apply(lambda row: pair_abnormal_fn(row["src_user_id"], row["dst_user_id"]), axis=1).astype(np.float32)
        frame["pair_abnormal_score"] = frame["abnormal_pair"].astype(np.float32)
        frame["tns_score"] = frame.apply(
            lambda row: float(tns_lookup.get(_undirected_pair_key(row["src_user_id"], row["dst_user_id"]), 0.0)),
            axis=1,
        ).astype(np.float32)
        frame["abnormal_tns_interaction"] = (frame["abnormal_pair"] * frame["tns_score"]).astype(np.float32)

        if use_relation_specific_denoise:
            if relation_name in {"UPU", "UTU"}:
                alpha = 0.5
                beta = 0.2
                gamma = 0.2
            elif relation_name == "USU":
                alpha = 0.2
                beta = 0.0
                gamma = 0.0
            else:
                alpha = 0.0
                beta = 0.1
                gamma = 0.0
            rank_score = frame["base_score"] * (1.0 + alpha * frame["abnormal_pair"]) * (1.0 + beta * frame["tns_score"]) * (1.0 + gamma * frame["abnormal_tns_interaction"])
        else:
            if str(topk_mode) == "abnormal_aware":
                rank_score = frame["base_score"] * (1.0 + float(alpha_abnormal) * frame["abnormal_pair"])
            elif str(topk_mode) == "abnormal_tns_aware":
                rank_score = (
                    frame["base_score"]
                    * (1.0 + float(alpha_abnormal) * frame["abnormal_pair"])
                    * (1.0 + float(beta_tns) * frame["tns_score"])
                    * (1.0 + float(gamma_interaction) * frame["abnormal_tns_interaction"])
                )
            else:
                rank_score = frame["base_score"]
        frame["rank_score"] = rank_score.astype(np.float32)
        frame["reliability_score"] = frame["rank_score"].astype(np.float32)

        candidate_m = int(candidate_topm.get(relation_name, relation_k.get(relation_name, 20)))
        candidate_m = max(candidate_m, relation_k.get(relation_name, 20))
        frame = frame.sort_values(["src_user_id", "rank_score", "dst_user_id"], ascending=[True, False, True], kind="mergesort")
        frame = frame.groupby("src_user_id", group_keys=False).head(candidate_m).reset_index(drop=True)

        final_frame = _routek_select_topk_with_optional_reserve(
            frame,
            src_col="src_user_id",
            rank_col="rank_score",
            base_col="base_score",
            k=int(relation_k.get(relation_name, 20)),
            use_base_reserve=bool(use_base_reserve),
            base_reserve_ratio=float(base_reserve_ratio),
        ).reset_index(drop=True)
        final_frame["edge_weight_before"] = final_frame["base_score"].astype(np.float32)
        final_frame["edge_weight_after"] = final_frame["base_score"].astype(np.float32)
        final_frame["edge_weight"] = final_frame["base_score"].astype(np.float32)

        selected_frames[relation_name] = final_frame
        final_frame.to_csv(edge_dir / f"{relation_name}_edges.csv", index=False)

        d1_pairs = set(zip(d1_frame["src_user_id"].astype(str), d1_frame["dst_user_id"].astype(str))) if not d1_frame.empty else set()
        new_pairs = set(zip(final_frame["src_user_id"].astype(str), final_frame["dst_user_id"].astype(str)))
        overlap = len(d1_pairs & new_pairs)
        overlap_rows.append(
            {
                "relation": relation_name,
                "candidate_topM": candidate_m,
                "final_k": int(relation_k.get(relation_name, 20)),
                "overlap_with_K0_top20": float(overlap / max(len(d1_pairs), 1)),
                "new_edges_ratio": float(len(new_pairs - d1_pairs) / max(len(new_pairs), 1)),
                "dropped_edges_ratio": float(len(d1_pairs - new_pairs) / max(len(d1_pairs), 1)),
            }
        )
        relation_specific_rows.append(
            {
                "relation": relation_name,
                "candidate_topM": candidate_m,
                "final_k": int(relation_k.get(relation_name, 20)),
                "avg_base_score": float(final_frame["base_score"].mean()) if not final_frame.empty else 0.0,
                "avg_abnormal_pair": float(final_frame["abnormal_pair"].mean()) if not final_frame.empty else 0.0,
                "avg_tns_score": float(final_frame["tns_score"].mean()) if not final_frame.empty else 0.0,
                "avg_rank_score": float(final_frame["rank_score"].mean()) if not final_frame.empty else 0.0,
                "avg_reliability_score": float(final_frame["reliability_score"].mean()) if not final_frame.empty else 0.0,
                "avg_neighbor_distance": float((1.0 - np.clip(final_frame["reliability_score"], 0.0, None) / max(float(final_frame["reliability_score"].max()), 1e-6)).mean()) if not final_frame.empty else 0.0,
            }
        )

        degree_counter = Counter(final_frame["src_user_id"].astype(str).tolist()) if not final_frame.empty else Counter()
        fake_fake = 0
        fake_real = 0
        real_real = 0
        for edge in final_frame.itertuples(index=False):
            src_label = int(user_label_map.get(str(edge.src_user_id), 0))
            dst_label = int(user_label_map.get(str(edge.dst_user_id), 0))
            if src_label == 1 and dst_label == 1:
                fake_fake += 1
            elif src_label == 0 and dst_label == 0:
                real_real += 1
            else:
                fake_real += 1
        num_edges = int(len(final_frame))
        quality_rows.append(
            {
                "relation": relation_name,
                "k": int(relation_k.get(relation_name, 20)),
                "candidate_topM": candidate_m,
                "num_edges": num_edges,
                "avg_degree": float(np.mean(list(degree_counter.values())) if degree_counter else 0.0),
                "isolated_user_count": int(user_df[~user_df["user_id"].astype(str).isin(degree_counter.keys())].shape[0]),
                "same_label_ratio": float((fake_fake + real_real) / max(num_edges, 1)),
                "fake_fake_ratio": float(fake_fake / max(num_edges, 1)),
                "fake_real_ratio": float(fake_real / max(num_edges, 1)),
                "real_real_ratio": float(real_real / max(num_edges, 1)),
                "avg_base_score": float(final_frame["base_score"].mean()) if num_edges else 0.0,
                "avg_abnormal_pair": float(final_frame["abnormal_pair"].mean()) if num_edges else 0.0,
                "avg_tns_score": float(final_frame["tns_score"].mean()) if num_edges else 0.0,
                "avg_rank_score": float(final_frame["rank_score"].mean()) if num_edges else 0.0,
            }
        )
        rank_rows.append(
            {
                "relation": relation_name,
                "k": int(relation_k.get(relation_name, 20)),
                "candidate_topM": candidate_m,
                "min_rank_score": float(final_frame["rank_score"].min()) if num_edges else 0.0,
                "mean_rank_score": float(final_frame["rank_score"].mean()) if num_edges else 0.0,
                "max_rank_score": float(final_frame["rank_score"].max()) if num_edges else 0.0,
                "mean_base_score": float(final_frame["base_score"].mean()) if num_edges else 0.0,
                "mean_abnormal_pair": float(final_frame["abnormal_pair"].mean()) if num_edges else 0.0,
                "mean_tns_score": float(final_frame["tns_score"].mean()) if num_edges else 0.0,
            }
        )
        degree_rows.append(
            {
                "relation": relation_name,
                "k": int(relation_k.get(relation_name, 20)),
                "candidate_topM": candidate_m,
                "num_edges": num_edges,
                "avg_degree": float(np.mean(list(degree_counter.values())) if degree_counter else 0.0),
                "isolated_user_count": int(user_df[~user_df["user_id"].astype(str).isin(degree_counter.keys())].shape[0]),
            }
        )

    pd.DataFrame(quality_rows).to_csv(metrics_dir / "topk_edge_quality_by_relation.csv", index=False)
    pd.DataFrame(rank_rows).to_csv(metrics_dir / "topk_rank_score_stats.csv", index=False)
    pd.DataFrame(degree_rows).to_csv(metrics_dir / "topk_relation_degree_stats.csv", index=False)
    if overlap_rows:
        pd.DataFrame(overlap_rows).to_csv(metrics_dir / "topk_selection_overlap.csv", index=False)
    if relation_specific_rows:
        pd.DataFrame(relation_specific_rows).to_csv(metrics_dir / "k2s_relation_specific_stats.csv", index=False)

    edge_config = {
        "graph_mode": "current",
        "topk_mode": str(topk_mode),
        "relation_k": relation_k,
        "candidate_topM": {key: int(value) for key, value in candidate_topm.items()},
        "alpha_abnormal": float(alpha_abnormal),
        "beta_tns": float(beta_tns),
        "gamma_interaction": float(gamma_interaction),
        "tns_phi_days": int(tns_phi_days),
        "abnormal_score_source": str(abnormal_score_source),
        "use_relation_specific_denoise": bool(use_relation_specific_denoise),
        "use_base_reserve": bool(use_base_reserve),
        "base_reserve_ratio": float(base_reserve_ratio),
        "preserve_d1_for_fixed_k0": bool(preserve_d1_for_fixed_k0),
        "notes": "Route K D1Main/RNS graph using D1-aligned features and current candidate edges.",
    }
    (edge_dir / "edge_build_config.json").write_text(json.dumps(edge_config, indent=2, ensure_ascii=False), encoding="utf-8")
    return selected_frames


