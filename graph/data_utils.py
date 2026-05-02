from __future__ import annotations

import csv
import gzip
import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


LABEL_LEAKAGE_COLUMNS = {
    "label",
    "is_fake",
    "is_real",
    "review_label",
    "user_label",
    "fake_review_count",
    "fake_ratio",
    "is_fake_user",
}


@dataclass
class PreparedGraphData:
    review_df: pd.DataFrame
    user_df: pd.DataFrame
    split_df: pd.DataFrame
    preprocessing_stats: dict[str, Any]
    source_path: Path
    canonical_csv_path: Path
    split_csv_path: Path
    legacy_tsv_dir: Path
    metadata_path: Path


def ensure_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def _find_column(columns: list[str], aliases: list[str]) -> str | None:
    normalized = {_normalize_name(col): col for col in columns}
    for alias in aliases:
        hit = normalized.get(_normalize_name(alias))
        if hit is not None:
            return hit
    return None


def combine_part_files(part_files: list[Path], output_path: str | Path, overwrite: bool = False) -> Path:
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        return output_path
    ensure_dir(output_path.parent)
    with output_path.open("wb") as out_file:
        for part_file in sorted(part_files):
            with part_file.open("rb") as in_file:
                shutil.copyfileobj(in_file, out_file)
    return output_path


def maybe_extract_gzip(gzip_path: str | Path, output_csv_path: str | Path, overwrite: bool = False) -> Path:
    gzip_path = Path(gzip_path)
    output_csv_path = Path(output_csv_path)
    if output_csv_path.exists() and not overwrite:
        return output_csv_path
    ensure_dir(output_csv_path.parent)
    with gzip.open(gzip_path, "rb") as src, output_csv_path.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    return output_csv_path


def resolve_yelpzip_source(
    graph_data_dir: str | Path,
    raw_output_dir: str | Path,
    explicit_data_path: str | Path | None = None,
    prefer_corrected_reviews: bool = True,
    overwrite_combined_files: bool = False,
) -> tuple[Path, dict[str, Any]]:
    graph_data_dir = Path(graph_data_dir)
    raw_output_dir = ensure_dir(raw_output_dir)

    if explicit_data_path is not None:
        explicit_path = Path(explicit_data_path)
        if explicit_path.exists():
            return explicit_path, {"source_mode": "explicit"}
        raise FileNotFoundError(f"Explicit YelpZip data path not found: {explicit_path}")

    metadata: dict[str, Any] = {"source_mode": "auto"}
    corrected_gzip = graph_data_dir / "YelpZip_reviews_correct.csv.gz"
    corrected_parts = sorted(graph_data_dir.glob("YelpZip_reviews_correct.csv.gz.part.*"))
    dataset_csv = graph_data_dir / "dataset" / "yelpzip.csv"

    if prefer_corrected_reviews and (corrected_gzip.exists() or corrected_parts):
        combined_gzip = raw_output_dir / "YelpZip_reviews_correct.csv.gz"
        if corrected_gzip.exists():
            if overwrite_combined_files or not combined_gzip.exists():
                shutil.copyfile(corrected_gzip, combined_gzip)
            metadata["corrected_input"] = str(corrected_gzip)
        else:
            combine_part_files(corrected_parts, combined_gzip, overwrite=overwrite_combined_files)
            metadata["corrected_parts"] = [str(part) for part in corrected_parts]

        combined_csv = raw_output_dir / "YelpZip_reviews_correct.csv"
        maybe_extract_gzip(combined_gzip, combined_csv, overwrite=overwrite_combined_files)
        metadata["resolved_path"] = str(combined_csv)
        metadata["resolved_variant"] = "corrected_reviews"
        return combined_csv, metadata

    if dataset_csv.exists():
        metadata["resolved_path"] = str(dataset_csv)
        metadata["resolved_variant"] = "dataset_yelpzip_csv"
        return dataset_csv, metadata

    raise FileNotFoundError(
        "Unable to resolve YelpZip source. Expected one of: explicit data path, "
        "graph data/YelpZip_reviews_correct.csv.gz(.part.*), or graph data/dataset/yelpzip.csv."
    )


def _derive_review_label(df: pd.DataFrame, label_col: str | None, tag_col: str | None) -> pd.Series:
    fake_words = {"fake", "spam", "fraud", "deceptive", "suspicious", "shill"}
    real_words = {"real", "genuine", "truthful", "organic", "authentic", "legit"}

    if tag_col is not None:
        normalized_tag = (
            df[tag_col]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )
        mapped = normalized_tag.map(
            lambda value: 1 if value in fake_words else 0 if value in real_words else np.nan
        )
        if mapped.notna().mean() > 0.95:
            return mapped.fillna(0).astype(int)

    if label_col is None:
        raise ValueError("Unable to infer review labels because no label-like column was found.")

    series = df[label_col]
    numeric = pd.to_numeric(series, errors="coerce")
    unique_values = sorted(set(value for value in numeric.dropna().unique().tolist()))

    if set(unique_values) == {-1.0, 1.0}:
        return (numeric < 0).astype(int)
    if set(unique_values) == {0.0, 1.0}:
        return numeric.astype(int)

    normalized_text = series.fillna("").astype(str).str.strip().str.lower()
    mapped = normalized_text.map(
        lambda value: 1 if value in fake_words else 0 if value in real_words else np.nan
    )
    if mapped.notna().mean() > 0.95:
        return mapped.fillna(0).astype(int)

    raise ValueError(
        "Unable to infer fake/real label semantics automatically. "
        f"Observed values in '{label_col}': {unique_values[:10]}"
    )


def standardize_review_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    unnamed_columns = [col for col in df.columns if _normalize_name(col).startswith("unnamed")]
    if unnamed_columns:
        df = df.drop(columns=unnamed_columns)

    columns = list(df.columns)
    user_col = _find_column(columns, ["user_id", "review_id", "reviewer_id"])
    product_col = _find_column(columns, ["product_id", "prod_id", "business_id", "asin", "parent_asin", "item_id"])
    rating_col = _find_column(columns, ["rating", "stars", "score", "overall"])
    label_col = _find_column(columns, ["label", "is_fake", "is_real", "spam_label"])
    tag_col = _find_column(columns, ["tag", "label_name", "class_name"])
    date_col = _find_column(columns, ["review_date", "date", "timestamp", "time", "review_time"])
    text_col = _find_column(columns, ["review_text", "text", "content", "body", "review"])

    missing = {
        "user_id": user_col,
        "product_id": product_col,
        "rating": rating_col,
        "review_text": text_col,
    }
    missing_required = [name for name, col in missing.items() if col is None]
    if missing_required:
        raise ValueError(f"Missing required review columns after schema inference: {missing_required}")

    standardized = pd.DataFrame(
        {
            "review_node_id": np.arange(len(df), dtype=np.int64),
            "user_id": df[user_col].astype(str).str.strip(),
            "product_id": df[product_col].astype(str).str.strip(),
            "rating": pd.to_numeric(df[rating_col], errors="coerce").fillna(0.0).astype(float),
            "review_text": df[text_col].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip(),
        }
    )
    standardized["review_date"] = (
        df[date_col].fillna("").astype(str).str.strip() if date_col is not None else ""
    )
    standardized["review_label"] = _derive_review_label(df, label_col=label_col, tag_col=tag_col)
    standardized["review_datetime"] = pd.to_datetime(standardized["review_date"], errors="coerce", utc=False)

    standardized = standardized[standardized["user_id"] != ""].copy()
    standardized = standardized[standardized["product_id"] != ""].copy()
    standardized = standardized[standardized["review_text"] != ""].copy()
    standardized["review_node_id"] = np.arange(len(standardized), dtype=np.int64)

    return standardized


def build_user_frame(review_df: pd.DataFrame) -> pd.DataFrame:
    user_summary = (
        review_df.groupby("user_id")
        .agg(
            total_reviews=("review_node_id", "count"),
            fake_review_count=("review_label", "sum"),
            avg_rating=("rating", "mean"),
            rating_std=("rating", "std"),
            avg_review_length=("review_text", lambda values: float(np.mean([len(text.split()) for text in values]))),
        )
        .reset_index()
    )
    user_summary["rating_std"] = user_summary["rating_std"].fillna(0.0)
    user_summary["user_label"] = (user_summary["fake_review_count"] >= 1).astype(int)
    return user_summary


def prune_inactive_and_conflicting_entities(
    review_df: pd.DataFrame,
    min_user_reviews: int = 3,
    min_product_reviews: int = 3,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    work_df = review_df.copy()
    initial_review_count = int(len(work_df))
    initial_user_count = int(work_df["user_id"].nunique())
    initial_product_count = int(work_df["product_id"].nunique())

    def _prune_inactive(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        iteration_count = 0
        while True:
            iteration_count += 1
            before_count = len(df)

            if min_user_reviews > 1:
                user_counts = df.groupby("user_id")["review_node_id"].count()
                active_users = set(user_counts[user_counts >= min_user_reviews].index.astype(str))
                df = df[df["user_id"].isin(active_users)].copy()

            if min_product_reviews > 1:
                product_counts = df.groupby("product_id")["review_node_id"].count()
                active_products = set(product_counts[product_counts >= min_product_reviews].index.astype(str))
                df = df[df["product_id"].isin(active_products)].copy()

            if len(df) == before_count:
                return df, iteration_count

    work_df, pre_conflict_iterations = _prune_inactive(work_df)
    conflict_user_ids: list[str] = []
    if not work_df.empty:
        label_variety = work_df.groupby("user_id")["review_label"].nunique()
        conflict_user_ids = label_variety[label_variety > 1].index.astype(str).tolist()
        if conflict_user_ids:
            work_df = work_df[~work_df["user_id"].isin(conflict_user_ids)].copy()

    work_df, post_conflict_iterations = _prune_inactive(work_df)

    work_df = work_df.reset_index(drop=True)
    work_df["review_node_id"] = np.arange(len(work_df), dtype=np.int64)

    final_user_counts = work_df.groupby("user_id")["review_node_id"].count() if not work_df.empty else pd.Series(dtype=int)
    final_product_counts = work_df.groupby("product_id")["review_node_id"].count() if not work_df.empty else pd.Series(dtype=int)

    stats = {
        "min_user_reviews": int(min_user_reviews),
        "min_product_reviews": int(min_product_reviews),
        "inactive_prune_iterations_before_conflict": int(pre_conflict_iterations),
        "inactive_prune_iterations_after_conflict": int(post_conflict_iterations),
        "initial_review_count": initial_review_count,
        "initial_user_count": initial_user_count,
        "initial_product_count": initial_product_count,
        "conflicting_user_count": int(len(conflict_user_ids)),
        "conflicting_user_ids_sample": conflict_user_ids[:20],
        "final_review_count": int(len(work_df)),
        "final_user_count": int(work_df["user_id"].nunique()),
        "final_product_count": int(work_df["product_id"].nunique()),
        "removed_review_count": int(initial_review_count - len(work_df)),
        "removed_user_count": int(initial_user_count - work_df["user_id"].nunique()),
        "removed_product_count": int(initial_product_count - work_df["product_id"].nunique()),
        "min_reviews_per_user_after_filter": int(final_user_counts.min()) if not final_user_counts.empty else 0,
        "min_reviews_per_product_after_filter": int(final_product_counts.min()) if not final_product_counts.empty else 0,
    }
    return work_df, stats


def _stratified_split(
    items: pd.DataFrame,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stratify = None
    label_counts = items["user_label"].value_counts()
    if len(label_counts) > 1 and label_counts.min() >= 2:
        stratify = items["user_label"]
    left, right = train_test_split(
        items,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )
    return left.reset_index(drop=True), right.reset_index(drop=True)


def split_users(
    user_df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> pd.DataFrame:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if not math.isclose(ratio_sum, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    base_df = user_df[["user_id", "user_label"]].drop_duplicates().reset_index(drop=True)
    train_df, temp_df = _stratified_split(base_df, test_size=(1.0 - train_ratio), seed=seed)
    relative_test_size = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = _stratified_split(temp_df, test_size=relative_test_size, seed=seed + 1)

    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")
    return pd.concat([train_df, val_df, test_df], ignore_index=True)


def sample_users_for_smoke(user_df: pd.DataFrame, max_users: int, seed: int) -> pd.DataFrame:
    if max_users <= 0 or len(user_df) <= max_users:
        return user_df

    buckets = []
    rng = np.random.default_rng(seed)
    for label_value, bucket_df in user_df.groupby("user_label"):
        bucket_df = bucket_df.sample(frac=1.0, random_state=seed + int(label_value)).reset_index(drop=True)
        buckets.append(bucket_df)

    sampled = []
    while len(sampled) < max_users:
        progressed = False
        for bucket_df in buckets:
            if not bucket_df.empty and len(sampled) < max_users:
                row = bucket_df.iloc[0]
                sampled.append(row)
                bucket_df.drop(bucket_df.index[0], inplace=True)
                progressed = True
        if not progressed:
            break

    sampled_df = pd.DataFrame(sampled).drop_duplicates(subset=["user_id"]).reset_index(drop=True)
    if len(sampled_df) < max_users:
        remaining = user_df[~user_df["user_id"].isin(sampled_df["user_id"])]
        if not remaining.empty:
            extra = remaining.sample(
                n=min(max_users - len(sampled_df), len(remaining)),
                random_state=seed,
            )
            sampled_df = pd.concat([sampled_df, extra], ignore_index=True)
    return sampled_df.head(max_users).reset_index(drop=True)


def sample_users_balanced(
    user_df: pd.DataFrame,
    target_total_users: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if user_df.empty:
        return user_df, {
            "requested_total_users": int(target_total_users),
            "applied": False,
            "reason": "empty_user_frame",
        }

    work_df = user_df.drop_duplicates(subset=["user_id"]).reset_index(drop=True)
    label_counts = work_df["user_label"].value_counts().to_dict()
    fake_count = int(label_counts.get(1, 0))
    real_count = int(label_counts.get(0, 0))
    if fake_count <= 0 or real_count <= 0:
        return work_df, {
            "requested_total_users": int(target_total_users),
            "available_fake_users": fake_count,
            "available_real_users": real_count,
            "applied": False,
            "reason": "single_class_after_filtering",
        }

    requested_per_class = max(int(target_total_users) // 2, 1) if int(target_total_users) > 0 else min(fake_count, real_count)
    actual_per_class = min(fake_count, real_count, requested_per_class)

    sampled_parts = []
    for label_value in (1, 0):
        label_df = work_df[work_df["user_label"] == label_value]
        if len(label_df) <= actual_per_class:
            sampled_parts.append(label_df.copy())
        else:
            sampled_parts.append(
                label_df.sample(n=actual_per_class, random_state=seed + int(label_value)).copy()
            )

    sampled_df = (
        pd.concat(sampled_parts, ignore_index=True)
        .drop_duplicates(subset=["user_id"])
        .sort_values("user_id")
        .reset_index(drop=True)
    )
    stats = {
        "requested_total_users": int(target_total_users),
        "requested_per_class": int(requested_per_class),
        "available_fake_users": fake_count,
        "available_real_users": real_count,
        "actual_fake_users": int((sampled_df["user_label"] == 1).sum()),
        "actual_real_users": int((sampled_df["user_label"] == 0).sum()),
        "actual_total_users": int(len(sampled_df)),
        "applied": True,
    }
    return sampled_df, stats


def write_legacy_textcls_splits(review_df: pd.DataFrame, split_df: pd.DataFrame, output_dir: str | Path) -> Path:
    output_dir = ensure_dir(output_dir)
    if "split" in review_df.columns:
        merged = review_df.copy()
    else:
        merged = review_df.merge(split_df[["user_id", "split"]], on="user_id", how="left")
    split_to_file = {"train": "train.tsv", "val": "dev.tsv", "test": "test.tsv"}

    for split_name, file_name in split_to_file.items():
        split_rows = merged[merged["split"] == split_name]
        target_path = output_dir / file_name
        with target_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            for row in split_rows.itertuples(index=False):
                safe_text = str(row.review_text).replace("\t", " ").replace("\r", " ").replace("\n", " ")
                writer.writerow([safe_text, str(int(row.review_label))])
    return output_dir


def prepare_graph_data(
    graph_data_dir: str | Path,
    output_dir: str | Path,
    data_path: str | Path | None,
    seed: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    min_user_reviews: int = 3,
    min_product_reviews: int = 3,
    prefer_corrected_reviews: bool = True,
    overwrite_combined_files: bool = False,
    smoke_max_users: int = 0,
    balance_user_labels: bool = False,
    balanced_user_count: int = 0,
) -> PreparedGraphData:
    output_dir = ensure_dir(output_dir)
    raw_output_dir = ensure_dir(output_dir / "raw")
    source_path, source_meta = resolve_yelpzip_source(
        graph_data_dir=graph_data_dir,
        raw_output_dir=raw_output_dir,
        explicit_data_path=data_path,
        prefer_corrected_reviews=prefer_corrected_reviews,
        overwrite_combined_files=overwrite_combined_files,
    )

    raw_df = pd.read_csv(source_path)
    review_df = standardize_review_dataframe(raw_df)
    review_df, preprocessing_stats = prune_inactive_and_conflicting_entities(
        review_df=review_df,
        min_user_reviews=min_user_reviews,
        min_product_reviews=min_product_reviews,
    )
    user_df = build_user_frame(review_df)

    if balance_user_labels:
        balanced_users, balanced_stats = sample_users_balanced(
            user_df=user_df,
            target_total_users=balanced_user_count,
            seed=seed,
        )
        review_df = review_df[review_df["user_id"].isin(set(balanced_users["user_id"].astype(str)))].reset_index(drop=True)
        review_df["review_node_id"] = np.arange(len(review_df), dtype=np.int64)
        user_df = build_user_frame(review_df)
        preprocessing_stats["balanced_user_sampling"] = balanced_stats

    if smoke_max_users > 0:
        sampled_users = sample_users_for_smoke(user_df, max_users=smoke_max_users, seed=seed)
        review_df = review_df[review_df["user_id"].isin(sampled_users["user_id"])].reset_index(drop=True)
        review_df["review_node_id"] = np.arange(len(review_df), dtype=np.int64)
        user_df = build_user_frame(review_df)
        preprocessing_stats["smoke_user_sample_size"] = int(len(user_df))

    split_df = split_users(
        user_df=user_df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    user_df = user_df.merge(split_df, on=["user_id", "user_label"], how="left")
    review_df = review_df.merge(split_df[["user_id", "split"]], on="user_id", how="left")

    canonical_csv_path = output_dir / "reviews_canonical.csv"
    split_csv_path = output_dir / "user_splits.csv"
    metadata_path = output_dir / "dataset_metadata.json"
    legacy_tsv_dir = output_dir / "legacy_textcls_data"

    review_df.to_csv(canonical_csv_path, index=False)
    split_df.to_csv(split_csv_path, index=False)
    user_df.to_csv(output_dir / "users_canonical.csv", index=False)
    write_legacy_textcls_splits(review_df, split_df, legacy_tsv_dir)

    metadata = {
        "source_meta": source_meta,
        "review_count": int(len(review_df)),
        "user_count": int(user_df["user_id"].nunique()),
        "product_count": int(review_df["product_id"].nunique()),
        "fake_review_count": int(review_df["review_label"].sum()),
        "fake_user_count": int(user_df["user_label"].sum()),
        "split_sizes": split_df["split"].value_counts().to_dict(),
        "preprocessing": preprocessing_stats,
        "balance_user_labels": bool(balance_user_labels),
        "balanced_user_count": int(balanced_user_count),
        "blocked_label_columns": sorted(LABEL_LEAKAGE_COLUMNS),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    return PreparedGraphData(
        review_df=review_df,
        user_df=user_df,
        split_df=split_df,
        preprocessing_stats=preprocessing_stats,
        source_path=source_path,
        canonical_csv_path=canonical_csv_path,
        split_csv_path=split_csv_path,
        legacy_tsv_dir=legacy_tsv_dir,
        metadata_path=metadata_path,
    )
