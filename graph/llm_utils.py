from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - local fallback
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []


PATTERN_TYPES = [
    "generic_promotion",
    "generic_attack",
    "lack_of_detail",
    "template_like",
    "exaggeration",
    "overly_absolute",
    "sentiment_rating_mismatch",
    "inconsistent_claim",
    "none",
]


def numeric_feature_columns() -> list[str]:
    base_columns = [
        "specificity_score",
        "template_score",
        "exaggeration_score",
        "experience_detail_score",
        "max_pattern_confidence",
        "num_abnormal_patterns",
    ]
    pattern_columns = [f"pattern_type__{pattern_name}" for pattern_name in PATTERN_TYPES]
    return base_columns + pattern_columns


def load_llm_cache(jsonl_path: str | Path) -> dict[int, dict[str, Any]]:
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"LLM cache JSONL not found: {path}")

    cache: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw_line = line.strip()
            if not raw_line:
                continue
            payload = json.loads(raw_line)
            if "review_node_id" not in payload:
                raise ValueError(
                    f"LLM cache line {line_number} is missing review_node_id: {path}"
                )
            review_node_id = int(payload["review_node_id"])
            cache[review_node_id] = normalize_llm_payload(payload)
    return cache


def normalize_llm_payload(payload: dict[str, Any]) -> dict[str, Any]:
    abnormal_patterns = payload.get("abnormal_patterns", []) or []
    normalized_patterns = []
    for pattern in abnormal_patterns:
        pattern_type = str(pattern.get("pattern_type", "none")).strip().lower() or "none"
        if pattern_type not in PATTERN_TYPES:
            pattern_type = "none"
        confidence = float(pattern.get("confidence", 0.0) or 0.0)
        normalized_patterns.append(
            {
                "pattern_type": pattern_type,
                "description": str(pattern.get("description", "") or ""),
                "evidence_span": str(pattern.get("evidence_span", "") or ""),
                "confidence": float(np.clip(confidence, 0.0, 1.0)),
            }
        )

    return {
        "review_node_id": int(payload["review_node_id"]),
        "abnormal_patterns": normalized_patterns,
        "sentiment": str(payload.get("sentiment", "neutral") or "neutral"),
        "specificity_score": float(payload.get("specificity_score", 0.0) or 0.0),
        "template_score": float(payload.get("template_score", 0.0) or 0.0),
        "exaggeration_score": float(payload.get("exaggeration_score", 0.0) or 0.0),
        "experience_detail_score": float(payload.get("experience_detail_score", 0.0) or 0.0),
        "claim_summary": str(payload.get("claim_summary", "") or ""),
    }


def empty_llm_payload(review_node_id: int) -> dict[str, Any]:
    return {
        "review_node_id": int(review_node_id),
        "abnormal_patterns": [],
        "sentiment": "neutral",
        "specificity_score": 0.0,
        "template_score": 0.0,
        "exaggeration_score": 0.0,
        "experience_detail_score": 0.0,
        "claim_summary": "",
    }


def _normalize_whitespace_with_mapping(text: str) -> tuple[str, list[int]]:
    normalized_chars: list[str] = []
    original_indices: list[int] = []
    last_was_space = False
    for index, char in enumerate(text):
        if char.isspace():
            if not last_was_space:
                normalized_chars.append(" ")
                original_indices.append(index)
            last_was_space = True
        else:
            normalized_chars.append(char.lower())
            original_indices.append(index)
            last_was_space = False
    return "".join(normalized_chars), original_indices


def _find_candidate_spans(review_text: str, evidence_span: str) -> list[tuple[int, int]]:
    review_text = review_text or ""
    evidence_span = (evidence_span or "").strip()
    if not review_text or not evidence_span:
        return []

    lowered_text = review_text.lower()
    lowered_span = evidence_span.lower()
    matches: list[tuple[int, int]] = []

    for match in re.finditer(re.escape(lowered_span), lowered_text):
        matches.append((match.start(), match.end()))
    if matches:
        return matches

    normalized_text, text_mapping = _normalize_whitespace_with_mapping(review_text)
    normalized_span, _ = _normalize_whitespace_with_mapping(evidence_span)
    if not normalized_span:
        return []

    start_index = 0
    while True:
        hit = normalized_text.find(normalized_span, start_index)
        if hit < 0:
            break
        end_hit = hit + len(normalized_span) - 1
        if hit < len(text_mapping) and end_hit < len(text_mapping):
            original_start = text_mapping[hit]
            original_end = text_mapping[end_hit] + 1
            matches.append((original_start, original_end))
        start_index = hit + 1
    return matches


def build_soft_token_mask(
    review_text: str,
    abnormal_patterns: list[dict[str, Any]],
    offsets: list[tuple[int, int]],
) -> tuple[np.ndarray, dict[str, Any]]:
    mask = np.zeros(len(offsets), dtype=np.float32)
    matched_spans = 0
    unmatched_spans = 0
    matched_tokens = 0
    unmatched_details: list[str] = []

    for pattern in abnormal_patterns:
        evidence_span = str(pattern.get("evidence_span", "") or "").strip()
        confidence = float(np.clip(pattern.get("confidence", 0.0) or 0.0, 0.0, 1.0))
        if not evidence_span or confidence <= 0.0:
            continue

        candidates = _find_candidate_spans(review_text, evidence_span)
        if not candidates:
            unmatched_spans += 1
            unmatched_details.append(evidence_span)
            continue

        matched_any_token = False
        for char_start, char_end in candidates:
            for token_index, (token_start, token_end) in enumerate(offsets):
                if token_start == token_end == 0:
                    continue
                if max(token_start, char_start) < min(token_end, char_end):
                    mask[token_index] = max(mask[token_index], confidence)
                    matched_any_token = True
        if matched_any_token:
            matched_spans += 1
        else:
            unmatched_spans += 1
            unmatched_details.append(evidence_span)

    matched_tokens = int((mask > 0).sum())
    stats = {
        "matched_spans": matched_spans,
        "unmatched_spans": unmatched_spans,
        "matched_mask_token_count": matched_tokens,
        "unmatched_evidence_spans": " || ".join(unmatched_details),
    }
    return mask, stats


def _payload_to_numeric_features(payload: dict[str, Any]) -> dict[str, float]:
    abnormal_patterns = payload.get("abnormal_patterns", []) or []
    active_patterns = [pattern for pattern in abnormal_patterns if pattern.get("pattern_type") != "none"]
    max_pattern_confidence = max([float(pattern.get("confidence", 0.0) or 0.0) for pattern in abnormal_patterns] or [0.0])

    row = {
        "specificity_score": float(payload.get("specificity_score", 0.0) or 0.0),
        "template_score": float(payload.get("template_score", 0.0) or 0.0),
        "exaggeration_score": float(payload.get("exaggeration_score", 0.0) or 0.0),
        "experience_detail_score": float(payload.get("experience_detail_score", 0.0) or 0.0),
        "max_pattern_confidence": float(np.clip(max_pattern_confidence, 0.0, 1.0)),
        "num_abnormal_patterns": float(len(active_patterns)),
    }
    for pattern_name in PATTERN_TYPES:
        row[f"pattern_type__{pattern_name}"] = 0.0
    for pattern in active_patterns:
        row[f"pattern_type__{pattern['pattern_type']}"] = 1.0
    return row


def build_llm_features_and_masks(
    review_df: pd.DataFrame,
    tokenizer: Any,
    llm_jsonl_path: str | Path | None,
    output_dir: str | Path,
    max_seq_length: int,
    debug_use_empty_mask: bool = False,
    mask_source: str = "llm",
) -> tuple[pd.DataFrame, np.ndarray]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_source = str(mask_source or "llm").lower()
    if mask_source not in {"llm", "full_text", "empty"}:
        raise ValueError(f"Unsupported mask_source: {mask_source}. Use llm, full_text, or empty.")

    if debug_use_empty_mask and mask_source == "llm" and llm_jsonl_path is None:
        mask_source = "empty"

    if mask_source != "llm":
        llm_cache: dict[int, dict[str, Any]] = {}
    elif llm_jsonl_path is None:
        if not debug_use_empty_mask:
            raise FileNotFoundError(
                "No LLM cache path was provided. Formal runs must pass --llm_jsonl_path. "
                "Use --mask_source full_text for the no-LLM fake-extractor baseline, "
                "or --debug_use_empty_mask only for smoke tests."
            )
        llm_cache: dict[int, dict[str, Any]] = {}
    else:
        cache_path = Path(llm_jsonl_path)
        if not cache_path.exists():
            if not debug_use_empty_mask:
                raise FileNotFoundError(
                    f"LLM cache JSONL not found: {cache_path}. "
                    "Use --debug_use_empty_mask only for smoke tests."
                )
            llm_cache = {}
        else:
            llm_cache = load_llm_cache(cache_path)
            if not debug_use_empty_mask:
                expected_ids = set(review_df["review_node_id"].astype(int).tolist())
                missing_ids = sorted(expected_ids - set(llm_cache.keys()))
                if missing_ids:
                    missing_path = output_dir / "missing_llm_cache_ids.txt"
                    missing_path.write_text(
                        "\n".join(str(review_node_id) for review_node_id in missing_ids),
                        encoding="utf-8",
                    )
                    raise ValueError(
                        f"LLM cache is incomplete: missing {len(missing_ids)} of {len(expected_ids)} reviews. "
                        f"Sample missing ids: {missing_ids[:10]}. "
                        f"Full missing id list was written to {missing_path}."
                    )

    feature_rows: list[dict[str, Any]] = []
    masks: list[np.ndarray] = []
    total_spans = 0
    total_matched_spans = 0
    total_unmatched_spans = 0

    iterator = tqdm(review_df.itertuples(index=False), total=len(review_df), desc="Building LLM masks")
    for row in iterator:
        review_node_id = int(row.review_node_id)
        payload = llm_cache.get(review_node_id, empty_llm_payload(review_node_id))
        encoding = tokenizer(
            row.review_text,
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
            return_offsets_mapping=True,
        )
        offsets = list(encoding["offset_mapping"])
        abnormal_patterns = payload.get("abnormal_patterns", []) or []
        total_spans += len([pattern for pattern in abnormal_patterns if pattern.get("evidence_span")])

        if mask_source == "full_text":
            mask = np.asarray(
                [1.0 if int(token_start) != int(token_end) else 0.0 for token_start, token_end in offsets],
                dtype=np.float32,
            )
            match_stats = {
                "matched_spans": 0,
                "unmatched_spans": 0,
                "matched_mask_token_count": int((mask > 0).sum()),
                "unmatched_evidence_spans": "",
            }
        elif mask_source == "empty":
            mask = np.zeros(len(offsets), dtype=np.float32)
            match_stats = {
                "matched_spans": 0,
                "unmatched_spans": 0,
                "matched_mask_token_count": 0,
                "unmatched_evidence_spans": "",
            }
        else:
            if debug_use_empty_mask and review_node_id not in llm_cache:
                abnormal_patterns = []

            mask, match_stats = build_soft_token_mask(
                review_text=row.review_text,
                abnormal_patterns=abnormal_patterns,
                offsets=offsets,
            )
        total_matched_spans += int(match_stats["matched_spans"])
        total_unmatched_spans += int(match_stats["unmatched_spans"])
        numeric = _payload_to_numeric_features(payload)

        feature_row = {
            "review_node_id": review_node_id,
            "sentiment": payload.get("sentiment", "neutral"),
            "claim_summary": payload.get("claim_summary", ""),
            "matched_mask_token_count": int(match_stats["matched_mask_token_count"]),
            "matched_spans": int(match_stats["matched_spans"]),
            "unmatched_spans": int(match_stats["unmatched_spans"]),
            "unmatched_evidence_spans": match_stats["unmatched_evidence_spans"],
            "debug_use_empty_mask": bool(debug_use_empty_mask and review_node_id not in llm_cache),
            "mask_source": mask_source,
            "llm_feature_available": float(mask_source == "llm" and review_node_id in llm_cache),
        }
        feature_row.update(numeric)
        masks.append(mask)
        feature_rows.append(feature_row)

    mask_array = np.asarray(masks, dtype=np.float32)
    feature_df = pd.DataFrame(feature_rows).sort_values("review_node_id").reset_index(drop=True)
    feature_df.to_csv(output_dir / "llm_review_features.csv", index=False)
    np.save(output_dir / "abnormal_token_masks.npy", mask_array)

    matched_review_count = int((feature_df["matched_mask_token_count"] > 0).sum())
    alignment_stats = pd.DataFrame(
        [
            {
                "num_reviews": int(len(review_df)),
                "mask_source": mask_source,
                "num_reviews_with_mask": matched_review_count,
                "num_reviews_without_mask": int(len(review_df) - matched_review_count),
                "num_total_spans": int(total_spans),
                "num_matched_spans": int(total_matched_spans),
                "num_unmatched_spans": int(total_unmatched_spans),
                "matched_span_ratio": float(total_matched_spans / max(total_spans, 1)),
            }
        ]
    )
    alignment_stats.to_csv(output_dir / "mask_alignment_stats.csv", index=False)
    feature_df[["review_node_id", "unmatched_evidence_spans"]].to_csv(
        output_dir / "unmatched_spans.csv",
        index=False,
    )

    return feature_df, mask_array
