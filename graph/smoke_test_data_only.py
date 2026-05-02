from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .data_utils import ensure_dir, prepare_graph_data
from .graph_pipeline import (
    apply_graph_guided_evidence_reweighting,
    build_edge_frames,
    build_review_and_user_artifacts,
    build_self_feature_matrix,
    compute_edge_stats,
)
from .llm_utils import build_llm_features_and_masks
from .relation_model import run_relation_aggregation_experiments


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOKEN_RE = re.compile(r"\S+")


class LocalSimpleTokenizer:
    def __init__(self, max_length: int = 128) -> None:
        self.max_length = max_length

    def __call__(
        self,
        texts: str | list[str],
        padding: str = "max_length",
        truncation: bool = True,
        max_length: int | None = None,
        return_offsets_mapping: bool = False,
    ):
        max_length = max_length or self.max_length
        if isinstance(texts, str):
            return self._encode_one(texts, max_length=max_length, return_offsets_mapping=return_offsets_mapping)
        return [self._encode_one(text, max_length=max_length, return_offsets_mapping=return_offsets_mapping) for text in texts]

    def _encode_one(self, text: str, max_length: int, return_offsets_mapping: bool):
        input_ids = [101]
        attention_mask = [1]
        offsets = [(0, 0)]
        for match in TOKEN_RE.finditer(str(text or "")):
            input_ids.append(abs(hash(match.group(0).lower())) % 2048 + 100)
            attention_mask.append(1)
            offsets.append((match.start(), match.end()))
            if len(input_ids) >= max_length - 1:
                break
        input_ids.append(102)
        attention_mask.append(1)
        offsets.append((0, 0))
        while len(input_ids) < max_length:
            input_ids.append(0)
            attention_mask.append(0)
            offsets.append((0, 0))
        payload = {
            "input_ids": input_ids[:max_length],
            "attention_mask": attention_mask[:max_length],
        }
        if return_offsets_mapping:
            payload["offset_mapping"] = offsets[:max_length]
        return payload


def main() -> None:
    output_root = ensure_dir(PROJECT_ROOT / "graph" / "outputs" / "local_data_smoke")
    prepared = prepare_graph_data(
        graph_data_dir=PROJECT_ROOT / "graph data",
        output_dir=output_root / "prepared_data",
        data_path=None,
        seed=42,
        min_user_reviews=3,
        min_product_reviews=3,
        prefer_corrected_reviews=False,
        smoke_max_users=80,
    )
    tokenizer = LocalSimpleTokenizer(max_length=96)
    llm_feature_df, _ = build_llm_features_and_masks(
        review_df=prepared.review_df,
        tokenizer=tokenizer,
        llm_jsonl_path=None,
        output_dir=output_root / "llm_mask",
        max_seq_length=96,
        debug_use_empty_mask=True,
        mask_source="full_text",
    )

    ordered_reviews = prepared.review_df.sort_values("review_node_id").reset_index(drop=True)
    text_lengths = ordered_reviews["review_text"].apply(lambda text: len(str(text).split())).to_numpy(dtype=np.float32)
    probabilities = np.clip(
        0.25
        + 0.45 * (ordered_reviews["rating"].to_numpy(dtype=np.float32) <= 2).astype(np.float32)
        + 0.20 * (text_lengths <= 40).astype(np.float32),
        0.01,
        0.99,
    )
    review_vectors = np.stack(
        [
            text_lengths / np.maximum(text_lengths.max(), 1.0),
            ordered_reviews["rating"].to_numpy(dtype=np.float32) / 5.0,
            probabilities,
            (ordered_reviews["review_label"].to_numpy(dtype=np.float32) * 0.5) + 0.1,
        ],
        axis=1,
    ).astype(np.float32)
    text_vectors = np.stack(
        [
            text_lengths / np.maximum(text_lengths.max(), 1.0),
            ordered_reviews["rating"].to_numpy(dtype=np.float32) / 5.0,
            np.log1p(text_lengths) / np.log1p(max(float(text_lengths.max()), 1.0)),
            1.0 - probabilities,
        ],
        axis=1,
    ).astype(np.float32)

    review_output_df = ordered_reviews[
        ["review_node_id", "user_id", "product_id", "rating", "review_date", "review_label"]
    ].copy()
    review_output_df["p_fake_review"] = probabilities
    review_output_df["review_gate"] = 1.0

    review_scores_df, user_df, _, user_abnormal_vectors, user_text_vectors = build_review_and_user_artifacts(
        review_df=ordered_reviews,
        llm_feature_df=llm_feature_df,
        review_output_df=review_output_df,
        review_vectors=review_vectors,
        text_vectors=text_vectors,
        output_dir=output_root,
        top_m=3,
        time_bucket="week",
    )
    edge_frames = build_edge_frames(
        user_df=user_df,
        user_text_vectors=user_text_vectors,
        user_abnormal_vectors=user_abnormal_vectors,
        output_dir=output_root,
        top_k=10,
        review_features=review_scores_df,
    )
    review_scores_df, user_df, graph_reweighted_vectors, graph_support_edges = apply_graph_guided_evidence_reweighting(
        review_features=review_scores_df,
        user_df=user_df,
        review_vectors=review_vectors,
        edge_frames=edge_frames,
        output_dir=output_root,
        top_m=3,
        graph_top_k=10,
        alpha=0.7,
        neighbor_review_cap=10,
    )
    edge_frames["GraphSupport"] = graph_support_edges
    compute_edge_stats(edge_frames=edge_frames, user_df=user_df, output_dir=output_root)
    self_features = build_self_feature_matrix(user_df, graph_reweighted_vectors)
    run_relation_aggregation_experiments(
        user_df=user_df,
        self_features=self_features,
        edge_frames=edge_frames,
        output_dir=output_root / "metrics",
        review_encoder_name="data_only_mock",
        model_kind="logreg",
        seed=42,
    )
    print("Local data-only smoke test finished.")
    print(output_root)


if __name__ == "__main__":
    main()
