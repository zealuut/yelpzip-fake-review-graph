from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from graph.graph_pipeline import (
    DEFAULT_LOGIC_THRESHOLD_MODE,
    DEFAULT_LOGIC_THRESHOLD_QUANTILE,
    DEFAULT_LOGIC_THRESHOLD_VALUE,
    _build_cb_like_edges,
    _build_knn_edges,
    _build_tns_heavy_feature_cache,
    _resolve_logic_threshold,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_DIR = PROJECT_ROOT / "graph" / "outputs" / "yelpzip_balanced_current_graph_no_reweight_20260502_160620"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute reusable TNS-heavy session and pair features.")
    parser.add_argument("--base_dir", default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--phi_days", type=int, default=5)
    parser.add_argument("--top_k", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir)
    review_scores_df = pd.read_csv(base_dir / "review_scores_enriched.csv")
    user_scores_path = base_dir / "user_scores_enriched.csv"
    logic_dir = base_dir / "logic_vectors"
    user_df = pd.read_csv(user_scores_path) if user_scores_path.exists() else pd.read_csv(logic_dir / "user_summary.csv")
    user_abnormal_vectors = np.load(logic_dir / "user_abnormal_vectors.npy")

    user_ids = user_df["user_id"].astype(str).tolist()
    logic_knn_edges = _build_knn_edges(
        user_ids=user_ids,
        vectors=user_abnormal_vectors,
        edge_type="LogicKNN",
        top_k=max(int(args.top_k) * 5, int(args.top_k) + 10),
    )
    tau_logic = _resolve_logic_threshold(
        candidate_edges=logic_knn_edges,
        mode=DEFAULT_LOGIC_THRESHOLD_MODE,
        quantile=DEFAULT_LOGIC_THRESHOLD_QUANTILE,
        threshold_value=DEFAULT_LOGIC_THRESHOLD_VALUE,
    )
    logic_edges = _build_cb_like_edges(
        user_df=user_df,
        candidate_edges=logic_knn_edges,
        vector_score_name="S_logic",
        edge_type="LogicAE_CB",
        top_k=int(args.top_k),
        min_vector_score=tau_logic,
        threshold_column="tau_logic",
    )
    cache = _build_tns_heavy_feature_cache(
        logic_edges=logic_edges,
        review_features=review_scores_df,
        user_df=user_df,
        user_abnormal_vectors=user_abnormal_vectors,
        phi_days=int(args.phi_days),
    )
    summary = {
        "base_dir": str(base_dir),
        "phi_days": int(args.phi_days),
        "logic_edge_count": int(len(logic_edges)),
        "pair_feature_count": int(len(cache["pair_df"])),
        "session_count": int(len(cache["session_df"])),
        "cache_dir": str(cache["cache_dir"]),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
