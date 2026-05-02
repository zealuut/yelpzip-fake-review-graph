from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .graph_pipeline import build_edge_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild senior edges from existing artifacts and report edge counts.")
    parser.add_argument("--artifacts_dir", required=True, help="Existing run output directory containing logic_vectors/")
    parser.add_argument("--output_dir", required=True, help="Where rebuilt edges and summary should be written")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--graph_mode", choices=["current", "senior", "senior_enhanced"], default="senior")
    parser.add_argument("--senior_usu_ratio", type=float, default=0.10)
    parser.add_argument("--logic_threshold_mode", choices=["quantile", "fixed", "none"], default="quantile")
    parser.add_argument("--logic_threshold_quantile", type=float, default=0.60)
    parser.add_argument("--logic_threshold_value", type=float, default=0.30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    logic_dir = artifacts_dir / "logic_vectors"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    user_df = pd.read_csv(logic_dir / "user_summary.csv")
    user_text_vectors = np.load(logic_dir / "user_text_vectors.npy")
    user_abnormal_vectors = np.load(logic_dir / "user_abnormal_vectors.npy")
    review_scores_path = artifacts_dir / "review_scores_enriched.csv"
    review_scores_df = pd.read_csv(review_scores_path) if review_scores_path.exists() else None

    edge_frames = build_edge_frames(
        user_df=user_df,
        user_text_vectors=user_text_vectors,
        user_abnormal_vectors=user_abnormal_vectors,
        output_dir=output_dir,
        top_k=args.top_k,
        review_features=review_scores_df,
        logic_threshold_mode=args.logic_threshold_mode,
        logic_threshold_quantile=args.logic_threshold_quantile,
        logic_threshold_value=args.logic_threshold_value,
        graph_mode=args.graph_mode,
        senior_usu_ratio=args.senior_usu_ratio,
    )

    summary = {}
    for edge_name, frame in edge_frames.items():
        summary[edge_name] = {
            "directed_rows": int(len(frame)),
            "estimated_undirected_pairs": int(len(frame) // 2),
        }
    (output_dir / "edge_count_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
