"""RouteV user-level vector quality metrics.

This CLI is intentionally small and route-local. It reads review-level abnormal
vectors plus the matching review output CSV, aggregates to user-level vectors,
and reports the same strict user-label proxy used for RouteV checkpoint
selection.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from graph.routes.routeV_vector_quality.src.user_level_proxy import (  # noqa: E402
    compute_user_vector_proxy,
    compute_user_vector_proxy_from_split,
    compute_user_vector_proxy_train_eval,
)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute strict RouteV user-level vector metrics")
    parser.add_argument("--review_vectors", required=True, help="Path to review abnormal vectors .npy")
    parser.add_argument("--review_output", required=True, help="Path to review_output.csv with user_label")
    parser.add_argument("--output_json", default=None, help="Optional metrics JSON path")
    parser.add_argument("--top_m", type=int, default=3)
    parser.add_argument("--score_column", default="p_fake_review")
    parser.add_argument("--label_column", default="user_label")
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--eval_split", default="val")
    parser.add_argument(
        "--mode",
        choices=["train_eval", "split_norm", "all_norm"],
        default="train_eval",
        help="train_eval matches RouteV checkpoint selection; split/all modes are diagnostics.",
    )
    parser.add_argument("--split", default="val", help="Split for split_norm diagnostic")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vector_path = Path(args.review_vectors)
    review_output_path = Path(args.review_output)
    review_vectors = np.load(vector_path)
    review_df = pd.read_csv(review_output_path)

    if args.label_column not in review_df.columns:
        raise ValueError(
            f"{review_output_path} is missing {args.label_column}; RouteV metrics require explicit user_label."
        )

    if args.mode == "train_eval":
        metrics = compute_user_vector_proxy_train_eval(
            review_vectors=review_vectors,
            review_df=review_df,
            train_split=args.train_split,
            eval_split=args.eval_split,
            top_m=args.top_m,
            score_column=args.score_column,
            label_column=args.label_column,
        )
    elif args.mode == "split_norm":
        metrics = compute_user_vector_proxy_from_split(
            review_vectors=review_vectors,
            review_df=review_df,
            split=args.split,
            top_m=args.top_m,
            score_column=args.score_column,
            label_column=args.label_column,
        )
    else:
        metrics = compute_user_vector_proxy(
            review_vectors=review_vectors,
            review_df=review_df,
            top_m=args.top_m,
            score_column=args.score_column,
            label_column=args.label_column,
        )

    payload = {
        "status": "ok",
        "review_vectors": str(vector_path),
        "review_output": str(review_output_path),
        "mode": args.mode,
        "strict_label_policy": {
            "label_column": args.label_column,
            "label_source": "prepared.user_df.user_label attached to review_output.csv",
            "forbidden_label_source": "review_label.max_per_user",
        },
        "metrics": metrics,
    }
    text = json.dumps(_json_ready(payload), indent=2, ensure_ascii=False)
    print(text)
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
