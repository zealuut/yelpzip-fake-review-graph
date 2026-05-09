from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def export_review_feature_frame(
    review_output_df: pd.DataFrame,
    review_vectors: np.ndarray,
    text_vectors: np.ndarray,
    output_dir: str | Path,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = review_output_df.copy()
    frame = frame[[c for c in frame.columns if c != "aux_logit"]].copy()
    frame.to_csv(output_dir / "routeL_review_features.csv", index=False)
    np.save(output_dir / "routeL_review_vectors.npy", review_vectors.astype(np.float32))
    np.save(output_dir / "routeL_text_vectors.npy", text_vectors.astype(np.float32))
    return frame


def assert_aux_not_exported(frame: pd.DataFrame) -> bool:
    return "aux_logit" not in frame.columns

