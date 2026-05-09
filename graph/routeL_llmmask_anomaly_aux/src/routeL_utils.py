from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd
import yaml


@dataclass(frozen=True)
class RouteLD1Bundle:
    project_root: Path
    d1_output_dir: Path
    base_dir: Path
    run_config: dict[str, Any]
    run_summary: dict[str, Any]
    review_df: pd.DataFrame
    llm_feature_df: pd.DataFrame
    abnormal_masks: np.ndarray


def project_root_from_here() -> Path:
    return Path(__file__).resolve().parents[3]


def d1_base_dir(project_root: Path | None = None) -> Path:
    project_root = project_root or project_root_from_here()
    return project_root / "graph/outputs/yelpzip_balanced_current_graph_no_reweight_20260502_160620"


def d1_output_dir(project_root: Path | None = None) -> Path:
    project_root = project_root or project_root_from_here()
    return project_root / "graph/outputs/routeD_tns_guided_logic_egat_20260504_200855/D1_EGAT_Base_LogicAE_CB"


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def load_d1_bundle(project_root: Path | None = None) -> RouteLD1Bundle:
    project_root = project_root or project_root_from_here()
    base = d1_base_dir(project_root)
    out = d1_output_dir(project_root)
    run_config = json.loads((base / "run_config.json").read_text(encoding="utf-8"))
    run_summary = json.loads((out / "run_summary.json").read_text(encoding="utf-8"))
    review_df = pd.read_csv(base / "prepared_data/reviews_canonical.csv")
    llm_feature_df = pd.read_csv(base / "llm_mask/llm_review_features.csv")
    abnormal_masks = np.load(base / "llm_mask/abnormal_token_masks.npy")
    return RouteLD1Bundle(
        project_root=project_root,
        d1_output_dir=out,
        base_dir=base,
        run_config=run_config,
        run_summary=run_summary,
        review_df=review_df,
        llm_feature_df=llm_feature_df,
        abnormal_masks=abnormal_masks,
    )


def batch_head(df: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    return df.sort_values("review_node_id").head(int(n)).reset_index(drop=True)


def json_dump(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

