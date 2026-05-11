from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from graph.graph_pipeline import build_self_feature_matrix

ASSET_PROJECT_ROOT = Path("/home/xyz/HuChao (2)/Bert-TextClassification")
BASE_PROTOCOL_DIR = ASSET_PROJECT_ROOT / "graph" / "outputs" / "yelpzip_balanced_current_graph_no_reweight_20260502_160620"
REFERENCE_D1_DIR = ASSET_PROJECT_ROOT / "graph" / "outputs" / "routeD_tns_guided_logic_egat_20260504_200855" / "D1_EGAT_Base_LogicAE_CB"
OUT_DIR = PROJECT_ROOT / "graph" / "outputs" / "D1_REPRO_PACK"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _sha256_array(arr: np.ndarray) -> str:
    return _sha256_bytes(np.ascontiguousarray(arr).tobytes())


def _sha256_df(df: pd.DataFrame) -> str:
    if df.empty:
        return "EMPTY"
    return _sha256_bytes(pd.util.hash_pandas_object(df, index=False).values.tobytes())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    edge_pack_dir = OUT_DIR / "edge_pack"
    edge_pack_dir.mkdir(parents=True, exist_ok=True)

    user_df = pd.read_csv(BASE_PROTOCOL_DIR / "user_scores_enriched.csv")
    review_df = pd.read_csv(BASE_PROTOCOL_DIR / "review_scores_enriched.csv")
    user_abnormal_vectors = np.load(BASE_PROTOCOL_DIR / "logic_vectors" / "user_abnormal_vectors.npy")
    self_features = build_self_feature_matrix(user_df.copy(), user_abnormal_vectors)

    user_order = user_df["user_id"].astype(str).tolist()
    label_vector = user_df["user_label"].astype(int).to_numpy(dtype=np.int64)

    split_indices = {}
    for split_name in ["train", "val", "test"]:
        split_indices[split_name] = user_df.index[user_df["split"].astype(str) == split_name].tolist()

    np.save(OUT_DIR / "final_self_feature_matrix.npy", self_features.astype(np.float32))
    np.save(OUT_DIR / "label_vector.npy", label_vector)
    (OUT_DIR / "user_order.json").write_text(json.dumps(user_order, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "split_indices.json").write_text(json.dumps(split_indices, ensure_ascii=False, indent=2), encoding="utf-8")

    edge_hashes = {}
    relation_counts = {}
    for relation in ["UPU", "UTU", "USU", "LogicAE_CB"]:
        src = REFERENCE_D1_DIR / "edges" / f"{relation}_edges.csv"
        dst = edge_pack_dir / f"{relation}_edges.csv"
        dst.write_bytes(src.read_bytes())
        edge_hashes[relation] = _sha256_file(dst)
        relation_counts[relation] = int(len(pd.read_csv(dst)))

    d1_config = json.loads((REFERENCE_D1_DIR / "config.json").read_text(encoding="utf-8"))
    d1_run_config = json.loads((REFERENCE_D1_DIR / "run_config.json").read_text(encoding="utf-8"))
    frozen = {
        "base_protocol_dir": str(BASE_PROTOCOL_DIR),
        "reference_d1_dir": str(REFERENCE_D1_DIR),
        "config": d1_config,
        "run_config": d1_run_config,
        "self_feature_shape": list(self_features.shape),
        "relation_counts": relation_counts,
    }
    (OUT_DIR / "config_frozen.json").write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8")

    hashes = {
        "final_self_feature_matrix_sha256": _sha256_array(self_features.astype(np.float32)),
        "label_vector_sha256": _sha256_array(label_vector),
        "user_scores_sha256": _sha256_df(user_df),
        "review_scores_sha256": _sha256_df(review_df),
        "user_abnormal_vectors_sha256": _sha256_array(user_abnormal_vectors),
        "edge_weight_hashes": edge_hashes,
    }
    (OUT_DIR / "edge_weight_hashes.json").write_text(json.dumps(edge_hashes, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "feature_hashes.json").write_text(json.dumps(hashes, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "git_commit.txt").write_text("ce6a9d6\n", encoding="utf-8")


if __name__ == "__main__":
    main()
