from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "d1_fresh_repro.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a no-old-output fresh D1 reproduction.")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--config_path", default=str(DEFAULT_CONFIG))
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _add_value_arg(cmd: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    cmd.extend([flag, str(value)])


def _git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "relative_path": str(path.relative_to(root)) if path.exists() and path.is_relative_to(root) else str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
        "sha256": _sha256(path),
    }


def _target_graph_row(output_root: Path) -> dict[str, Any]:
    csv_path = output_root / "metrics" / "model_results.csv"
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    if df.empty:
        return {}
    target = df.copy()
    if "edge_set" in target.columns:
        filtered = target[target["edge_set"] == "Base_LogicAE_CB"]
        if not filtered.empty:
            target = filtered
    if "model_name" in target.columns:
        filtered = target[target["model_name"] == "current_egat_edge_aware_gat"]
        if not filtered.empty:
            target = filtered
    return target.sort_values("auc", ascending=False).iloc[0].to_dict()


def _build_command(output_root: Path, cfg: dict[str, Any], smoke_test: bool) -> list[str]:
    base = cfg["base_protocol"]
    cmd = [sys.executable, "-m", "graph.run_final_experiment"]
    _add_value_arg(cmd, "--graph_data_dir", base["graph_data_dir"])
    _add_value_arg(cmd, "--output_dir", output_root)
    _add_value_arg(cmd, "--mask_source", base.get("mask_source", "full_text"))
    _add_value_arg(cmd, "--seed", base.get("seed", 42))
    _add_value_arg(cmd, "--train_ratio", base.get("train_ratio", 0.64))
    _add_value_arg(cmd, "--val_ratio", base.get("val_ratio", 0.16))
    _add_value_arg(cmd, "--test_ratio", base.get("test_ratio", 0.20))
    _add_value_arg(cmd, "--balanced_user_count", base.get("balanced_user_count", 6742))
    _add_value_arg(cmd, "--min_user_reviews", base.get("min_user_reviews", 3))
    _add_value_arg(cmd, "--min_product_reviews", base.get("min_product_reviews", 3))
    _add_value_arg(cmd, "--time_bucket", base.get("time_bucket", "week"))
    _add_value_arg(cmd, "--graph_mode", base.get("graph_mode", "current"))
    _add_value_arg(cmd, "--top_k", base.get("top_k", 20))
    _add_value_arg(cmd, "--top_m", base.get("top_m", 3))
    _add_value_arg(cmd, "--logic_threshold_mode", base.get("logic_threshold_mode", "quantile"))
    _add_value_arg(cmd, "--logic_threshold_quantile", base.get("logic_threshold_quantile", 0.60))
    _add_value_arg(cmd, "--logic_threshold_value", base.get("logic_threshold_value", 0.30))
    _add_value_arg(cmd, "--model_backbone", base.get("model_backbone", "current_egat"))
    _add_value_arg(cmd, "--relation_model", base.get("relation_model", "edge_aware_gat"))
    _add_value_arg(cmd, "--edge_set", base.get("edge_set", "Base_LogicAE_CB"))
    _add_value_arg(cmd, "--max_seq_length", base.get("max_seq_length", 256))
    _add_value_arg(cmd, "--review_encoder", base.get("review_encoder", "llm_masked_logic"))
    _add_value_arg(cmd, "--primary_model_name_or_path", base["primary_model_name_or_path"])
    _add_value_arg(cmd, "--legacy_roberta_model_dir", base.get("legacy_roberta_model_dir"))
    _add_value_arg(cmd, "--vector_dim", base.get("vector_dim", 256))
    _add_value_arg(cmd, "--batch_size", base.get("batch_size", 16))
    _add_value_arg(cmd, "--learning_rate", base.get("learning_rate", 2e-5))
    _add_value_arg(cmd, "--num_epochs", 1 if smoke_test else base.get("num_epochs", 3))
    _add_value_arg(cmd, "--patience", 1 if smoke_test else base.get("patience", 2))
    _add_value_arg(cmd, "--device", base.get("device", "auto"))

    if base.get("balance_user_labels", True):
        cmd.append("--balance_user_labels")
    if base.get("disable_graph_reweighting", True):
        cmd.append("--disable_graph_reweighting")
    if smoke_test:
        cmd.extend(["--smoke_test", "--smoke_max_users", "160"])
    return cmd


def _artifact_reuse(cfg: dict[str, Any], output_root: Path) -> list[dict[str, Any]]:
    base = cfg["base_protocol"]
    generated = [
        ("prepared_data", "canonical_data_and_split"),
        ("llm_mask", "full_text_mask_artifacts"),
        ("review_encoder/best_review_encoder.pt", "review_encoder_checkpoint"),
        ("review_encoder/review_encoder_metrics.json", "review_encoder_metrics"),
        ("logic_vectors/review_text_vectors.npy", "review_text_vectors"),
        ("logic_vectors/user_text_vectors.npy", "user_text_vectors"),
        ("logic_vectors/review_abnormal_vectors.npy", "review_abnormal_vectors"),
        ("logic_vectors/user_abnormal_vectors.npy", "user_abnormal_vectors"),
        ("logic_vectors/user_abnormal_vectors_initial.npy", "user_abnormal_vectors_initial"),
        ("logic_vectors/review_abnormal_scores.csv", "review_abnormal_scores"),
        ("review_scores_enriched.csv", "review_scores"),
        ("user_scores_enriched.csv", "user_scores"),
        ("edges/UPU_edges.csv", "behavior_edge"),
        ("edges/UTU_edges.csv", "behavior_edge"),
        ("edges/USU_edges.csv", "behavior_edge"),
        ("edges/TextSim_edges.csv", "text_vector_edge"),
        ("edges/CB_edges.csv", "text_vector_edge"),
        ("edges/LogicAE_CB_edges.csv", "abnormal_vector_edge"),
        ("metrics/edge_stats.csv", "edge_diagnostics"),
        ("metrics/model_results.csv", "graph_model_metrics"),
    ]
    rows: list[dict[str, Any]] = [
        {
            "path": base["graph_data_dir"],
            "class": "raw_input_data",
            "reuse_mode": "allowed_external_input",
            "reason": "Raw YelpZip graph data source; not a historical model/vector/edge artifact.",
        },
        {
            "path": base["primary_model_name_or_path"],
            "class": "pretrained_model",
            "reuse_mode": "allowed_external_input",
            "reason": "Same local pretrained RoBERTa path as historical D1; not a trained D1 checkpoint.",
        },
    ]
    rows.extend(
        {
            "path": str(output_root / rel),
            "class": cls,
            "reuse_mode": "regenerated",
            "reason": "Generated by this no-old-output fresh D1 reproduction route.",
        }
        for rel, cls in generated
    )
    rows.append(
        {
            "path": "/home/xyz/HuChao (2)/Bert-TextClassification/graph/outputs/*",
            "class": "historical_d1_outputs",
            "reuse_mode": "forbidden_for_training_inputs",
            "reason": "This route must not read old D1 checkpoints, vectors, scores, edges, predictions, or metrics as inputs.",
        }
    )
    return rows


def _write_audit(output_root: Path, cfg: dict[str, Any], command: list[str], started_at: str, finished_at: str) -> None:
    row = _target_graph_row(output_root)
    ref = cfg.get("reference_d1_metrics_for_audit_only", {})
    auc = row.get("auc")
    auc_min = cfg.get("d1_floor", {}).get("auc_min")
    passed = auc is not None and auc_min is not None and float(auc) >= float(auc_min)
    key_files = [
        "run_config.json",
        "run_summary.json",
        "prepared_data/reviews_canonical.csv",
        "prepared_data/users_canonical.csv",
        "prepared_data/user_splits.csv",
        "review_encoder/best_review_encoder.pt",
        "review_encoder/review_encoder_metrics.json",
        "logic_vectors/user_text_vectors.npy",
        "logic_vectors/user_abnormal_vectors.npy",
        "review_scores_enriched.csv",
        "user_scores_enriched.csv",
        "edges/UPU_edges.csv",
        "edges/UTU_edges.csv",
        "edges/USU_edges.csv",
        "edges/TextSim_edges.csv",
        "edges/CB_edges.csv",
        "edges/LogicAE_CB_edges.csv",
        "metrics/edge_stats.csv",
        "metrics/model_results.csv",
    ]
    audit = {
        "experiment_name": cfg.get("experiment_name"),
        "experiment_type": cfg.get("experiment_type"),
        "historical_d1_outputs_used_for_training": False,
        "started_at": started_at,
        "finished_at": finished_at,
        "project_root": str(PROJECT_ROOT),
        "git_commit": _git_value(["rev-parse", "HEAD"]),
        "git_status_short": _git_value(["status", "--short"]),
        "command": command,
        "target_graph_row": row,
        "reference_d1_metrics_for_audit_only": ref,
        "d1_floor": cfg.get("d1_floor"),
        "passed_d1_floor": bool(passed),
        "delta_vs_reference": {
            key: (float(row[key]) - float(ref[key])) if key in row and key in ref else None
            for key in ["auc", "ap", "f1", "recall", "precision"]
        },
        "key_files": [_file_record(output_root / rel, output_root) for rel in key_files],
    }
    base_protocol = cfg.get("base_protocol", {})
    _save_json(
        output_root / "artifact_reuse.json",
        {
            "experiment_name": cfg.get("experiment_name"),
            "experiment_type": cfg.get("experiment_type"),
            "historical_d1_outputs_used_for_training": False,
            "review_checkpoint_policy": base_protocol.get("review_checkpoint_policy"),
            "allowed_external_inputs": [
                base_protocol.get("graph_data_dir"),
                base_protocol.get("primary_model_name_or_path"),
            ],
            "items": _artifact_reuse(cfg, output_root),
        },
    )
    _save_json(output_root / "d1_fresh_repro_audit.json", audit)


def main() -> None:
    args = parse_args()
    cfg = _load_json(Path(args.config_path))
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.resume and (output_root / "d1_fresh_repro_audit.json").exists():
        print(f"[resume] existing audit found: {output_root / 'd1_fresh_repro_audit.json'}", flush=True)
        return

    _save_json(output_root / "routeD1_fresh_repro_config.json", cfg)
    started_at = datetime.now().isoformat(timespec="seconds")
    cmd = _build_command(output_root, cfg, args.smoke_test)
    print("experiment_type=fresh_d1_train", flush=True)
    print("historical_d1_outputs_used_for_training=false", flush=True)
    print("command=" + " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    finished_at = datetime.now().isoformat(timespec="seconds")
    _write_audit(output_root, cfg, [str(part) for part in cmd], started_at, finished_at)
    row = _target_graph_row(output_root)
    print(
        "target_result="
        + json.dumps(
            {key: row.get(key) for key in ["model_name", "edge_set", "auc", "ap", "f1", "recall", "precision"]},
            ensure_ascii=False,
        ),
        flush=True,
    )
    print(f"audit_path={output_root / 'd1_fresh_repro_audit.json'}", flush=True)


if __name__ == "__main__":
    main()
