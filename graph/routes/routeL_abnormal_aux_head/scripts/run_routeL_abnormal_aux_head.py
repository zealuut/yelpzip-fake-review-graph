from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "abnormal_aux_phase1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--config_path", default=str(DEFAULT_CONFIG))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--skip_basecheck", action="store_true")
    parser.add_argument("--basecheck_only", action="store_true")
    return parser.parse_args()


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _as_cli_value(value: Any) -> str:
    if isinstance(value, bool):
        raise TypeError("Boolean values must be handled as flags.")
    return str(value)


def _add_value_arg(cmd: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    cmd.extend([flag, _as_cli_value(value)])


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_review_metrics(exp_dir: Path) -> dict[str, Any]:
    path = exp_dir / "review_encoder" / "review_encoder_metrics.json"
    if not path.exists():
        return {}
    try:
        return _load_json(path)
    except Exception as exc:
        return {"error": f"could not read {path}: {exc}"}


def _load_graph_row(exp_dir: Path) -> dict[str, Any]:
    csv_path = exp_dir / "metrics" / "model_results.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if not df.empty:
            target = df.loc[df.get("edge_set", "") == "Base_LogicAE_CB"]
            return (target.iloc[-1] if not target.empty else df.iloc[-1]).to_dict()
    summary_path = exp_dir / "run_summary.json"
    if summary_path.exists():
        try:
            summary = _load_json(summary_path)
            row = summary.get("best_graph_model")
            if isinstance(row, dict):
                return row
        except Exception:
            pass
    return {}


def _summary_row(exp_dir: Path, variant: dict[str, Any], resolved: dict[str, Any]) -> dict[str, Any]:
    graph_row = _load_graph_row(exp_dir)
    review_metrics = _load_review_metrics(exp_dir)
    val_metrics = review_metrics.get("val_metrics", {}) if isinstance(review_metrics, dict) else {}
    val_aux_metrics = review_metrics.get("val_aux_metrics", {}) if isinstance(review_metrics, dict) else {}
    train_aux_metrics = review_metrics.get("train_aux_metrics", {}) if isinstance(review_metrics, dict) else {}
    return {
        "experiment_name": variant["experiment_name"],
        "abnormal_aux_lambda": resolved["abnormal_aux_lambda"],
        "abnormal_aux_position": resolved["abnormal_aux_position"],
        "disable_cross_attention": resolved["disable_cross_attention"],
        "disable_logic_bilstm": resolved["disable_logic_bilstm"],
        "logic_pooling": resolved["logic_pooling"],
        "gate_mode": resolved["gate_mode"],
        "review_val_auc": val_metrics.get("auc"),
        "review_val_ap": val_metrics.get("ap"),
        "review_val_f1": val_metrics.get("f1"),
        "aux_train_auc": train_aux_metrics.get("auc"),
        "aux_val_auc": val_aux_metrics.get("auc"),
        "aux_val_ap": val_aux_metrics.get("ap"),
        "graph_val_auc": graph_row.get("val_auc"),
        "graph_val_ap": graph_row.get("val_ap"),
        "graph_AUC": graph_row.get("auc"),
        "graph_AP": graph_row.get("ap"),
        "graph_F1": graph_row.get("f1"),
        "graph_Recall": graph_row.get("recall"),
        "graph_Precision": graph_row.get("precision"),
        "output_dir": str(exp_dir),
        "notes": variant.get("notes", ""),
    }


def _build_command(*, exp_dir: Path, cfg: dict[str, Any], variant: dict[str, Any], resolved: dict[str, Any], smoke_test: bool) -> list[str]:
    base = cfg["base_protocol"]
    cmd = [sys.executable, "-m", "graph.run_final_experiment"]
    _add_value_arg(cmd, "--graph_data_dir", base["graph_data_dir"])
    _add_value_arg(cmd, "--output_dir", str(exp_dir))
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

    _add_value_arg(cmd, "--abnormal_aux_lambda", resolved["abnormal_aux_lambda"])
    _add_value_arg(cmd, "--abnormal_aux_position", resolved["abnormal_aux_position"])
    _add_value_arg(cmd, "--logic_pooling", resolved["logic_pooling"])
    _add_value_arg(cmd, "--gate_mode", resolved["gate_mode"])
    if resolved["disable_cross_attention"]:
        cmd.append("--disable_cross_attention")
    if resolved["disable_logic_bilstm"]:
        cmd.append("--disable_logic_bilstm")
    return cmd


def _resolve_variant(defaults: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(defaults)
    for key in [
        "abnormal_aux_lambda",
        "abnormal_aux_position",
        "disable_cross_attention",
        "disable_logic_bilstm",
        "logic_pooling",
        "gate_mode",
    ]:
        if key in variant:
            resolved[key] = variant[key]
    return resolved


def _run_variant(
    *,
    args: argparse.Namespace,
    cfg: dict[str, Any],
    output_root: Path,
    variant: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    experiment_name = str(variant["experiment_name"])
    exp_dir = output_root / experiment_name
    resolved = _resolve_variant(cfg.get("defaults", {}), variant)
    if args.resume and (exp_dir / "run_summary.json").exists():
        print(f"[resume] skip completed {experiment_name}", flush=True)
        row = _summary_row(exp_dir, variant, resolved)
        rows.append(row)
        return row

    print(f"starting {experiment_name} resolved={resolved}", flush=True)
    _save_json(
        exp_dir / "routeL_variant_config.json",
        {
            "experiment_name": experiment_name,
            "resolved": resolved,
            "variant": variant,
            "base_protocol": cfg["base_protocol"],
            "strict_base_policy": cfg.get("strict_base_policy"),
        },
    )
    cmd = _build_command(
        exp_dir=exp_dir,
        cfg=cfg,
        variant=variant,
        resolved=resolved,
        smoke_test=args.smoke_test,
    )
    print("command=" + " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    row = _summary_row(exp_dir, variant, resolved)
    rows.append(row)
    print(
        f"completed {experiment_name} graph_auc={row.get('graph_AUC')} aux_val_auc={row.get('aux_val_auc')}",
        flush=True,
    )
    pd.DataFrame(rows).to_csv(output_root / "routeL_abnormal_aux_head_summary.csv", index=False)
    return row


def _basecheck_variant(cfg: dict[str, Any]) -> dict[str, Any]:
    basecheck = cfg.get("basecheck", {})
    return {
        "experiment_name": basecheck.get("experiment_name", "D1_NO_AUX_fresh_control"),
        "abnormal_aux_lambda": 0.0,
        "notes": basecheck.get(
            "notes",
            "Fresh D1 no-aux control. This must reproduce D1 before aux-head ablations are trusted.",
        ),
    }


def main() -> None:
    args = parse_args()
    cfg = _load_json(Path(args.config_path))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    variants = cfg["variants"][:1] if args.smoke_test else cfg["variants"]

    rows: list[dict[str, Any]] = []
    basecheck = cfg.get("basecheck", {})
    if basecheck.get("enabled", True) and not args.skip_basecheck:
        row = _run_variant(
            args=args,
            cfg=cfg,
            output_root=output_root,
            variant=_basecheck_variant(cfg),
            rows=rows,
        )
        graph_auc = row.get("graph_AUC")
        min_graph_auc = basecheck.get("min_graph_auc")
        failed = (
            not args.smoke_test
            and min_graph_auc is not None
            and (graph_auc is None or float(graph_auc) < float(min_graph_auc))
        )
        if failed:
            failure = {
                "reason": "fresh D1 no-aux basecheck did not reproduce the required D1 floor",
                "graph_AUC": graph_auc,
                "min_graph_auc": min_graph_auc,
                "reference_graph_auc": basecheck.get("reference_graph_auc"),
                "strict_base_policy": cfg.get("strict_base_policy"),
                "basecheck_output_dir": row.get("output_dir"),
            }
            _save_json(output_root / "BASECHECK_FAILED.json", failure)
            pd.DataFrame(rows).to_csv(output_root / "routeL_abnormal_aux_head_summary.csv", index=False)
            raise SystemExit(json.dumps(failure, ensure_ascii=False))
        if args.basecheck_only:
            pd.DataFrame(rows).to_csv(output_root / "routeL_abnormal_aux_head_summary.csv", index=False)
            return

    for variant in variants:
        _run_variant(args=args, cfg=cfg, output_root=output_root, variant=variant, rows=rows)

    _save_json(
        output_root / "run_summary.json",
        {
            "experiment_name": "routeL_abnormal_aux_head_phase1",
            "config_path": str(args.config_path),
            "summary_csv": str(output_root / "routeL_abnormal_aux_head_summary.csv"),
            "rows": rows,
        },
    )
    pd.DataFrame(rows).to_csv(output_root / "routeL_abnormal_aux_head_summary.csv", index=False)
    print(f"summary_path={output_root / 'routeL_abnormal_aux_head_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
