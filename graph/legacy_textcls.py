from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import main as legacy_main  # noqa: E402
from SyntaxAwareSubSentence import args as sass_args  # noqa: E402
from roberta import args as roberta_args  # noqa: E402


LEGACY_ARG_MODULES = {
    "SyntaxAwareSubSentence": sass_args,
    "roberta": roberta_args,
}


def _build_legacy_config(
    model_name: str,
    data_dir: Path,
    output_root: Path,
    roberta_model_dir: str,
    gpu_ids: str,
    num_epochs: int,
    train_batch_size: int,
    dev_batch_size: int,
    test_batch_size: int,
    print_step: int,
    early_stop: int,
) -> Any:
    arg_module = LEGACY_ARG_MODULES[model_name]
    saved_argv = sys.argv[:]
    sys.argv = [saved_argv[0]]
    try:
        config = arg_module.get_args(
            str(data_dir),
            str(output_root) + os.sep,
            str(output_root / "cache") + os.sep,
            str(PROJECT_ROOT / "pretrain_model" / "roberta-base" / "vocab.json"),
            roberta_model_dir,
            str(output_root / "logs") + os.sep,
        )
    finally:
        sys.argv = saved_argv

    model_output_dir = output_root / model_name
    config.model_name = model_name
    config.save_name = f"{model_name}_graph_yelpzip"
    config.data_dir = str(data_dir)
    config.output_dir = str(model_output_dir) + os.sep
    config.cache_dir = str(output_root / "cache" / model_name) + os.sep
    config.log_dir = str(output_root / "logs" / model_name) + os.sep
    config.roberta_model_dir = roberta_model_dir
    config.train_batch_size = train_batch_size
    config.dev_batch_size = dev_batch_size
    config.test_batch_size = test_batch_size
    config.num_train_epochs = float(num_epochs)
    config.print_step = print_step
    config.early_stop = early_stop
    config.gpu_ids = gpu_ids
    config.version_tag = f"graph_final_{model_name.lower()}_yelpzip"
    config.change_note = "graph pipeline exported user-split YelpZip review TSVs for legacy review-level baselines"
    config.previous_result_summary = "graph final experiment legacy review-level baseline"
    return config


def _parse_legacy_test_result(result_path: Path) -> dict[str, Any]:
    if not result_path.exists():
        raise FileNotFoundError(f"Legacy test_result.txt not found: {result_path}")
    text = result_path.read_text(encoding="utf-8", errors="ignore")

    loss_match = re.search(r"Loss:\s*([0-9.]+)", text)
    acc_match = re.search(r"Acc:\s*([0-9.]+)\s*%", text)
    auc_match = re.search(r"AUC:([0-9.]+)", text)
    return {
        "loss": float(loss_match.group(1)) if loss_match else None,
        "accuracy": float(acc_match.group(1)) / 100.0 if acc_match else None,
        "auc": float(auc_match.group(1)) if auc_match else None,
        "result_path": str(result_path),
    }


def run_legacy_review_baselines(
    data_dir: str | Path,
    output_dir: str | Path,
    roberta_model_dir: str,
    gpu_ids: str,
    num_epochs: int,
    train_batch_size: int,
    dev_batch_size: int,
    test_batch_size: int,
    print_step: int,
    early_stop: int,
) -> pd.DataFrame:
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(PROJECT_ROOT)

    rows: list[dict[str, Any]] = []
    for model_name in ["SyntaxAwareSubSentence", "roberta"]:
        config = _build_legacy_config(
            model_name=model_name,
            data_dir=data_dir,
            output_root=output_dir,
            roberta_model_dir=roberta_model_dir,
            gpu_ids=gpu_ids,
            num_epochs=num_epochs,
            train_batch_size=train_batch_size,
            dev_batch_size=dev_batch_size,
            test_batch_size=test_batch_size,
            print_step=print_step,
            early_stop=early_stop,
        )
        legacy_main(config, config.save_name, ["0", "1"], False)
        result_path = Path(config.output_dir) / "test_result.txt"
        metrics = _parse_legacy_test_result(result_path)
        rows.append(
            {
                "model_name": model_name,
                "data_dir": str(data_dir),
                "output_dir": str(config.output_dir),
                **metrics,
            }
        )

    results_df = pd.DataFrame(rows)
    results_df.to_csv(output_dir / "legacy_baseline_results.csv", index=False)
    (output_dir / "legacy_baseline_results.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return results_df
