from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from graph.routes.routeTNSGD_GroupFirst_NodeFeature.src.tnsgd_group_first import (
    TNSGDConfig,
    run_tnsgd_group_first,
)


def _load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _config_from_raw(raw: dict[str, Any], *, phi_days: int | None = None) -> TNSGDConfig:
    return TNSGDConfig(
        experiment_name=str(raw.get("experiment_name", "TNSGD-GroupFirst")),
        phi_days=int(phi_days if phi_days is not None else raw.get("phi_days", 5)),
        delta_I=float(raw.get("delta_I", 0.5)),
        merge_jaccard=float(raw.get("merge_jaccard", 0.8)),
        top_sequence_pool=int(raw.get("top_sequence_pool", 300)),
        strategy_top_n=int(raw.get("strategy_top_n", 30)),
        strategy_last_n=int(raw.get("strategy_last_n", 0)),
        min_raw_group_size=int(raw.get("min_raw_group_size", 3)),
        min_core_group_size=int(raw.get("min_core_group_size", 2)),
        group_size_norm=float(raw.get("group_size_norm", 10.0)),
        asset_project_root=Path(str(raw.get("asset_project_root", "/home/xyz/HuChao (2)/Bert-TextClassification"))),
        base_protocol_dir=Path(
            str(
                raw.get(
                    "base_protocol_dir",
                    "/home/xyz/HuChao (2)/Bert-TextClassification/graph/outputs/yelpzip_balanced_current_graph_no_reweight_20260502_160620",
                )
            )
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--run_grid", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    raw = _load_config(config_path)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root) if args.output_root else PROJECT_ROOT / "graph" / "outputs" / f"routeTNSGD_GroupFirst_NodeFeature_{ts}"
    output_root.mkdir(parents=True, exist_ok=True)

    if args.run_grid:
        rows: list[dict[str, Any]] = []
        for phi in raw.get("phi_days_grid", [1, 3, 5, 7]):
            cfg = _config_from_raw(raw, phi_days=int(phi))
            phi_dir = output_root / f"phi{cfg.phi_days}"
            summary = run_tnsgd_group_first(cfg, phi_dir)
            rows.append(summary)
        pd.DataFrame(rows).to_csv(output_root / "tnsgd_grid_summary.csv", index=False)
        print(output_root)
        return

    cfg = _config_from_raw(raw)
    summary = run_tnsgd_group_first(cfg, output_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(output_root)


if __name__ == "__main__":
    main()
