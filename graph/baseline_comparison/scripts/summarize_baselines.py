from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from graph.baseline_comparison.src.build_full_base_graph import load_full_base_bundle
from graph.baseline_comparison.src.data_loader import load_protocol_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--protocol", default="current_topk", choices=["current_topk", "full_base"])
    parser.add_argument("--summary-prefix", default="baseline_summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    rows: list[dict] = []

    for run_summary_path in sorted(output_root.glob("*/run_summary.json")):
        payload = json.loads(run_summary_path.read_text(encoding="utf-8"))
        metrics = payload["metrics"]
        rows.append(metrics)

    bundle = load_full_base_bundle() if args.protocol == "full_base" else load_protocol_bundle()
    rows.append(bundle.reference_metrics)
    summary_df = pd.DataFrame(rows)
    csv_path = output_root / f"{args.summary_prefix}.csv"
    md_path = output_root / f"{args.summary_prefix}.md"
    summary_df.to_csv(csv_path, index=False)
    try:
        markdown = summary_df.to_markdown(index=False)
    except Exception:
        markdown = "# Baseline Summary\n\n```csv\n" + summary_df.to_csv(index=False) + "```\n"
    md_path.write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
