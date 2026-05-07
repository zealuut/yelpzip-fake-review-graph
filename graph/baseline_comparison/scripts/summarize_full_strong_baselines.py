from __future__ import annotations

import argparse
from pathlib import Path

from graph.baseline_comparison.scripts.summarize_baselines import main as summarize_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import sys

    sys.argv = [
        "summarize_baselines",
        "--output-root",
        str(Path(args.output_root)),
        "--protocol",
        "full_base",
        "--summary-prefix",
        "full_strong_baseline_summary",
    ]
    summarize_main()


if __name__ == "__main__":
    main()
