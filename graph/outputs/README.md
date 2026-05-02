## Curated Outputs

This repository intentionally tracks only small, analysis-friendly artifacts from selected experiment runs under `graph/outputs/`.

Tracked output files are limited to items such as:

- `run_summary.json`
- `prepared_data/dataset_metadata.json`
- `metrics/model_results*.csv`
- `metrics/edge_stats.csv`
- `edges/edge_build_config.json`
- analysis markdown notes

Large or redundant artifacts remain untracked, including:

- checkpoints and model weights
- cached LLM outputs
- full prepared datasets
- review encodings
- logs and temporary files

If a new run produces useful lightweight artifacts, add only the specific small files that are needed for interpretation or reproduction.
