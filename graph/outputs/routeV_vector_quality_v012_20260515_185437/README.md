# RouteV V012 Output Snapshot

This committed output snapshot contains only small analysis/provenance files
from `routeV_vector_quality_v012_20260515_185437`.

Included:
- Top-level `routeV_summary.json`.
- Per-variant `run_summary.json`, `run_config.json`, and
  `routeV_variant_config.json`.
- Per-variant graph metrics and edge statistics.
- Per-variant review-training metrics, training history, and proxy selection.
- Control baseline manifest and compact metric summaries.

Intentionally omitted:
- Review encoder checkpoints (`*.pt`).
- Dense vectors and masks (`*.npy`).
- Large prepared/raw data files.
- Full review output, user-score, review-score, and edge CSV artifacts.

Formal target row is `Base_LogicAE_CB` with
`current_egat_edge_aware_gat`. The best V012 result is `V1a_supcon_reg`
with AUC `0.8591378973182074`, AP `0.857937837920376`, and F1
`0.7816901408450704`.
