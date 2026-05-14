# D1 Baseline Metric Comparison

Current promoted baseline: `repeat01` fresh `yelpcurrent` artifact retrain plus RouteD D1 graph rerun.

Observation: among the two completed retrain repeats, review validation AUC does not track D1 graph AUC. If anything, the observed direction is inverse, but `n=2` is too small for a statistical conclusion.

| name | review_epoch | review_val_auc | d1_auc | d1_ap | d1_f1 | d1_recall | d1_precision | review_val_auc_source | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| repeat01 | 1 | 0.749608 | 0.854557 | 0.852213 | 0.782728 | 0.856072 | 0.720960 | artifact/review_encoder/review_encoder_metrics.json | Same split/protocol/seed=42; repeated training is not bit-stable. |
| repeat02 | 2 | 0.755008 | 0.846499 | 0.840664 | 0.780552 | 0.890555 | 0.694737 | artifact/review_encoder/review_encoder_metrics.json | Same split/protocol/seed=42; repeated training is not bit-stable. |
| CANONICAL_D1_repeat01_fresh_yelpcurrent_plus_D1_graph | 1 | 0.749608 | 0.854557 | 0.852213 | 0.782728 | 0.856072 | 0.720960 | baseline_manifest.json copied from repeat01 artifact metrics | Current governance baseline selected by user; same values as repeat01. |
| old_RouteD_D1_graph_only_0p85637 | 2 | 0.753835 | 0.856371 | 0.858369 | 0.778172 | 0.823088 | 0.737903 | not stored in RouteD output; inherited from base_dir review_encoder_metrics.json | Audit/reference only; graph-only run from old yelpcurrent artifact, no fresh retrain. |
| governance_epoch2_fresh_repro_diagnostic | 2 | 0.753368 | 0.848468 | 0.843817 | 0.777238 | 0.839580 | 0.723514 | review_encoder/review_encoder_metrics.json | Fresh diagnostic; not the promoted baseline. |

Notes:
- `old_RouteD_D1_graph_only_0p85637` does not contain review metrics directly in its RouteD output; its review val AUC is inherited from the old `base_dir` artifact.
- Rows with missing review metrics would be marked `NA`; no such row was needed here because the relevant review metrics were recoverable from source artifacts.
