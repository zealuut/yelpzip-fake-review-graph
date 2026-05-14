# routeD1_fresh_repro

This route has two separate D1 baseline roles:

1. `fresh_yelpcurrent_artifact_plus_d1_graph_baseline`: the branch-approved
   repeat01 fresh yelpcurrent artifact retrain plus RouteD D1 graph rerun used
   as the current governance D1 strong baseline.
2. `fresh_d1_train`: a same-protocol fresh retrain path used for
   reproducibility diagnostics and paired controls when encoder/head/loss code
   changes.

The current canonical baseline is not the older fixed `0.85637` graph-only pack.
It is a complete newly retrained artifact followed by the validated D1 graph
stage. The fresh runner below remains useful for paired controls, but repeated
training is not bit-stable, so promoted baselines must be explicit.

The route must not read historical D1 output directories for training inputs.
In particular, it must not reuse old review encoder checkpoints, review scores,
text vectors, abnormal vectors, self-feature matrices, graph edges, predictions,
or metrics.

Canonical governance D1 baseline:

- Stable alias: `graph/outputs/routeD1_fresh_repro_CANONICAL_D1`
- Baseline alias: `graph/outputs/routeD1_YELPCURRENT_RETRAIN_BASELINE`
- Timestamped copy:
  `graph/outputs/routeD1_yelpcurrent_retrain_baseline_repeat01_20260514_183204`
- Full retrained artifact:
  `graph/outputs/routeD1_yelpcurrent_retrain_baseline_repeat01_20260514_183204/artifact`
- D1 graph rerun:
  `graph/outputs/routeD1_yelpcurrent_retrain_baseline_repeat01_20260514_183204/d1_graph`
- Manifest: `baseline_manifest.json`
- Target row: `Base_LogicAE_CB` / `current_egat_edge_aware_gat`
- Metrics: AUC `0.8545569793813736`, AP `0.8522126584244006`,
  F1 `0.7827278958190541`, Recall `0.856071964017991`,
  Precision `0.7209595959595959`.
- Review artifact checkpoint: epoch `1`, val AUC `0.7496077001`.

Use this baseline for default D1 comparisons and fixed-D1 route work. If an
experiment changes review encoder, abnormal head, aux loss, mask semantics, or
vector-producing code, regenerate affected artifacts and compare against a
paired fresh no-change control.

Older fixed RouteD graph-only pack:

- `graph/outputs/routeD1_fresh_repro_canonical_strongest_20260514_0p856370`
- AUC `0.8563709149922789`; kept for audit/reference, no longer the primary
  governance baseline after repeat01 promotion.

Fresh runner contract:

Allowed external inputs:

- Raw YelpZip graph data under `/home/xyz/HuChao (2)/Bert-TextClassification/graph data`.
- Local pretrained model under `/home/xyz/HuChao (2)/Bert-TextClassification/local_models/roberta-base`.

Generated inside the route output:

- `prepared_data/`
- `llm_mask/` with full-text masks for the selected tokenizer/sequence length
- `review_encoder/best_review_encoder.pt`
- `logic_vectors/` review/user text and abnormal vectors
- `review_scores_enriched.csv`
- `user_scores_enriched.csv`
- `edges/` including UPU/UTU/USU/TextSim/CB/LogicAE_CB
- `metrics/model_results.csv`
- `artifact_reuse.json`
- `d1_fresh_repro_audit.json`

Recovered D1 policy:

- The no-aux governance D1 control uses fixed `num_epochs=2` for the review
  encoder. The historical D1 base artifact also selected epoch 2.
- A 3-epoch fresh run produced stronger review-level train metrics but shifted
  the user-vector/top-k edge distribution and dropped `Base_LogicAE_CB` EGAT to
  about `0.8206` AUC.
- The recovered epoch-2 fresh control reached AUC `0.848467820062982`, AP
  `0.843816985438882`, and F1 `0.7772380291464261` in
  `graph/outputs/routeD1_fresh_repro_epoch2_20260514_112300`.
- The branch-approved canonical pack is `routeD1_fresh_repro_CANONICAL_D1`,
  now pointed at the repeat01 fresh yelpcurrent artifact plus D1 graph result
  with AUC `0.8545569793813736`.
- The epoch-2 fresh control is no longer the default best D1 result. It is kept
  as a paired-control diagnostic for experiments that must retrain changed
  encoder/head/loss components.

Scientific use:

- Default D1 comparisons should use the canonical locked pack above.
- Fresh runner outputs are for reproducibility diagnosis and paired controls
  when an experiment changes encoder/head/loss/vector-producing code.
- If a fresh no-change control does not reach its declared floor, stop the
  downstream queue and treat the result as a reproduction failure diagnosis.

Run through tmux-safe queue:

```bash
bash graph/routes/routeD1_fresh_repro/scripts/run_routeD1_fresh_repro_queue.sh
```

Run directly:

```bash
python3 -u graph/routes/routeD1_fresh_repro/scripts/run_routeD1_fresh_repro.py \
  --output_root graph/outputs/routeD1_fresh_repro_manual
```
