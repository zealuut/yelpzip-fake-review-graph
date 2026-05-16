# Route V: Vector-Quality-Aware Training

## Scientific Question

The review encoder's abnormal head is trained to classify fake reviews (review-level BCE),
but the downstream graph model needs user-level abnormal vectors with good separability.
Empirical evidence shows these objectives are misaligned: higher review val AUC correlates
with *lower* graph AUC (see `baseline_metric_comparison.md` in the canonical D1 output).

Can we improve graph-stage performance by optimizing for vector quality rather than
review-level classification accuracy?

## Experiment Type

`fresh_d1_train` — all variants train the review encoder from scratch under the D1
`ce6a9d6` protocol. No frozen vectors, checkpoints, or edges are reused as primary
comparison artifacts.

Strict label policy:
- Review BCE still uses `review_label`.
- V0 checkpoint proxy, V1 regularizer, and V2 graph-head contrastive objective
  use explicit `user_label` from `prepared.user_df.user_label`.
- `review_label.max()` is not an allowed implementation source for RouteV vector
  proxy or vector regularization, even if it happens to match the current data.

## Hypothesis

The review-level BCE loss compresses the abnormal vector space around a classification
boundary. When aggregated to user level (top-m mean pooling), this compressed space
loses structural information that the graph model needs. By either:
- selecting checkpoints that maximize user-level vector quality (V0), or
- adding a user-level contrastive regularizer during training (V1), or
- decoupling the graph vector from the classification head entirely (V2),

we can produce abnormal vectors that yield better graph AUC without degrading the
overall pipeline.

## Variants

| Name | Change | Architecture Modified? |
|------|--------|----------------------|
| V_control | Route baseline pack control: promoted fresh artifact plus D1 graph stage | No |
| V0_proxy_checkpoint | Same training, checkpoint selected by user-level vector AUC using `user_label` | No |
| V1a_supcon_reg | Training loss += lambda * SupCon(user_vectors, `user_label`) | No (loss only) |
| V1b_triplet_reg | Training loss += lambda * TripletMargin(user_vectors, `user_label`) | No (loss only) |
| V2_dual_head | Separate graph_vector head with user-level contrastive loss using `user_label` | Yes (additive) |

## Baseline / Control

- **Route baseline reference** is
  `graph/outputs/routeBaseline_D1_FRESH_RETRAIN_BASELINE`: the promoted best-of-5
  full fresh retrain artifact plus RouteD/D1 graph-stage result. It is not a
  fixed-artifact graph-only rerun.
- **V_control** is now constructed identically to that route baseline pack. The
  runner copies the promoted pack's `artifact/` and paired `d1_graph/` into the
  RouteV output and reads the target graph row from
  `d1_graph/metrics/model_results.csv`. It is the in-route comparison control
  for V0/V1/V2 and must pass the D1 floor (graph AUC >= 0.840) before any
  variant is interpreted.
- `graph/routes/baseline/` owns the primary fresh route baseline. Its old
  fixed-artifact strongest pack is audit/reference only.
- Route baseline target row:
  `Base_LogicAE_CB` with `current_egat_edge_aware_gat`, AUC `0.8546`, AP `0.8522`.
- RouteV outputs write `baseline_metadata` into `run_config.json` and
  `run_summary.json`; top-level queue summaries also record the route contract
  and reference baselines.

## Historical Failed Controls

`graph/outputs/routeV_vector_quality_v012_20260514_231534` is a failed control-gate
run. It ran `V_control`, selected epoch 3 by review val AUC, and produced graph
AUC `0.8199236213976968`, below the floor `0.84` and below the fresh route
baseline AUC `0.8545569793813736`. The queue stopped before formal V0/V1/V2
variants, so V012 must not be interpreted as evidence for those variants.

`graph/outputs/routeV_vector_quality_strict_20260515_140020` also failed under
the old RouteV-local control construction, selecting epoch 3 and producing graph
AUC `0.806025772720836`. These failed controls motivated changing `V_control`
to match the promoted route baseline construction exactly.

## V012 Baseline-Pack Snapshot

`graph/outputs/routeV_vector_quality_v012_20260515_185437` is the first formal
V012 run after changing `V_control` to the promoted route-baseline-pack
construction. `V0`, `V1a`, and `V2` are fresh retrains; they do not reuse the
control checkpoint/vectors/edges.

| Variant | AUC | AP | F1 | Recall | Precision | Note |
| --- | --- | --- | --- | --- | --- | --- |
| V_control | 0.854557 | 0.852213 | 0.782728 | 0.856072 | 0.720960 | Route baseline pack control |
| V0_proxy_checkpoint | 0.853426 | 0.851646 | 0.773869 | 0.808096 | 0.742424 | User-vector proxy selected epoch 1 |
| V1a_supcon_reg | 0.859138 | 0.857938 | 0.781690 | 0.832084 | 0.737052 | Best V012 result; +0.004581 AUC vs control |
| V2_dual_head | 0.841909 | 0.837583 | 0.770438 | 0.883058 | 0.683295 | Underperformed; do not continue unchanged |

Interpretation: user-level SupCon regularization is promising, but future
paper-facing variants that modify or attach training to the abnormal head must
fresh-train their own stage-1 and stage-2 checkpoints. V1a can motivate the
design, not serve as an initialization artifact for formal downstream variants.

## Key Evidence (from multirun diagnostic)

| Run | Review Val AUC | Graph AUC | Direction |
|-----|---------------|-----------|-----------|
| repeat01 (epoch 1) | 0.7496 | 0.8546 | Lower review -> Higher graph |
| repeat02 (epoch 2) | 0.7550 | 0.8465 | Higher review -> Lower graph |

## Artifact Reuse

Per governance rules, V0/V1/V2 regenerate:
- `best_review_encoder.pt`
- `review_text_vectors.npy`, `user_text_vectors.npy`
- `review_abnormal_vectors.npy`, `user_abnormal_vectors.npy`
- All downstream edges (TextSim, CB, LogicAE_CB, TNSGuided)
- Self-feature matrix

`V_control` is the exception: it is a route-baseline-pack control, so it copies
the promoted fresh baseline `artifact/` and paired `d1_graph/` under the RouteV
control output and records this reuse in `artifact_reuse`.

Allowed reuse (data protocol unchanged):
- `prepared_data/reviews_canonical.csv`, `users_canonical.csv`, `user_splits.csv`
- Static behavior edges (UPU, UTU, USU)

Each RouteV run writes an `artifact_reuse` section into `run_config.json` and
`run_summary.json`. It marks learned review checkpoints, abnormal/text vectors,
review scores, TextSim/CB/LogicAE/TNS-guided edges, and graph metrics as
`regenerated`.

## Configs

- `configs/routeV_variants.json`: queue config for V_control/V0/V1/V2.
- `configs/V0_proxy_checkpoint.json`: strict V0 contract.
- `configs/V1_vec_separability.json`: strict V1 contract.
- `configs/V2_dual_objective.json`: strict V2 contract.

V2 uses `detach_fusion=false` in the strict config so the graph-vector objective
shares the backbone with weighted gradients (`review_bce` weight 1.0,
`dual_head_lambda` for the vector objective), matching the RouteV design.

## Running

```bash
# Smoke test
python -m graph.routes.routeV_vector_quality.scripts.run_routeV_queue \
    --output_root graph/outputs/routeV_vector_quality_smoke_$(date +%Y%m%d_%H%M%S) \
    --smoke_test

# Full V012 queue used for the first RouteV run
python -m graph.routes.routeV_vector_quality.scripts.run_routeV_queue \
    --output_root graph/outputs/routeV_vector_quality_v012_$(date +%Y%m%d_%H%M%S) \
    --variants V_control V0_proxy_checkpoint V1a_supcon_reg V2_dual_head

# Full queue including the secondary V1 triplet variant
python -m graph.routes.routeV_vector_quality.scripts.run_routeV_queue \
    --output_root graph/outputs/routeV_vector_quality_$(date +%Y%m%d_%H%M%S)

# Recompute strict vector-quality metrics for an existing RouteV review output
python -m graph.routes.routeV_vector_quality.scripts.vector_quality_metrics \
    --review_vectors graph/outputs/<routeV_run>/<variant>/review_encoder/selected_review_vectors.npy \
    --review_output graph/outputs/<routeV_run>/<variant>/review_encoder/review_output.csv
```

### Exploratory Attach From V1a

`scripts/run_routeV_attach.py` is the V2b exploratory attach runner. It starts
from the V1a checkpoint only to test direction quickly, so its outputs are
marked `exploratory_only_checkpoint_reuse` and must not be used as formal
paper-facing fresh results.

The first queue uses fixed V1a top-m review selection and fixed
`Base_LogicAE_CB` topology, trains a frozen graph surrogate, and lets graph node
BCE gradients update only the abnormal/vector-side review encoder modules:

```bash
python -m graph.routes.routeV_vector_quality.scripts.run_routeV_attach \
    --output_root graph/outputs/routeV_attach_from_v1a_$(date +%Y%m%d_%H%M%S) \
    --config_path graph/routes/routeV_vector_quality/configs/routeV_attach_from_v1a.json
```

Initial variants:

- `V2_control_attach_zero_from_V1a`
- `V2b_beta002_from_V1a`
- `V2b_beta005_from_V1a`
- `V2b_beta010_from_V1a`

If a beta variant beats V1a in this exploratory setting, the formal follow-up
must be a fresh RouteV attach run with its own stage-1 checkpoint, regenerated
vectors, fixed-control run, and graph-stage artifacts.

## Implementation Status

- [x] V0: User-level proxy checkpoint selection with explicit `user_label` (`src/user_level_proxy.py`)
- [x] V1: Vector separability regularizer with explicit `user_label` (`src/vector_reg_loss.py`)
- [x] V2: Dual-head wrapper (`src/dual_head_encoder.py`)
- [x] V2b exploratory attach runner with frozen graph-loss backprop (`scripts/run_routeV_attach.py`)
- [x] Route-local dataloader with user ids per batch
- [x] Route-local dataloader with explicit `user_label` per review row
- [x] Route-local training loop with per-epoch checkpoints
- [x] Per-variant strict configs and vector-quality metrics CLI
- [x] Queue runner that fails hard on real errors

## Integration Notes

V0/V1/V2 are now integrated in `scripts/run_routeV_queue.py` without modifying shared
D1 training code. The route-local dataset adds `user_id_idx` to each batch, V1 applies
the regularizer to `review_vector`, and V2 applies it to the separate `graph_vector`
head. V0 checkpoint selection uses a train-user fitted, val-user evaluated proxy rather
than fitting and evaluating on the same validation users. All vector-quality labels are
attached from `prepared.user_df.user_label` before dataloader construction and are also
written into `review_encoder/review_output.csv` for audit.
