# AGENTS.md

## Governance Branch Goal

This branch exists to preserve a recoverable D1 `ce6a9d6` baseline while
allowing small, route-owned experiments to run without corrupting the baseline
or each other.

The canonical strong D1 base for this branch is now the repeat01 fresh
`yelpcurrent` artifact retrain plus RouteD D1 graph rerun, promoted by user
decision after the May 14, 2026 multirun diagnostic. The stable alias is
`graph/outputs/routeD1_fresh_repro_CANONICAL_D1`, which points to
`graph/outputs/routeD1_yelpcurrent_retrain_baseline_repeat01_20260514_183204`.
The baseline manifest is `baseline_manifest.json` inside that output.

This is the default governance D1 baseline for downstream route comparisons:
it uses a newly retrained complete artifact, then the validated D1 graph stage.
It should be preferred over the older fixed `0.85637` RouteD graph-only pack
when the question is whether a fresh yelpcurrent artifact can support D1. For
any experiment that changes the review encoder, abnormal/text head, training
loss, vector generation, or graph training code, still run a same-protocol
fresh no-change control as the paired baseline and regenerate all affected
vectors/edges/checkpoints.

Primary active routes:

- `graph/routes/routeK_topk/`: K-line top-k and candidate-pool experiments.
- `graph/routes/routeG_egatpp/`: G-line EGAT++ graph-backbone experiments.
- `graph/routes/routeL_abnormal_aux_head/`: abnormal aux-head and internal D1
  abnormal-head ablations.
- `graph/routes/routeD1_fresh_repro/`: canonical locked D1 base metadata plus
  branch-local fresh-reproduction diagnostics and paired-control runner.
- `graph/routes/routeTNSGD_GroupFirst_NodeFeature/`: group-first TNSGD-style
  unsupervised group discovery and node-feature export.

## Experiment Type Must Be Declared

Every route or queue must decide which type it is before running:

- `canonical_locked_d1_base`: the branch-approved locked old D1 pack copied
  into governance outputs and used as the current best D1 baseline. It is
  artifact-anchored by design and must carry provenance.
- `fresh_d1_train`: trains the review encoder/vectors and graph model from
  scratch under the D1 `ce6a9d6` protocol. This is required for aux-loss,
  review-model, abnormal-head, vector-generation, or training-loss changes.
- `artifact_anchored_graph_diag`: reads fixed D1 artifacts only for
  graph-only/topology/edge-weight diagnostics. These runs may be useful, but
  they must not be described as strict fresh D1 reproduction.
- `analysis_only`: reads existing outputs and writes diagnostics only.

Each experiment output should record this type in its config/summary when
possible. If a script reuses `base_dir`, `logic_vectors`, `review_scores`,
prepared data, or old edges, its README/config/log must say that it is
artifact-anchored.

## Strict D1 Base Policy

Strict D1 base means:

1. Same D1 `ce6a9d6` protocol and defaults.
2. Same data split policy, seed, graph data source, model path, mask source,
   top-k/top-m, edge-set, graph mode, and relation model.
3. Fresh training of the changed pipeline, not reuse of frozen review vectors,
   abnormal vectors, graph edges, or checkpoints.
4. A no-change control must run first for any route that changes encoder,
   vector, loss, graph-model, or shared training code.

For D1-preserving routes, the first control must be named clearly, such as
`D1_NO_AUX_fresh_control`, `K0_StrictControl`, or `G0_D1_Control`. If that
control does not reproduce the D1 floor within the route's declared tolerance,
stop the queue and do not interpret downstream variants.

Canonical governance D1 baseline for future default use:

- Stable alias:
  `graph/outputs/routeD1_fresh_repro_CANONICAL_D1`
- Baseline alias:
  `graph/outputs/routeD1_YELPCURRENT_RETRAIN_BASELINE`
- Timestamped copy:
  `graph/outputs/routeD1_yelpcurrent_retrain_baseline_repeat01_20260514_183204`
- Full retrained artifact:
  `graph/outputs/routeD1_yelpcurrent_retrain_baseline_repeat01_20260514_183204/artifact`
- D1 graph rerun from that artifact:
  `graph/outputs/routeD1_yelpcurrent_retrain_baseline_repeat01_20260514_183204/d1_graph`
- Manifest:
  `baseline_manifest.json`
- Target row: `Base_LogicAE_CB` with `current_egat_edge_aware_gat`
- Metrics: AUC `0.8545569793813736`, AP `0.8522126584244006`,
  F1 `0.7827278958190541`, Recall `0.856071964017991`,
  Precision `0.7209595959595959`.
- Review artifact checkpoint: epoch `1`, validation AUC `0.7496077001`.

Use this fresh-artifact-plus-D1-graph baseline as the branch's current strong D1
baseline unless an experiment explicitly requires a newly paired control because
it changes vector-producing or training code.

Older fixed RouteD graph-only reference, kept for audit but no longer primary:

Historical reference D1 for graph `Base_LogicAE_CB` is:

- AUC: `0.8563709149922789`
- AP: `0.858368711617606`
- F1: `0.7781715095676824`
- Source output:
  `graph/outputs/routeD_tns_guided_logic_egat_20260504_200855/D1_EGAT_Base_LogicAE_CB`

Governance recovered fresh D1 control, for reproducibility diagnostics and
paired fresh controls:

- Route: `graph/routes/routeD1_fresh_repro/`
- Required review-encoder policy: fixed `num_epochs=2` for the no-aux D1
  control. The epoch-3 checkpoint overfits the review scorer and changes the
  user-vector/top-k edge distribution enough to invalidate D1 recovery.
- Reference recovery output:
  `graph/outputs/routeD1_fresh_repro_epoch2_20260514_112300`
- Metrics: AUC `0.848467820062982`, AP `0.843816985438882`,
  F1 `0.7772380291464261`.

Do not use `run_summary.best_graph_model` blindly when validating D1. It can
select `Behavior_LR` or another non-target row. Read `metrics/model_results.csv`
and select the intended `edge_set`/`model_name` row, usually
`Base_LogicAE_CB` with `current_egat_edge_aware_gat`.

## D1 Baseline Protection

Treat the D1 base as a scientific control, not just a compatibility target.

The following shared files are protected and must not be changed for routine
route experiments:

- `graph/graph_pipeline.py`
- `graph/relation_model.py`
- `graph/review_models.py`
- `graph/review_training.py`
- `graph/run_final_experiment.py`
- `graph/scripts/route_runner.py`

If a shared-file change is unavoidable:

1. Keep historical D1/default behavior unchanged.
2. Add any new behavior behind an explicit opt-in argument or config value.
3. Preserve RNG/init order for default D1. Do not instantiate new modules,
   layers, random tensors, samplers, or data shuffles in the default path unless
   the corresponding feature is enabled.
4. Document the reason in the owning route README or config.
5. Check `git diff -- <shared-file>` before and after the edit.
6. Run a smoke check and a D1 no-change control before launching a long queue.

Do not tune shared defaults to make a route result look better. Route-specific
logic belongs under the route directory unless there is a clear reusable reason
to promote it.

## Route Isolation Rules

Each route should be runnable from its own folder with its own scripts, configs,
README, logs/status conventions, and summaries. A route should not depend on
uncommitted behavior from another route unless that dependency is explicitly
documented.

When adding a new route:

1. Put runner scripts under `graph/routes/<route_name>/scripts/`.
2. Put route configs under `graph/routes/<route_name>/configs/` when useful.
3. Write a README explaining the scientific question, experiment type,
   baseline/control, and allowed artifact reuse.
4. Write outputs to `graph/outputs/<route_name>_<timestamp>/`.
5. Write logs/status to `graph/logs` or `graph/logs/status` in this branch.

Do not modify K/G/L/TNSGD route scripts to fix another route unless the change
is intentionally shared and documented.

## Artifact Reuse Rules

Artifact reuse must be decided per artifact, not per output directory. A route
may reuse one file from a historical D1 output and still be required to
regenerate another file from the same directory. Never say "reuse D1 artifact"
without naming which artifact class is reused.

Allowed for strict fresh D1 runs, if the data protocol is unchanged:

- Canonical data and split artifacts: `prepared_data/reviews_canonical.csv`,
  `prepared_data/users_canonical.csv`, `prepared_data/user_splits.csv`, and
  `prepared_data/dataset_metadata.json`. Reusing the split is encouraged to keep
  evaluation comparable. Regenerate these if filtering, balancing, sampling,
  preprocessing, label policy, or source data changes.
- Static user identity/behavior columns: `user_id`, `user_label`, `split`,
  review counts, rating/time statistics, `product_set`, `time_bucket_set`, and
  behavior-only scores such as `RD`, `EXR`, `MRO`, `AD`, `ATR`, and
  `behavior_anomaly_score`, provided the source data and behavior-feature code
  are unchanged. Do not blindly reuse a whole node-feature matrix if it also
  contains learned text/abnormal vectors.
- Static behavior edges: `UPU_edges.csv`, `UTU_edges.csv`, and `USU_edges.csv`
  may be reused only when graph mode, top-k, time bucket, source users/reviews,
  and the behavior-edge formulas are unchanged. These edges do not depend on
  learned text or abnormal vectors in the current D1 pipeline.

Only allowed for `artifact_anchored_graph_diag`, not strict fresh D1 claims:

- `TextSim_edges.csv`, because it is built from `user_text_vectors`.
- `CB_edges.csv`, because it is derived from TextSim candidates plus user
  features.
- `LogicAE_CB_edges.csv`, because it is built from `user_abnormal_vectors`.
- `TNSGuided_LogicAE_CB_edges.csv`, because it depends on `LogicAE_CB` plus
  temporal/review-score context.
- `GraphSupport_edges.csv`, edge stats, cached model predictions, and metrics.

Must be regenerated for any experiment that changes the review encoder,
abnormal/text head, mask semantics, auxiliary loss, pooling, gate, cross
attention, vector dimension, tokenizer/max length, or review-training code:

- `review_encoder/best_review_encoder.pt`.
- `logic_vectors/review_text_vectors.npy` and `logic_vectors/user_text_vectors.npy`.
- `logic_vectors/review_abnormal_vectors.npy`,
  `logic_vectors/user_abnormal_vectors.npy`, and
  `logic_vectors/user_abnormal_vectors_initial.npy`.
- `logic_vectors/review_abnormal_scores.csv`.
- `review_scores_enriched.csv`, because `p_fake_review`, `review_gate`, and
  `evidence_score` come from the current review encoder/head.
- Any precomputed self-feature matrix, because D1 self features concatenate
  dense behavior features with `user_abnormal_vectors`. Reuse only the verified
  behavior columns, then concatenate the newly generated abnormal vectors.
- `TextSim`, `CB`, `LogicAE_CB`, and TNS-guided logic edges if their upstream
  vectors or review-score inputs changed.

Conditional external-mask reuse:

- `llm_mask/abnormal_token_masks.npy` and `llm_mask/llm_review_features.csv`
  may be reused only if mask source, tokenizer/model path, max sequence length,
  review text, and mask-generation semantics are unchanged. If a route uses
  `mask_source=full_text`, these files are not evidence that LLM masks were
  used.

Artifact dependency map for current D1:

| Artifact | Main dependencies | Strict fresh reuse rule |
| --- | --- | --- |
| `prepared_data/*.csv`, `user_splits.csv` | source data, filtering, split seed | Reuse if data protocol unchanged |
| `UPU_edges.csv` | users/reviews, products, ratings/times, graph mode, top-k | Reuse if behavior topology unchanged |
| `UTU_edges.csv` | users, time buckets, graph mode, top-k | Reuse if behavior topology unchanged |
| `USU_edges.csv` | behavior/user stats, graph mode, top-k | Reuse if behavior topology unchanged |
| `user_summary.csv`, `user_scores_enriched.csv` | behavior aggregation over canonical reviews | Reuse only verified behavior columns |
| `review_scores_enriched.csv` | review encoder/head outputs plus review metadata | Regenerate if encoder/head/loss changes |
| `user_text_vectors.npy` | review text encoder/vector aggregation | Regenerate if encoder/loss/text representation changes |
| `user_abnormal_vectors.npy` | abnormal head/logic tower/vector aggregation | Regenerate if abnormal head, aux loss, mask, gate, pooling, or cross attention changes |
| `TextSim_edges.csv` | `user_text_vectors` | Regenerate if text vectors change |
| `CB_edges.csv` | TextSim candidates and user features | Regenerate if text vectors or relevant user features change |
| `LogicAE_CB_edges.csv` | `user_abnormal_vectors`, logic threshold/top-k | Regenerate if abnormal vectors or logic edge parameters change |
| `TNSGuided_LogicAE_CB_edges.csv` | LogicAE edges, review temporal context, TNS params | Regenerate if LogicAE/review scores/TNS params change |
| `best_review_encoder.pt` | review training code/objective/data | Never reuse as strict base for encoder/head/loss experiments |

Each route that reads historical artifacts must write an `artifact_reuse` section
in its config or summary with one row per artifact: `path`, `class`, `reuse_mode`
(`strict_reusable`, `artifact_anchored_only`, `regenerated`, or `forbidden`),
and `reason`. If this manifest is missing, treat the run as exploratory only.

## Main Directory Boundary

`/home/xyz/HuChao (2)/Bert-TextClassification` is the noisy main working tree.
Governance-branch scripts may read fixed reference artifacts from it, including
the D1 protocol outputs and prepared base artifacts, but they should not write
logs, outputs, configs, or code into that directory.

Never treat a main-directory result as governance-trusted unless the script,
commit/snapshot, data source, model path, and artifact reuse mode are recorded.
Main can be evidence, not authority.

Queue scripts in this branch must use branch-local paths:

- logs: `$ROOT_DIR/graph/logs`
- outputs: `$ROOT_DIR/graph/outputs`

Long governance queues should be launched through tmux with
`graph/routes/scripts/start_governance_kg_tmux.sh` so they survive SSH or
conversation disconnects. If a route has its own queue script, it must follow
the same branch-local log/output/status pattern.

## Queue Policy

Before running long queues, verify that the queue script is being launched from
this governance branch and that `ROOT_DIR` resolves to this branch root.

Every long queue should:

1. Print branch, commit, output dir, log file, and status file at startup.
2. Maintain a status file under `graph/logs/status/`.
3. Emit heartbeats with the active PID and latest output file.
4. Stop on failed strict-control/basecheck rather than continuing variants.
5. Summarize traceback/OOM/error lines on failure.

Do not leave a queue in a silent `waiting` or `starting` state without a status
reason and next observable action.

## K/G/L Specific Guardrails

K-line guardrails:

- If a K route claims to preserve D1 topology, `K0` must match D1 within the
  declared tolerance before K1/K2/K4 are interpreted.
- If the design is fixed-prefix top-k, top-10 must be a prefix of top-20 under
  the same scoring formula. If candidate-pool size changes the ordering, the
  scoring formula changed and must be documented as a different experiment.
- TNS-derived signals should not be added by default unless their diagnostic
  fake-rate/AUC/AP evidence supports the claim.

G-line guardrails:

- A G-line control must state whether it reuses historical D1 vectors/edges or
  trains fresh.
- Standard GAT/GraphSAGE/RGCN comparisons should live in independent route
  folders and use standard model definitions unless explicitly documented.

L-line guardrails:

- Aux-head experiments must use `fresh_d1_train` strict base, never frozen D1
  vectors/checkpoints as the primary comparison.
- `D1_NO_AUX_fresh_control` must pass before `D1_AUX_full` or ablations are
  interpreted.
- Aux heads intended to test abnormal text representations should not silently
  consume numeric features unless the variant name/config says so.

TNSGD/group-first guardrails:

- Group discovery is unsupervised. Do not directly label groups as fake during
  discovery.
- Use labels only after discovery for purity/coverage diagnostics.
- User-level group features should be treated as sparse context for graph
  models, not as direct classifier scores unless single-feature diagnostics
  justify that use.
