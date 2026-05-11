# Repository Agent Rules

This repository uses **route isolation** and **D1 reproducibility governance**.

## 1. Route Isolation Is Mandatory

Every experimental line must own its own folder under:

```text
graph/routes/
```

Expected structure:

```text
graph/routes/
  routeD_d1_main/
  routeK_topk/
  routeL_text_anomaly/
  routeTextHeads/
  baseline_comparison/
```

Each route should keep its own:

- `configs/`
- `scripts/`
- `src/`
- `README.md`

### Hard Rule

Work for one route must not silently change another route.

That means:

- Route L logic must not be added directly into shared D/K codepaths.
- Route K logic must not silently alter D1 behavior.
- Texthead experiments must not overwrite mainline graph defaults.

## 2. Shared / Main Logic Must Remain Backward-Compatible

Shared logic belongs in the stable core layer target:

```text
graph/core/
```

Until the full refactor is complete, the current shared legacy files are treated as **core candidates**:

- `graph/data_utils.py`
- `graph/graph_pipeline.py`
- `graph/relation_model.py`
- `graph/review_models.py`
- `graph/review_training.py`
- `graph/run_final_experiment.py`
- `graph/scripts/route_runner.py`

### Hard Rule

If main/core logic must be changed:

- the change must be backward-compatible;
- the old behavior must remain the default;
- new behavior must be behind an explicit config flag;
- no route may change shared defaults just to support itself.

Examples of acceptable config gating:

- `topk_mode = fixed20 | abnormal_tns | bandit`
- `review_encoder = d1_original | routeL_texthead | ...`
- `use_aux_loss = false | true`

Examples of unacceptable changes:

- silently changing default graph construction;
- silently changing default review encoder behavior;
- changing shared edge weighting logic without a flag;
- changing default feature composition across all routes.

## 3. Core Must Not Import Route Code

Allowed:

- route code imports core/shared logic

Forbidden:

- core imports route-specific code

This prevents route-specific experiments from becoming hidden global dependencies.

## 4. D1 Compatibility Gate

Any change that affects shared graph behavior, shared features, or shared model code must be checked against D1 reproducibility.

Minimum rule:

- run the D1 smoke test;
- if K0 fixed-top20 cannot stay within:
  - `abs(AUC - D1_AUC) <= 0.003`
  - `abs(AP - D1_AP) <= 0.003`
- then the branch must be marked as **not strict D1 compatible**.

Relevant governance files:

- `graph/governance/ROUTE_ISOLATION_AND_D1_REPRO_GOVERNANCE.md`
- `graph/governance/d1_smoke_test.py`
- `graph/outputs/D1_REPRO_PACK/`

## 5. D1 Repro Pack Must Be Preserved

The frozen D1 reproducibility pack is the reference anchor for future work:

```text
graph/outputs/D1_REPRO_PACK/
```

It contains:

- final self feature matrix
- user order
- label vector
- split indices
- D1 edge pack
- edge hashes
- frozen config
- git commit marker

Do not replace or mutate this pack casually.
If a new repro pack must be created, create a new versioned pack instead of overwriting the old one.

## 6. Output Governance

Every meaningful run should preserve analysis-grade artifacts, at minimum:

- `run_config.json`
- `git_commit.txt`
- `changed_files.txt`
- `feature_hashes.json`
- `edge_hashes.json`
- `split_hash.json`
- `label_hash.json`
- `metrics/model_results.csv`

Large artifacts (models, full edge CSVs, large `.npy`) may be excluded from lightweight pushes, but analysis artifacts must remain.

## 7. Safety Rule for Agents

Before editing:

1. determine whether the task belongs to a specific route;
2. if yes, implement inside that route folder first;
3. only touch shared/main code if absolutely necessary;
4. if touching shared/main code, preserve default behavior and document the compatibility flag.

## 8. Practical Default

If there is any ambiguity:

- isolate the change under `graph/routes/<route_name>/`
- do **not** modify shared defaults
- do **not** let one route change another route’s behavior

This repository prefers duplicated route-local experiment logic over unsafe shared mutation.
