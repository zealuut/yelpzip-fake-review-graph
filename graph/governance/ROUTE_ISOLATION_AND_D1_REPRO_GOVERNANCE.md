# Route Isolation & D1 Repro Governance

## Purpose

This branch establishes a governance layer around the historical `ce6a9d6` code line so that:

- core graph behavior remains backward-compatible by default;
- route-specific experiments stop mutating shared defaults;
- D1 can be reproduced from a frozen artifact pack;
- future route work must pass a D1 smoke gate before claiming to be "based on D1".

## Governance Rules

### 1. `graph/core/` is the stable core layer

Core should eventually hold shared, backward-compatible code only:

- `data_utils.py`
- `feature_builder.py`
- `graph_builder.py`
- `egat_model.py`
- `metrics.py`
- `artifact_io.py`

Rules:

- core may only receive backward-compatible changes;
- no route-specific default behavior is allowed in core;
- any new behavior must be behind an explicit config flag;
- default flags must reproduce the historical D1 behavior.

Examples:

- `topk_mode = fixed20 | abnormal_tns | bandit`
- `review_encoder = d1_original | routeL_texthead | ...`

Never silently change shared defaults inside core.

### 2. `graph/routes/` isolates experiment lines

Each route should own its:

- `configs/`
- `scripts/`
- `src/`
- `README.md`

Route code may import core, but core must not import route code.

This prevents failures like:

- changing `review_training.py` for Route L;
- accidentally shifting D1 / K0 / D-route behavior.

### 3. `graph/outputs/` must preserve reproducibility evidence

Every run should preserve:

- `run_config.json`
- `git_commit.txt`
- `changed_files.txt`
- `feature_hashes.json`
- `edge_hashes.json`
- `split_hash.json`
- `label_hash.json`
- `metrics/model_results.csv`

For D1-like mainline runs, a reproducibility pack is required:

- `graph/outputs/D1_REPRO_PACK/`
  - `final_self_feature_matrix.npy`
  - `user_order.json`
  - `label_vector.npy`
  - `split_indices.json`
  - `edge_pack/`
  - `edge_weight_hashes.json`
  - `config_frozen.json`
  - `git_commit.txt`

Without this, later "K0 should equal D1" checks remain ambiguous.

## Core Candidates in `ce6a9d6`

Current likely core candidates:

- `graph/data_utils.py`
- `graph/graph_pipeline.py`
- `graph/relation_model.py`
- `graph/review_models.py`
- `graph/review_training.py`
- `graph/run_final_experiment.py`
- `graph/scripts/route_runner.py`

These are **legacy core candidates**, not automatically approved for mutation.

## Files Modified Later by D / K / L / TextHeads Work

Observed after `ce6a9d6` on `main`:

- `graph/graph_pipeline.py`
- `graph/relation_model.py`
- `graph/run_all.env.sh`
- `graph/scripts/route_runner.py`
- `graph/scripts/routek_runner.py`
- `graph/scripts/routek_d1main_rns_runner.py`
- `graph/scripts/run_routeK_adaptive_topk.sh`
- `graph/scripts/run_routeK_d1main_rns_topk.sh`
- `graph/scripts/run_routeD_tns_confirmed_logic_egat_v2.sh`
- `graph/scripts/run_routeD_tns_heavy_logic.sh`
- `graph/scripts/build_tns_heavy_features.py`
- `graph/routeL_llmmask_anomaly_aux/**`
- `graph/baseline_comparison/**`

These must be treated as route-specific or post-D1 extensions.

## D1 Smoke Gate

Any future core change must pass a D1 smoke gate before being treated as D1-compatible.

Minimum requirement:

- rebuild K0 fixed top20;
- `abs(AUC - D1_AUC) <= 0.003`
- `abs(AP - D1_AP) <= 0.003`

If not, the branch must be marked as:

- `NOT_STRICT_D1_COMPATIBLE`

and downstream routes must not claim "based on D1" without qualification.

## Immediate Intent of This Branch

This governance branch does **not** rewrite the entire repository. It provides:

- isolation scaffolding;
- a D1 reproducibility pack builder;
- a D1 smoke test entry point;
- explicit governance documentation.

It is a control surface for future cleanup, not a full refactor.
