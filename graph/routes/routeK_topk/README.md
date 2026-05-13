# Route K Top-K Strategy

This route owns K-line graph strategy experiments.

Current contents include the D1 strict-attachment K2/K4 runner:

- `scripts/run_routeD1_kattach_k2k4_ce6a9d6.py`
- `scripts/run_routeD1_kattach_candidate_pool.py`

## Scope

This route is responsible for:

- top-k graph selection strategies;
- abnormal/TNS-aware ranking;
- relation-wise K allocation;
- bandit-style K search;
- D1-attached K strategy evaluation.

## Candidate-Pool Variant

`run_routeD1_kattach_candidate_pool.py` is the route-governed implementation of the
original intended K-line idea:

- keep strict D1 protocol and D1 asset base;
- use larger candidate pools (`candidate_topM > final_k`);
- re-select final top-k edges by reliability score;
- optionally search relation-wise `k` using bandit warmup/full-train flow.

This is distinct from the earlier fixed-top20 rerank-only K2 variant.

## Isolation Rule

Route K code may import shared/core graph logic, but it must not change shared defaults just to support K-line experiments.

If shared graph code must be extended:

- the old default behavior must remain unchanged;
- new behavior must be explicit and configurable.

## Shared Helper Note

`graph/graph_pipeline.py` includes the TNS-heavy helper functions required by
`build_routek_d1main_rns_topk_graph_frames`. These helpers are compatibility
plumbing for the existing D1-attached Route K path, not a change to shared D1
defaults. Cache reads are validated against `phi_days` and LogicAE candidate
edge count to avoid reusing stale probe caches.
