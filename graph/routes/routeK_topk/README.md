# Route K Top-K Strategy

This route owns K-line graph strategy experiments.

Current contents include the D1 strict-attachment K2/K4 runner:

- `scripts/run_routeD1_kattach_k2k4_ce6a9d6.py`

## Scope

This route is responsible for:

- top-k graph selection strategies;
- abnormal/TNS-aware ranking;
- relation-wise K allocation;
- bandit-style K search;
- D1-attached K strategy evaluation.

## Isolation Rule

Route K code may import shared/core graph logic, but it must not change shared defaults just to support K-line experiments.

If shared graph code must be extended:

- the old default behavior must remain unchanged;
- new behavior must be explicit and configurable.
