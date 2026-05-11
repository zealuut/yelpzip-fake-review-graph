# Route Isolation

This directory is the target structure for route-specific code isolation.

Expected pattern:

- `routeD_d1_main/`
- `routeK_topk/`
- `routeL_text_anomaly/`
- `routeTextHeads/`
- `baseline_comparison/`

Each route should keep its own:

- `configs/`
- `scripts/`
- `src/`
- `README.md`

Shared logic belongs in `graph/core/` only if it is backward-compatible and default-safe.
