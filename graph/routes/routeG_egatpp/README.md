# Route G EGAT++

This route owns graph-backbone enhancement experiments built on the strict D1 base.

## Scope

- `G0`: strict D1 graph backbone reproduction on D1 edge artifacts;
- `G1`: EGAT++ Lite
  - 2-layer EGAT
  - multi-head = 4
  - residual connection
  - layer norm
  - dropout
- `G2`: EGAT++ Full
  - 2-layer EGAT
  - multi-head = 4
  - GATv2-style attention
  - edge gate
  - residual
  - layer norm
  - dropout

## Isolation Rule

Route G code may import shared/core graph logic, but it must not change shared defaults
just to support G-line experiments.

Shared-layer extensions must remain backward compatible and keep historical defaults unchanged.
