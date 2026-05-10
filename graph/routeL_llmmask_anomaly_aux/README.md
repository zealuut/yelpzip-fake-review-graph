# Route L isolated text-anomaly sandbox

Isolated Route L sandbox for end-to-end text anomaly head experiments under the D1 graph protocol.

Scope:
- no changes to main experiment code
- all Route L model/training/export logic stays in this directory
- D1 assets are reused only as protocol/artifact inputs
- each experiment retrains its own text head, re-exports review/user vectors, rebuilds graph inputs, then runs the same EGAT protocol
