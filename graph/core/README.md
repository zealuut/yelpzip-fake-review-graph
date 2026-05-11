# Core Layer Candidate

This directory is introduced as the stable shared layer target.

It is intentionally lightweight in this governance branch.

Rules:

- no route-specific default behavior;
- backward-compatible flags only;
- historical D1 defaults must remain the default path;
- every core-affecting change must be checked by `graph/governance/d1_smoke_test.py`.
