# AGENTS.md

## Governance Branch Goal

This branch exists to preserve a recoverable D1 baseline while allowing small,
route-owned experiments to run on top of that baseline.

Primary active routes:

- `graph/routes/routeK_topk/`: K-line top-k and candidate-pool experiments.
- `graph/routes/routeG_egatpp/`: G-line EGAT++ graph-backbone experiments.

## D1 Baseline Protection

Treat the D1 base as a stable compatibility target.

The following shared files are protected and must not be changed for routine
route experiments:

- `graph/graph_pipeline.py`
- `graph/relation_model.py`
- `graph/scripts/route_runner.py`

If a shared-file change is unavoidable:

1. Keep historical D1/default behavior unchanged.
2. Add any new behavior behind an explicit opt-in argument or config value.
3. Document the reason in the owning route README or config.
4. Check `git diff -- <shared-file>` before and after the edit.
5. Run a smoke check for the affected route before launching a long queue.

Do not tune shared defaults to make a route result look better. Route-specific
logic belongs under the route directory unless there is a clear reusable reason
to promote it.

## Main Directory Boundary

`/home/xyz/HuChao (2)/Bert-TextClassification` is the noisy main working tree.
Governance-branch scripts may read fixed reference artifacts from it, including
the D1 protocol outputs and prepared base artifacts, but they should not write
logs, outputs, configs, or code into that directory.

Queue scripts in this branch must use branch-local paths:

- logs: `$ROOT_DIR/graph/logs`
- outputs: `$ROOT_DIR/graph/outputs`

Long governance queues should be launched through tmux with
`graph/routes/scripts/start_governance_kg_tmux.sh` so they survive SSH or
conversation disconnects.

## Queue Policy

Before running long queues, verify that the queue script is being launched from
this governance branch and that `ROOT_DIR` resolves to this branch root.
