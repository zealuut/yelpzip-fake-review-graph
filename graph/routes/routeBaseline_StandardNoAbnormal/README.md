# Route Baseline Standard No-Abnormal

This route owns clean comparison baselines for standard GNN-family models.

## Contract

- Code is frozen inside this route and must not import `graph.baseline_comparison`.
- Model folders are independently runnable through their own `run.sh`, `config.json`, and `train.py`.
- Data is read from fixed snapshot output directories only.
- Node features exclude abnormal text vectors and abnormal-head-derived feature names.
- Graph edges exclude LogicAE, TNS, abnormal reweighting, and abnormal gates.
- Models are topology-only standard GAT, GraphSAGE, and R-GCN baselines.

## Active Experiments

- `models/gat/`
- `models/graphsage/`
- `models/rgcn/`

Run all:

```bash
bash graph/routes/routeBaseline_StandardNoAbnormal/scripts/run_standard_no_abnormal_queue.sh
```

Run one model:

```bash
bash graph/routes/routeBaseline_StandardNoAbnormal/models/gat/run.sh /path/to/output_root
```
