# routeTNSGD_GroupFirst_NodeFeature

Group-discovery-first route inspired by TNSGD. This is not a full reproduction of the paper. The goal is a faithful first-step group discovery layer that produces interpretable group and user membership artifacts before any graph model consumes them.

Default setting:

- `phi_days = 5`
- `delta_I = 0.5`
- `merge_jaccard = 0.8`
- top suspicious reviewer strategy: `Top-30&Last-0` after a Top-300 temporal-neighbor-sequence pool

Main outputs:

- `tnsgd_groups.csv`
- `tnsgd_user_group_membership.csv`
- `tnsgd_user_node_features.parquet`

Diagnostics:

- `tnsgd_group_label_diagnostics.csv` uses labels only after unsupervised discovery, to inspect group purity and normal-vs-abnormal grouping separation.
- `tnsgd_summary.json` records parameters, counts, and output paths.

Implementation notes:

- Discovery uses reviewer ISS from the existing behavior indicators `RD`, `AD`, `EXR`, `MRO`, and `ATR`.
- Co-review temporal events are built from same-business reviews within `phi_days`; selected seed users are high-ISS reviewers with long temporal neighbor sequences.
- Candidate groups are burst sessions in each selected seed user's temporal neighbor sequence.
- Candidate groups are merged when they share the same burst window and member Jaccard is at least `merge_jaccard`.
- Core members are raw members with `ISS >= delta_I`; raw-only members are retained in the membership table for later graph features.
- Group score `GSS` is the mean of interpretable approximations of `GRT`, `GS`, `GRD`, `GOR`, `GER`, and `GCAR` on the discovered group context.

Run:

```bash
bash graph/routes/routeTNSGD_GroupFirst_NodeFeature/scripts/run_route_tnsgd_group_first.sh
```

Optional grid:

```bash
python3 graph/routes/routeTNSGD_GroupFirst_NodeFeature/scripts/run_route_tnsgd_group_first.py \
  --config graph/routes/routeTNSGD_GroupFirst_NodeFeature/configs/tnsgd_group_first_phi5.yaml \
  --output_root graph/outputs/routeTNSGD_GroupFirst_NodeFeature_grid \
  --run_grid
```
