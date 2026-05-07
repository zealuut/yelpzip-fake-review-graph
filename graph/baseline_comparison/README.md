# Baseline Comparison

This directory contains isolated graph baseline suites that do not overwrite the main `graph/outputs/` experiment tree.

Available protocols:

1. `current top-k graph`
- `top_k = 20`
- balanced users
- no graph reweighting
- same split / labels / feature source as `routeD_tns_guided_logic_egat_20260504_200855/D1_EGAT_Base_LogicAE_CB`
- models:
  - `GAT_CurrentTopK_Base_LogicAE_CB`
  - `GraphSAGE_CurrentTopK_Base_LogicAE_CB`
  - `RGCN_CurrentTopK_Base_LogicAE_CB`

2. `FullBase_UPU_UTU_USU graph baseline`
- full behavior graph using only `UPU`, `UTU`, `USU`
- same user set / same labels / same split as the current D1 reference row
- clean numeric user features from `logic_vectors/user_summary.csv`
- no `LogicAE_CB`, no `CB`, no `TextSim`, no `TNS`, no abnormal compression, no LLM mask, no self gate
- models:
  - `GraphSAGE_FullBase`
  - `RGCN_FullBase`
  - `GAT_FullBase_Light`
  - optional `GCN_FullBase`

The `full_base` suite may internally reuse the same cached full behavior graph construction that other project code calls `senior`, but this comparison should be reported as:

`FullBase_UPU_UTU_USU graph baseline`

It is a strong full behavior graph baseline, not a reproduction of any complete prior model stack.
