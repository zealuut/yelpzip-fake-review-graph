# Baseline Comparison

This directory contains an isolated baseline comparison suite for the current paper protocol only.

Protocol:
- `current top-k graph`
- `top_k = 20`
- balanced users
- no graph reweighting
- same split / labels / feature source as `routeD_tns_guided_logic_egat_20260504_200855/D1_EGAT_Base_LogicAE_CB`

Models:
- `GAT_CurrentTopK_Base_LogicAE_CB`
- `GraphSAGE_CurrentTopK_Base_LogicAE_CB`
- `RGCN_CurrentTopK_Base_LogicAE_CB`

This suite does **not** use senior graph variants, TNS, abnormal compression, NodeGAT, self-gates, mutual logic features, or any other new module. It is only for baseline backbone comparison under the current paper protocol.
