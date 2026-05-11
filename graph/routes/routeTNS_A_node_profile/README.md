Route TNS-A: TNS Node Profile as User Feature

This route keeps the D1 graph protocol fixed:

- `GRAPH_MODE=current`
- `EDGE_SET=Base_LogicAE_CB`
- relations: `UPU`, `UTU`, `USU`, `LogicAE_CB`
- `MODEL_BACKBONE=current_egat`
- `RELATION_MODEL=edge_aware_gat`

It does not add TNS user-user edges, group nodes, or hypergraph structure.
It only computes user-level temporal suspicious group profile features and
appends them to the original D1 self feature matrix.

Route-local files only. Core/default behavior must remain unchanged.
