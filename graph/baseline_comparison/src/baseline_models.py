from __future__ import annotations

from dataclasses import dataclass

import torch


def segment_softmax(scores: torch.Tensor, src_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    if scores.ndim == 1:
        max_per_node = torch.full((num_nodes,), -1e9, dtype=scores.dtype, device=scores.device)
        max_per_node.scatter_reduce_(0, src_index, scores, reduce="amax", include_self=True)
        exp_scores = torch.exp(scores - max_per_node[src_index])
        denom = torch.zeros((num_nodes,), dtype=scores.dtype, device=scores.device)
        denom.index_add_(0, src_index, exp_scores)
        return exp_scores / (denom[src_index] + 1e-9)

    pieces = []
    for head in range(scores.shape[1]):
        pieces.append(segment_softmax(scores[:, head], src_index, num_nodes).unsqueeze(-1))
    return torch.cat(pieces, dim=-1)


@dataclass
class GraphBatch:
    x: torch.Tensor
    src: torch.Tensor
    dst: torch.Tensor
    relation_id: torch.Tensor | None = None
    num_nodes: int = 0


class GATCurrentTopK(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads for GAT.")
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.lin = torch.nn.Linear(input_dim, hidden_dim, bias=False)
        self.att_src = torch.nn.Parameter(torch.empty(heads, self.head_dim))
        self.att_dst = torch.nn.Parameter(torch.empty(heads, self.head_dim))
        self.self_lin = torch.nn.Linear(input_dim, hidden_dim, bias=True)
        self.classifier = torch.nn.Linear(hidden_dim, 1)
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.ELU()
        torch.nn.init.xavier_uniform_(self.lin.weight)
        torch.nn.init.xavier_uniform_(self.self_lin.weight)
        torch.nn.init.zeros_(self.self_lin.bias)
        torch.nn.init.xavier_uniform_(self.att_src)
        torch.nn.init.xavier_uniform_(self.att_dst)

    def forward(self, batch: GraphBatch) -> torch.Tensor:
        x = batch.x
        src = batch.src
        dst = batch.dst
        num_nodes = batch.num_nodes
        h = self.lin(self.dropout(x)).view(num_nodes, self.heads, self.head_dim)
        src_h = h[src]
        dst_h = h[dst]
        scores = torch.nn.functional.leaky_relu(
            (src_h * self.att_src.unsqueeze(0)).sum(-1) + (dst_h * self.att_dst.unsqueeze(0)).sum(-1),
            negative_slope=0.2,
        )
        alpha = self.dropout(segment_softmax(scores, src, num_nodes))
        aggregated_heads = []
        for head_idx in range(self.heads):
            msg = dst_h[:, head_idx, :] * alpha[:, head_idx].unsqueeze(-1)
            agg = torch.zeros((num_nodes, self.head_dim), dtype=x.dtype, device=x.device)
            agg.index_add_(0, src, msg)
            aggregated_heads.append(agg)
        aggregated = torch.cat(aggregated_heads, dim=-1)
        hidden = self.activation(self.self_lin(x) + aggregated)
        hidden = self.dropout(hidden)
        return self.classifier(hidden).squeeze(-1)


class GraphSAGECurrentTopK(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.self_lin = torch.nn.Linear(input_dim, hidden_dim)
        self.neigh_lin = torch.nn.Linear(input_dim, hidden_dim, bias=False)
        self.classifier = torch.nn.Linear(hidden_dim, 1)
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.ReLU()

    def forward(self, batch: GraphBatch) -> torch.Tensor:
        x = batch.x
        src = batch.src
        dst = batch.dst
        num_nodes = batch.num_nodes
        msg = x[dst]
        agg = torch.zeros((num_nodes, x.shape[1]), dtype=x.dtype, device=x.device)
        deg = torch.zeros((num_nodes, 1), dtype=x.dtype, device=x.device)
        agg.index_add_(0, src, msg)
        deg.index_add_(0, src, torch.ones((src.shape[0], 1), dtype=x.dtype, device=x.device))
        agg = agg / deg.clamp(min=1.0)
        hidden = self.activation(self.self_lin(x) + self.neigh_lin(agg))
        hidden = self.dropout(hidden)
        return self.classifier(hidden).squeeze(-1)


class RGCNCurrentTopK(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_relations: int, num_bases: int, dropout: float) -> None:
        super().__init__()
        self.num_relations = num_relations
        self.num_bases = min(max(1, num_bases), num_relations)
        self.bases = torch.nn.Parameter(torch.empty(self.num_bases, input_dim, hidden_dim))
        self.coefficients = torch.nn.Parameter(torch.empty(num_relations, self.num_bases))
        self.self_lin = torch.nn.Linear(input_dim, hidden_dim)
        self.classifier = torch.nn.Linear(hidden_dim, 1)
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.ReLU()
        torch.nn.init.xavier_uniform_(self.bases)
        torch.nn.init.xavier_uniform_(self.coefficients)

    def forward(self, batch: GraphBatch) -> torch.Tensor:
        if batch.relation_id is None:
            raise ValueError("R-GCN requires relation_id in GraphBatch.")
        x = batch.x
        src = batch.src
        dst = batch.dst
        rel = batch.relation_id
        num_nodes = batch.num_nodes
        hidden = self.self_lin(x)
        relation_weights = torch.einsum("rb,bij->rij", self.coefficients, self.bases)
        for relation_idx in range(self.num_relations):
            mask = rel == relation_idx
            if int(mask.sum()) == 0:
                continue
            src_r = src[mask]
            dst_r = dst[mask]
            w_r = relation_weights[relation_idx]
            msg = x[dst_r] @ w_r
            agg = torch.zeros((num_nodes, w_r.shape[1]), dtype=x.dtype, device=x.device)
            deg = torch.zeros((num_nodes, 1), dtype=x.dtype, device=x.device)
            agg.index_add_(0, src_r, msg)
            deg.index_add_(0, src_r, torch.ones((src_r.shape[0], 1), dtype=x.dtype, device=x.device))
            hidden = hidden + agg / deg.clamp(min=1.0)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        return self.classifier(hidden).squeeze(-1)
