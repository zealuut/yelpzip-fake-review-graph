from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from graph.relation_model import EdgePack, _segment_softmax


def build_egatpp_components(torch: Any):
    nn = torch.nn

    class MultiHeadEdgeAttentionLayer(nn.Module):
        def __init__(
            self,
            hidden_dim: int,
            edge_dim: int,
            heads: int = 4,
            dropout: float = 0.2,
            gatv2: bool = False,
            edge_gate: bool = False,
        ) -> None:
            super().__init__()
            self.hidden_dim = hidden_dim
            self.heads = max(1, int(heads))
            if hidden_dim % self.heads != 0:
                raise ValueError("hidden_dim must be divisible by heads")
            self.head_dim = hidden_dim // self.heads
            self.gatv2 = bool(gatv2)
            self.edge_gate = bool(edge_gate)

            self.src_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.dst_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.edge_proj = nn.Linear(edge_dim, hidden_dim, bias=False)
            self.msg_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.out_proj = nn.Linear(hidden_dim, hidden_dim)
            self.dropout = nn.Dropout(dropout)
            self.norm = nn.LayerNorm(hidden_dim)
            self.activation = nn.LeakyReLU(0.2)

            if self.gatv2:
                self.gatv2_proj = nn.Linear(self.head_dim * 3, self.head_dim, bias=False)
                self.attn_vec = nn.Parameter(torch.empty(self.heads, self.head_dim))
                nn.init.xavier_uniform_(self.attn_vec)
            else:
                self.attn_src = nn.Parameter(torch.empty(self.heads, self.head_dim))
                self.attn_dst = nn.Parameter(torch.empty(self.heads, self.head_dim))
                self.attn_edge = nn.Parameter(torch.empty(self.heads, self.head_dim))
                nn.init.xavier_uniform_(self.attn_src)
                nn.init.xavier_uniform_(self.attn_dst)
                nn.init.xavier_uniform_(self.attn_edge)

            self.gate_mlp = (
                nn.Sequential(
                    nn.Linear(self.head_dim * 3, self.head_dim),
                    nn.ReLU(),
                    nn.Linear(self.head_dim, 1),
                )
                if self.edge_gate
                else None
            )

        def forward(self, node_repr: Any, edge_pack: EdgePack) -> Any:
            if edge_pack.src.size == 0:
                return torch.zeros_like(node_repr)

            src = torch.as_tensor(edge_pack.src, dtype=torch.long, device=node_repr.device)
            dst = torch.as_tensor(edge_pack.dst, dtype=torch.long, device=node_repr.device)
            edge_features = torch.as_tensor(edge_pack.edge_features, dtype=node_repr.dtype, device=node_repr.device)
            weights = torch.as_tensor(edge_pack.weight, dtype=node_repr.dtype, device=node_repr.device).clamp(min=1e-6)

            src_repr = self.src_proj(node_repr[src]).view(-1, self.heads, self.head_dim)
            dst_repr = self.dst_proj(node_repr[dst]).view(-1, self.heads, self.head_dim)
            edge_repr = self.edge_proj(edge_features).view(-1, self.heads, self.head_dim)

            if self.gatv2:
                attn_input = torch.cat([src_repr, dst_repr, edge_repr], dim=-1)
                logits = self.gatv2_proj(self.activation(attn_input))
                scores = (logits * self.attn_vec.unsqueeze(0)).sum(dim=-1)
            else:
                scores = (
                    (src_repr * self.attn_src.unsqueeze(0)).sum(dim=-1)
                    + (dst_repr * self.attn_dst.unsqueeze(0)).sum(dim=-1)
                    + (edge_repr * self.attn_edge.unsqueeze(0)).sum(dim=-1)
                )

            scores = scores + torch.log(weights).unsqueeze(-1)
            msg = self.msg_proj(node_repr[dst]).view(-1, self.heads, self.head_dim)

            if self.edge_gate and self.gate_mlp is not None:
                gate_in = torch.cat([src_repr, dst_repr, edge_repr], dim=-1)
                gate = torch.sigmoid(self.gate_mlp(gate_in)).squeeze(-1)
            else:
                gate = None

            out = torch.zeros(node_repr.shape[0], self.heads, self.head_dim, dtype=node_repr.dtype, device=node_repr.device)
            for h in range(self.heads):
                alpha = _segment_softmax(scores[:, h], src, node_repr.shape[0], torch)
                alpha = self.dropout(alpha)
                head_msg = msg[:, h, :] * alpha.unsqueeze(-1)
                if gate is not None:
                    head_msg = head_msg * gate[:, h].unsqueeze(-1)
                out[:, h, :].index_add_(0, src, head_msg)

            merged = out.reshape(node_repr.shape[0], self.hidden_dim)
            merged = self.out_proj(merged)
            merged = self.dropout(merged)
            return self.norm(node_repr + merged)

    class EGATPPClassifier(nn.Module):
        def __init__(
            self,
            input_dim: int,
            relations: Sequence[str],
            edge_dim_map: dict[str, int],
            hidden_dim: int = 128,
            heads: int = 4,
            dropout: float = 0.2,
            num_layers: int = 2,
            gatv2: bool = False,
            edge_gate: bool = False,
        ) -> None:
            super().__init__()
            self.relations = list(relations)
            self.hidden_dim = int(hidden_dim)
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            self.self_proj = nn.Linear(input_dim, hidden_dim)
            self.layers = nn.ModuleList(
                [
                    nn.ModuleDict(
                        {
                            relation: MultiHeadEdgeAttentionLayer(
                                hidden_dim=hidden_dim,
                                edge_dim=max(1, edge_dim_map.get(relation, 1)),
                                heads=heads,
                                dropout=dropout,
                                gatv2=gatv2,
                                edge_gate=edge_gate,
                            )
                            for relation in self.relations
                        }
                    )
                    for _ in range(max(1, int(num_layers)))
                ]
            )
            self.relation_gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Sequential(
                nn.LayerNorm(hidden_dim * 2),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

        def _fuse_relations(self, current: Any, relation_outputs: dict[str, Any]) -> Any:
            outputs = []
            logits = []
            for relation in self.relations:
                rel_out = relation_outputs[relation]
                outputs.append(rel_out.unsqueeze(1))
                gate_in = torch.cat([current, rel_out], dim=-1)
                logits.append(self.relation_gate(gate_in).unsqueeze(1))
            stacked = torch.cat(outputs, dim=1)
            gate_logits = torch.cat(logits, dim=1).squeeze(-1)
            gate = torch.softmax(gate_logits, dim=1).unsqueeze(-1)
            fused = (stacked * gate).sum(dim=1)
            return fused

        def forward(self, node_features: Any, edge_packs: dict[str, EdgePack]) -> Any:
            current = self.input_proj(node_features)
            self_repr = self.self_proj(node_features)
            for layer in self.layers:
                rel_outputs = {relation: layer[relation](current, edge_packs[relation]) for relation in self.relations}
                fused = self._fuse_relations(current, rel_outputs)
                current = current + self.dropout(fused)
            logits = self.classifier(torch.cat([self_repr, current], dim=-1)).squeeze(-1)
            return logits

    return EGATPPClassifier
