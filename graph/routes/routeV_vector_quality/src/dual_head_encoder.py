"""V2: Dual-objective training with decoupled vector head.

Wraps the existing LLMMaskedLogicEncoder and adds a separate vector projection
head that is trained with a user-level contrastive objective. The original
review_vector path remains for backward compatibility; the new graph_vector
path produces vectors optimized for downstream graph separability.

This does NOT modify the shared review_models.py. It wraps the encoder and
adds a parallel head.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class DualHeadOutput:
    review_vector: torch.Tensor
    review_logit: torch.Tensor
    text_vector: torch.Tensor
    gate: torch.Tensor
    graph_vector: torch.Tensor
    abnormal_aux_logit: torch.Tensor | None = None


class GraphVectorHead(nn.Module):
    """Separate projection head for graph-optimized vectors.

    Takes the same fusion representation as the original abnormal_vector_mlp
    but projects through an independent MLP with different capacity/objective.
    """

    def __init__(
        self,
        fusion_dim: int,
        hidden_dim: int,
        vector_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vector_dim),
        )
        self.norm = nn.LayerNorm(vector_dim)

    def forward(self, fusion: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(fusion))


class DualHeadWrapper(nn.Module):
    """Wraps an existing LLMMaskedLogicEncoder with a parallel graph vector head.

    The wrapper intercepts the fusion tensor (before abnormal_vector_mlp) and
    feeds it to both the original path and the new graph_vector_head.

    Usage:
        base_encoder = build_review_model(...)  # existing D1 encoder
        dual = DualHeadWrapper(base_encoder, vector_dim=256)
        # Training: use dual.forward() which returns DualHeadOutput
        # The graph_vector field is what gets aggregated to user level
    """

    def __init__(
        self,
        base_encoder: nn.Module,
        vector_dim: int = 256,
        graph_hidden_dim: int | None = None,
        dropout: float = 0.1,
        detach_fusion: bool = False,
    ) -> None:
        super().__init__()
        self.base = base_encoder
        self.detach_fusion = bool(detach_fusion)

        # Infer fusion_dim from the base encoder's abnormal_vector_mlp input
        fusion_dim = self.base.abnormal_vector_mlp[0].in_features
        if graph_hidden_dim is None:
            graph_hidden_dim = self.base.hidden_size

        self.graph_head = GraphVectorHead(
            fusion_dim=fusion_dim,
            hidden_dim=graph_hidden_dim,
            vector_dim=vector_dim,
            dropout=dropout,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        abnormal_token_mask: torch.Tensor,
        numeric_features: torch.Tensor,
    ) -> DualHeadOutput:
        # Replicate the base encoder forward but intercept fusion
        primary_out = self.base.primary(input_ids=input_ids, attention_mask=attention_mask)
        token_states = primary_out.last_hidden_state
        cls_state = getattr(primary_out, "pooler_output", None)
        if cls_state is None:
            cls_state = token_states[:, 0]

        soft_mask = abnormal_token_mask.float() * attention_mask.float()
        fallback_rows = soft_mask.sum(dim=1, keepdim=True) <= 0
        if fallback_rows.any():
            soft_mask = soft_mask.clone()
            soft_mask[fallback_rows.squeeze(-1)] = attention_mask[fallback_rows.squeeze(-1)].float()

        mask_bool = soft_mask > 0
        masked_states = token_states * soft_mask.unsqueeze(-1)

        if self.base.disable_logic_bilstm:
            logic_query = self.base.logic_mean_proj(self.base._masked_mean(token_states, mask_bool))
        else:
            logic_states, _ = self.base.logic_bilstm(masked_states)
            if self.base.logic_pooling == "mean":
                pooled_logic = self.base._masked_mean(logic_states, mask_bool)
            else:
                pooled_logic = self.base.logic_pool(logic_states, mask_bool)
            logic_query = self.base.logic_proj(pooled_logic)

        if self.base.disable_cross_attention:
            cross_context = torch.zeros_like(logic_query)
        else:
            cross_context = self.base.cross_attn(
                logic_query.unsqueeze(1),
                token_states,
                token_states,
                padding_mask=attention_mask == 0,
            ).squeeze(1)

        numeric_proj = self.base.numeric_proj(numeric_features)
        gate_stats = torch.cat([
            soft_mask.mean(dim=1, keepdim=True),
            soft_mask.max(dim=1, keepdim=True).values,
            attention_mask.float().mean(dim=1, keepdim=True),
        ], dim=-1)
        gate_inputs = torch.cat([numeric_features, gate_stats], dim=-1)

        if self.base.gate_mode == "no_gate":
            gate = torch.ones(cross_context.size(0), 1, device=cross_context.device)
        elif self.base.gate_mode == "fixed_half":
            gate = torch.full((cross_context.size(0), 1), 0.5, device=cross_context.device)
        elif self.base.gate_mode == "numeric_only":
            gate = torch.sigmoid(self.base.gate_mlp_numeric(numeric_features))
        elif self.base.gate_mode == "text_only":
            gate = torch.sigmoid(self.base.gate_mlp_text(gate_stats))
        else:
            gate = torch.sigmoid(self.base.gate_mlp(gate_inputs))
        gated_cross = gate * cross_context

        fusion_parts = [cls_state, logic_query, gated_cross, numeric_proj]
        if self.base.secondary is not None and self.base.secondary_proj is not None:
            secondary_out = self.base.secondary(input_ids=input_ids, attention_mask=attention_mask)
            secondary_cls = getattr(secondary_out, "pooler_output", None)
            if secondary_cls is None:
                secondary_cls = secondary_out.last_hidden_state[:, 0]
            fusion_parts.append(self.base.secondary_proj(secondary_cls))

        fusion = torch.cat(fusion_parts, dim=-1)

        # Original path
        review_vector = self.base.abnormal_vector_mlp(fusion)
        review_vector = self.base.vector_norm(review_vector)
        review_logit = self.base.review_classifier(review_vector).squeeze(-1)
        text_vector = self.base.text_vector_mlp(cls_state)

        # New graph vector path. RouteV strict V2 uses detach_fusion=False so
        # the weighted graph loss can train the shared backbone as specified.
        graph_input = fusion.detach() if self.detach_fusion else fusion
        graph_vector = self.graph_head(graph_input)

        abnormal_aux_logit = None
        if self.base.abnormal_aux_head is not None:
            aux_input = self.base._build_aux_input(
                logic_query=logic_query,
                cross_context=cross_context,
                gated_cross=gated_cross,
                review_vector=review_vector,
            )
            abnormal_aux_logit = self.base.abnormal_aux_head(aux_input).squeeze(-1)

        return DualHeadOutput(
            review_vector=review_vector,
            review_logit=review_logit,
            text_vector=text_vector,
            gate=gate.squeeze(-1),
            graph_vector=graph_vector,
            abnormal_aux_logit=abnormal_aux_logit,
        )
