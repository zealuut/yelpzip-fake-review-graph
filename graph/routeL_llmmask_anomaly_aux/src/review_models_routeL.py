from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

try:
    from transformers import AutoModel
except Exception:  # pragma: no cover
    AutoModel = None


@dataclass
class RouteLReviewEncoderOutput:
    review_vector: torch.Tensor
    review_logit: torch.Tensor
    aux_logit: torch.Tensor
    text_vector: torch.Tensor
    gate: torch.Tensor


class SafeCrossAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, target_len, hidden_size = query.shape
        source_len = key.shape[1]
        query = self.q_proj(query).view(batch_size, target_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(key).view(batch_size, source_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(value).view(batch_size, source_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask.unsqueeze(1).unsqueeze(1), -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0, posinf=0.0, neginf=0.0)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, value)
        context = context.transpose(1, 2).contiguous().view(batch_size, target_len, hidden_size)
        return self.out_proj(context)


class AttentionPool(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.scorer(hidden_states).squeeze(-1)
        scores = scores.masked_fill(~mask.bool(), -1e9)
        weights = F.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)


class RouteLLMMaskedLogicEncoder(nn.Module):
    def __init__(
        self,
        primary_model_name_or_path: str,
        numeric_feature_dim: int,
        vector_dim: int = 256,
        secondary_model_name_or_path: str | None = None,
        dropout: float = 0.1,
        freeze_primary: bool = False,
        freeze_secondary: bool = False,
        fusion_mode: str = "early",
    ) -> None:
        super().__init__()
        if AutoModel is None:
            raise ImportError("transformers is required for RouteLLMMaskedLogicEncoder")
        self.fusion_mode = str(fusion_mode).lower()
        self.primary = AutoModel.from_pretrained(primary_model_name_or_path)
        self.hidden_size = int(self.primary.config.hidden_size)
        self.logic_hidden = max(self.hidden_size // 2, 128)
        self.secondary = None
        self.secondary_proj = None
        if secondary_model_name_or_path:
            self.secondary = AutoModel.from_pretrained(secondary_model_name_or_path)
            self.secondary_proj = nn.Sequential(
                nn.Linear(int(self.secondary.config.hidden_size), self.hidden_size),
                nn.Tanh(),
                nn.Dropout(dropout),
            )
        if freeze_primary:
            for p in self.primary.parameters():
                p.requires_grad = False
        if self.secondary is not None and freeze_secondary:
            for p in self.secondary.parameters():
                p.requires_grad = False

        self.logic_bilstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.logic_hidden,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.2,
        )
        self.logic_proj = nn.Sequential(
            nn.Linear(self.logic_hidden * 2, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.logic_pool = AttentionPool(self.logic_hidden * 2)
        self.cross_attn = SafeCrossAttention(self.hidden_size, num_heads=8, dropout=dropout)
        self.numeric_proj = nn.Sequential(
            nn.Linear(numeric_feature_dim, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(numeric_feature_dim + 3, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        fusion_dim = self.hidden_size * 3 + self.hidden_size // 2
        if self.secondary_proj is not None:
            fusion_dim += self.hidden_size
        self.review_fusion = nn.Sequential(
            nn.Linear(fusion_dim, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, vector_dim),
        )
        self.vector_norm = nn.LayerNorm(vector_dim)
        self.text_vector_mlp = nn.Sequential(
            nn.Linear(self.hidden_size, vector_dim),
            nn.Tanh(),
        )
        self.review_classifier = nn.Linear(vector_dim, 1)
        self.aux_proj = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.aux_classifier = nn.Linear(self.hidden_size, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        abnormal_token_mask: torch.Tensor,
        numeric_features: torch.Tensor,
        warmup_active: bool = False,
    ):
        primary_out = self.primary(input_ids=input_ids, attention_mask=attention_mask)
        token_states = primary_out.last_hidden_state
        cls_state = getattr(primary_out, "pooler_output", None)
        if cls_state is None:
            cls_state = token_states[:, 0]

        soft_mask = abnormal_token_mask.float() * attention_mask.float()
        fallback_rows = soft_mask.sum(dim=1, keepdim=True) <= 0
        if fallback_rows.any():
            soft_mask = soft_mask.clone()
            soft_mask[fallback_rows.squeeze(-1)] = attention_mask[fallback_rows.squeeze(-1)].float()

        masked_states = token_states * soft_mask.unsqueeze(-1)
        logic_states, _ = self.logic_bilstm(masked_states)
        logic_query = self.logic_pool(logic_states, soft_mask > 0)
        logic_query = self.logic_proj(logic_query)
        cross_context = self.cross_attn(
            logic_query.unsqueeze(1), token_states, token_states, padding_mask=attention_mask == 0
        ).squeeze(1)

        gate_inputs = torch.cat(
            [
                numeric_features,
                soft_mask.mean(dim=1, keepdim=True),
                soft_mask.max(dim=1, keepdim=True).values,
                attention_mask.float().mean(dim=1, keepdim=True),
            ],
            dim=-1,
        )
        gate = torch.sigmoid(self.gate_mlp(gate_inputs))
        numeric_proj = self.numeric_proj(numeric_features)
        gated_cross = gate * cross_context

        aux_repr = self.aux_proj(torch.cat([logic_query, gated_cross], dim=-1))
        aux_logit = self.aux_classifier(aux_repr).squeeze(-1)

        if self.fusion_mode == "late" and warmup_active:
            logic_query_for_main = torch.zeros_like(logic_query)
            gated_cross_for_main = torch.zeros_like(gated_cross)
        else:
            logic_query_for_main = logic_query
            gated_cross_for_main = gated_cross

        fusion_parts = [cls_state, logic_query_for_main, gated_cross_for_main, numeric_proj]
        if self.secondary is not None and self.secondary_proj is not None:
            secondary_out = self.secondary(input_ids=input_ids, attention_mask=attention_mask)
            secondary_cls = getattr(secondary_out, "pooler_output", None)
            if secondary_cls is None:
                secondary_cls = secondary_out.last_hidden_state[:, 0]
            fusion_parts.append(self.secondary_proj(secondary_cls))

        review_vector = self.review_fusion(torch.cat(fusion_parts, dim=-1))
        review_vector = self.vector_norm(review_vector)
        review_logit = self.review_classifier(review_vector).squeeze(-1)
        text_vector = self.text_vector_mlp(cls_state)
        return RouteLReviewEncoderOutput(
            review_vector=review_vector,
            review_logit=review_logit,
            aux_logit=aux_logit,
            text_vector=text_vector,
            gate=gate.squeeze(-1),
        )

