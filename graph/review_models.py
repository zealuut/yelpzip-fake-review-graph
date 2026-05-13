from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

try:
    from transformers import AutoModel
except Exception:  # pragma: no cover - resolved on the server/runtime
    AutoModel = None


@dataclass
class ReviewEncoderOutput:
    review_vector: torch.Tensor
    review_logit: torch.Tensor
    text_vector: torch.Tensor
    gate: torch.Tensor
    abnormal_aux_logit: torch.Tensor | None = None


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

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
        pooled = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)
        return pooled


class LLMMaskedLogicEncoder(nn.Module):
    def __init__(
        self,
        primary_model_name_or_path: str,
        numeric_feature_dim: int,
        vector_dim: int = 256,
        secondary_model_name_or_path: str | None = None,
        dropout: float = 0.1,
        freeze_primary: bool = False,
        freeze_secondary: bool = False,
        abnormal_aux_enabled: bool = False,
        abnormal_aux_position: str = "logic_gated_cross",
        disable_cross_attention: bool = False,
        disable_logic_bilstm: bool = False,
        logic_pooling: str = "attention",
        gate_mode: str = "learned",
    ) -> None:
        super().__init__()
        if AutoModel is None:
            raise ImportError("transformers is required for LLMMaskedLogicEncoder")

        abnormal_aux_position = str(abnormal_aux_position or "logic_gated_cross").lower()
        if abnormal_aux_position not in {
            "logic_query",
            "cross_context",
            "gated_cross",
            "logic_gated_cross",
            "logic_cross_context",
            "final_review_vector",
        }:
            raise ValueError(f"Unsupported abnormal_aux_position: {abnormal_aux_position}")
        logic_pooling = str(logic_pooling or "attention").lower()
        if logic_pooling not in {"attention", "mean"}:
            raise ValueError(f"Unsupported logic_pooling: {logic_pooling}")
        gate_mode = str(gate_mode or "learned").lower()
        if gate_mode not in {"learned", "no_gate", "fixed_half", "numeric_only", "text_only"}:
            raise ValueError(f"Unsupported gate_mode: {gate_mode}")

        self.primary = AutoModel.from_pretrained(primary_model_name_or_path)
        self.hidden_size = int(self.primary.config.hidden_size)
        self.logic_hidden = max(self.hidden_size // 2, 128)
        self.secondary = None
        self.secondary_proj = None
        self.abnormal_aux_enabled = bool(abnormal_aux_enabled)
        self.abnormal_aux_position = abnormal_aux_position
        self.disable_cross_attention = bool(disable_cross_attention)
        self.disable_logic_bilstm = bool(disable_logic_bilstm)
        self.logic_pooling = logic_pooling
        self.gate_mode = gate_mode

        if secondary_model_name_or_path:
            self.secondary = AutoModel.from_pretrained(secondary_model_name_or_path)
            self.secondary_proj = nn.Sequential(
                nn.Linear(int(self.secondary.config.hidden_size), self.hidden_size),
                nn.Tanh(),
                nn.Dropout(dropout),
            )

        if freeze_primary:
            for parameter in self.primary.parameters():
                parameter.requires_grad = False
        if self.secondary is not None and freeze_secondary:
            for parameter in self.secondary.parameters():
                parameter.requires_grad = False

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
        gate_input_dim = numeric_feature_dim + 3
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_input_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        fusion_dim = self.hidden_size * 3 + self.hidden_size // 2
        if self.secondary_proj is not None:
            fusion_dim += self.hidden_size
        self.abnormal_vector_mlp = nn.Sequential(
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

        # Keep the default D1 initialization stream identical: create ablation-only
        # modules only when the corresponding ablation is active.
        self.logic_mean_proj = None
        if self.disable_logic_bilstm:
            self.logic_mean_proj = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        self.gate_mlp_numeric = None
        if self.gate_mode == "numeric_only":
            self.gate_mlp_numeric = nn.Sequential(
                nn.Linear(numeric_feature_dim, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1),
            )
        self.gate_mlp_text = None
        if self.gate_mode == "text_only":
            self.gate_mlp_text = nn.Sequential(
                nn.Linear(3, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1),
            )

        self.abnormal_aux_head = None
        if self.abnormal_aux_enabled:
            aux_input_dim = self._resolve_aux_input_dim(vector_dim=vector_dim)
            self.abnormal_aux_head = nn.Sequential(
                nn.Linear(aux_input_dim, self.hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_size, 1),
            )

    def _resolve_aux_input_dim(self, vector_dim: int) -> int:
        if self.abnormal_aux_position in {"logic_query", "cross_context", "gated_cross"}:
            return self.hidden_size
        if self.abnormal_aux_position in {"logic_gated_cross", "logic_cross_context"}:
            return self.hidden_size * 2
        if self.abnormal_aux_position == "final_review_vector":
            return vector_dim
        raise ValueError(f"Unsupported abnormal_aux_position: {self.abnormal_aux_position}")

    @staticmethod
    def _masked_mean(hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.float().unsqueeze(-1)
        return (hidden_states * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)

    def _build_aux_input(
        self,
        *,
        logic_query: torch.Tensor,
        cross_context: torch.Tensor,
        gated_cross: torch.Tensor,
        review_vector: torch.Tensor,
    ) -> torch.Tensor:
        if self.abnormal_aux_position == "logic_query":
            return logic_query
        if self.abnormal_aux_position == "cross_context":
            return cross_context
        if self.abnormal_aux_position == "gated_cross":
            return gated_cross
        if self.abnormal_aux_position == "logic_gated_cross":
            return torch.cat([logic_query, gated_cross], dim=-1)
        if self.abnormal_aux_position == "logic_cross_context":
            return torch.cat([logic_query, cross_context], dim=-1)
        if self.abnormal_aux_position == "final_review_vector":
            return review_vector
        raise ValueError(f"Unsupported abnormal_aux_position: {self.abnormal_aux_position}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        abnormal_token_mask: torch.Tensor,
        numeric_features: torch.Tensor,
    ) -> ReviewEncoderOutput:
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

        mask_bool = soft_mask > 0
        masked_states = token_states * soft_mask.unsqueeze(-1)
        if self.disable_logic_bilstm:
            logic_query = self.logic_mean_proj(self._masked_mean(token_states, mask_bool))
        else:
            logic_states, _ = self.logic_bilstm(masked_states)
            if self.logic_pooling == "mean":
                pooled_logic = self._masked_mean(logic_states, mask_bool)
            else:
                pooled_logic = self.logic_pool(logic_states, mask_bool)
            logic_query = self.logic_proj(pooled_logic)
        if self.disable_cross_attention:
            cross_context = torch.zeros_like(logic_query)
        else:
            cross_context = self.cross_attn(
                logic_query.unsqueeze(1),
                token_states,
                token_states,
                padding_mask=attention_mask == 0,
            ).squeeze(1)

        numeric_proj = self.numeric_proj(numeric_features)
        gate_stats = torch.cat(
            [
                soft_mask.mean(dim=1, keepdim=True),
                soft_mask.max(dim=1, keepdim=True).values,
                attention_mask.float().mean(dim=1, keepdim=True),
            ],
            dim=-1,
        )
        gate_inputs = torch.cat(
            [
                numeric_features,
                gate_stats,
            ],
            dim=-1,
        )
        if self.gate_mode == "no_gate":
            gate = torch.ones(cross_context.size(0), 1, device=cross_context.device, dtype=cross_context.dtype)
        elif self.gate_mode == "fixed_half":
            gate = torch.full((cross_context.size(0), 1), 0.5, device=cross_context.device, dtype=cross_context.dtype)
        elif self.gate_mode == "numeric_only":
            gate = torch.sigmoid(self.gate_mlp_numeric(numeric_features))
        elif self.gate_mode == "text_only":
            gate = torch.sigmoid(self.gate_mlp_text(gate_stats))
        else:
            gate = torch.sigmoid(self.gate_mlp(gate_inputs))
        gated_cross = gate * cross_context

        fusion_parts = [cls_state, logic_query, gated_cross, numeric_proj]
        if self.secondary is not None and self.secondary_proj is not None:
            secondary_out = self.secondary(input_ids=input_ids, attention_mask=attention_mask)
            secondary_cls = getattr(secondary_out, "pooler_output", None)
            if secondary_cls is None:
                secondary_cls = secondary_out.last_hidden_state[:, 0]
            fusion_parts.append(self.secondary_proj(secondary_cls))

        review_vector = self.abnormal_vector_mlp(torch.cat(fusion_parts, dim=-1))
        review_vector = self.vector_norm(review_vector)
        review_logit = self.review_classifier(review_vector).squeeze(-1)
        text_vector = self.text_vector_mlp(cls_state)
        abnormal_aux_logit = None
        if self.abnormal_aux_head is not None:
            aux_input = self._build_aux_input(
                logic_query=logic_query,
                cross_context=cross_context,
                gated_cross=gated_cross,
                review_vector=review_vector,
            )
            abnormal_aux_logit = self.abnormal_aux_head(aux_input).squeeze(-1)
        return ReviewEncoderOutput(
            review_vector=review_vector,
            review_logit=review_logit,
            text_vector=text_vector,
            gate=gate.squeeze(-1),
            abnormal_aux_logit=abnormal_aux_logit,
        )


class MockReviewEncoder(nn.Module):
    def __init__(self, numeric_feature_dim: int, vector_dim: int = 128, vocab_size: int = 4096) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, vector_dim)
        self.numeric_proj = nn.Sequential(
            nn.Linear(numeric_feature_dim, vector_dim),
            nn.ReLU(),
        )
        self.review_classifier = nn.Linear(vector_dim, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        abnormal_token_mask: torch.Tensor,
        numeric_features: torch.Tensor,
    ) -> ReviewEncoderOutput:
        embedded = self.embedding(input_ids % self.embedding.num_embeddings)
        token_mask = attention_mask.unsqueeze(-1).float()
        pooled = (embedded * token_mask).sum(dim=1) / token_mask.sum(dim=1).clamp(min=1.0)
        pooled = pooled + self.numeric_proj(numeric_features)
        review_logit = self.review_classifier(pooled).squeeze(-1)
        return ReviewEncoderOutput(
            review_vector=pooled,
            review_logit=review_logit,
            text_vector=pooled,
            gate=torch.ones(pooled.size(0), device=pooled.device),
        )
