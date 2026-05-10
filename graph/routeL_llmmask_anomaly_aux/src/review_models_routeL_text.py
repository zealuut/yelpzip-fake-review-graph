from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

try:
    from transformers import AutoModel
except Exception:  # pragma: no cover
    AutoModel = None


@dataclass
class RouteLTextEvidenceOutput:
    review_vector: torch.Tensor
    review_logit: torch.Tensor
    evidence_logit: torch.Tensor
    text_vector: torch.Tensor
    gate: torch.Tensor
    token_evidence_scores: torch.Tensor


def _masked_mean(hidden_states: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return (hidden_states * weights.unsqueeze(-1)).sum(dim=1) / denom


def _content_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.bool().clone()
    if mask.shape[1] > 0:
        mask[:, 0] = False
    return mask


class RouteLTextEvidenceEncoder(nn.Module):
    def __init__(
        self,
        primary_model_name_or_path: str,
        vector_dim: int = 256,
        experiment_kind: str = "exp1_learned_token_evidence",
        extra_feature_dim: int = 0,
        topk_tokens: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if AutoModel is None:
            raise ImportError("transformers is required for RouteLTextEvidenceEncoder")
        self.primary = AutoModel.from_pretrained(primary_model_name_or_path)
        self.hidden_size = int(self.primary.config.hidden_size)
        self.vector_dim = int(vector_dim)
        self.experiment_kind = str(experiment_kind)
        self.topk_tokens = int(topk_tokens)
        self.extra_feature_dim = int(extra_feature_dim)

        self.evidence_scorer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, 1),
        )
        self.evidence_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 2, 1),
        )
        self.text_vector_mlp = nn.Sequential(
            nn.Linear(self.hidden_size, vector_dim),
            nn.Tanh(),
        )

        self.evidence_stats_dim = 4
        self.local_phrase_dim = 0
        self.style_dim = 0
        self.drift_dim = 0

        if self.experiment_kind == "exp3_local_phrase_cnn":
            cnn_channels = max(self.hidden_size // 4, 64)
            self.conv2 = nn.Conv1d(self.hidden_size, cnn_channels, kernel_size=2, padding=1)
            self.conv3 = nn.Conv1d(self.hidden_size, cnn_channels, kernel_size=3, padding=1)
            self.conv5 = nn.Conv1d(self.hidden_size, cnn_channels, kernel_size=5, padding=2)
            self.local_phrase_dim = cnn_channels * 3
        else:
            self.conv2 = None
            self.conv3 = None
            self.conv5 = None

        if self.experiment_kind == "exp4_psycholinguistic_style":
            if self.extra_feature_dim <= 0:
                raise ValueError("extra_feature_dim must be > 0 for exp4_psycholinguistic_style")
            self.style_mlp = nn.Sequential(
                nn.Linear(self.extra_feature_dim, self.hidden_size // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_size // 2, self.hidden_size // 2),
                nn.GELU(),
            )
            self.style_dim = self.hidden_size // 2
        else:
            self.style_mlp = None

        if self.experiment_kind == "exp5_semantic_drift":
            if self.extra_feature_dim <= 0:
                raise ValueError("extra_feature_dim must be > 0 for exp5_semantic_drift")
            self.drift_mlp = nn.Sequential(
                nn.Linear(self.extra_feature_dim, self.hidden_size // 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_size // 4, self.hidden_size // 4),
                nn.GELU(),
            )
            self.drift_dim = self.hidden_size // 4
        else:
            self.drift_mlp = None

        if self.experiment_kind in {"exp1_learned_token_evidence", "exp2_topk_token_evidence"}:
            fusion_dim = self.hidden_size * 2 + self.evidence_stats_dim
        elif self.experiment_kind == "exp3_local_phrase_cnn":
            fusion_dim = self.hidden_size * 2 + self.local_phrase_dim
        elif self.experiment_kind == "exp4_psycholinguistic_style":
            fusion_dim = self.hidden_size * 2 + self.style_dim
        elif self.experiment_kind == "exp5_semantic_drift":
            fusion_dim = self.hidden_size * 2 + self.drift_dim
        else:
            raise ValueError(f"Unsupported experiment kind: {self.experiment_kind}")

        self.review_fusion = nn.Sequential(
            nn.Linear(fusion_dim, self.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, vector_dim),
        )
        self.vector_norm = nn.LayerNorm(vector_dim)
        self.review_classifier = nn.Linear(vector_dim, 1)

    def _build_evidence_repr(
        self,
        token_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        content_mask = _content_mask(attention_mask)
        valid_mask = content_mask.float()
        evidence_logits = self.evidence_scorer(token_states).squeeze(-1)
        evidence_scores = torch.sigmoid(evidence_logits) * valid_mask
        evidence_sum = evidence_scores.sum(dim=1, keepdim=True)
        valid_count = valid_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        evidence_mean = evidence_sum / valid_count
        evidence_max = evidence_scores.max(dim=1, keepdim=True).values
        active_ratio = ((evidence_scores > 0.5).float() * valid_mask).sum(dim=1, keepdim=True) / valid_count
        evidence_stats = torch.cat([evidence_sum, evidence_mean, evidence_max, active_ratio], dim=-1)

        if self.experiment_kind == "exp2_topk_token_evidence":
            masked_scores = evidence_scores.masked_fill(~content_mask, -1e9)
            k = min(self.topk_tokens, token_states.shape[1])
            topk_scores, topk_indices = torch.topk(masked_scores, k=k, dim=1)
            gathered = torch.gather(
                token_states,
                1,
                topk_indices.unsqueeze(-1).expand(-1, -1, token_states.shape[-1]),
            )
            topk_valid = (topk_scores > -1e8).float().unsqueeze(-1)
            denom = topk_valid.sum(dim=1).clamp_min(1.0)
            z = (gathered * topk_valid).sum(dim=1) / denom
        else:
            z = _masked_mean(token_states, evidence_scores)
        return z, evidence_stats, evidence_scores

    def _local_phrase_vector(self, token_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.conv2 is None or self.conv3 is None or self.conv5 is None:
            raise RuntimeError("CNN branch requested but convolutions are not initialized")
        x = token_states * attention_mask.unsqueeze(-1).float()
        x = x.transpose(1, 2)
        convs = [
            F.gelu(self.conv2(x)),
            F.gelu(self.conv3(x)),
            F.gelu(self.conv5(x)),
        ]
        pooled = [torch.max(conv, dim=2).values for conv in convs]
        return torch.cat(pooled, dim=-1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        extra_features: torch.Tensor | None = None,
    ) -> RouteLTextEvidenceOutput:
        primary_out = self.primary(input_ids=input_ids, attention_mask=attention_mask)
        token_states = primary_out.last_hidden_state
        cls_state = getattr(primary_out, "pooler_output", None)
        if cls_state is None:
            cls_state = token_states[:, 0]

        evidence_repr, evidence_stats, evidence_scores = self._build_evidence_repr(token_states, attention_mask)
        evidence_logit = self.evidence_classifier(evidence_repr).squeeze(-1)

        fusion_parts = [cls_state, evidence_repr]
        if self.experiment_kind in {"exp1_learned_token_evidence", "exp2_topk_token_evidence"}:
            fusion_parts.append(evidence_stats)
        elif self.experiment_kind == "exp3_local_phrase_cnn":
            fusion_parts.append(self._local_phrase_vector(token_states, attention_mask))
        elif self.experiment_kind == "exp4_psycholinguistic_style":
            if extra_features is None:
                raise ValueError("extra_features required for exp4_psycholinguistic_style")
            fusion_parts.append(self.style_mlp(extra_features.float()))
        elif self.experiment_kind == "exp5_semantic_drift":
            if extra_features is None:
                raise ValueError("extra_features required for exp5_semantic_drift")
            fusion_parts.append(self.drift_mlp(extra_features.float()))

        review_vector = self.review_fusion(torch.cat(fusion_parts, dim=-1))
        review_vector = self.vector_norm(review_vector)
        review_logit = self.review_classifier(review_vector).squeeze(-1)
        text_vector = self.text_vector_mlp(cls_state)
        gate = evidence_scores.mean(dim=1)
        return RouteLTextEvidenceOutput(
            review_vector=review_vector,
            review_logit=review_logit,
            evidence_logit=evidence_logit,
            text_vector=text_vector,
            gate=gate,
            token_evidence_scores=evidence_scores,
        )
