"""V1: Vector separability regularization loss.

Adds a user-level contrastive regularizer during review encoder training.
Within each batch, reviews are grouped by user_id, mean-pooled to get
per-user vectors, then a contrastive loss pushes fake-user vectors apart
from real-user vectors in embedding space. Fake/real classes are read from
explicit user labels, not inferred from review labels.

This loss is computed per-batch as an approximation (not all users appear
in every batch), so it acts as a stochastic regularizer rather than an
exact objective.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class UserVectorSeparabilityLoss(nn.Module):
    """Batch-level user-vector contrastive regularizer.

    For each batch:
    1. Group review vectors by user_id
    2. Mean-pool to get user-level vectors
    3. Compute supervised contrastive loss (SupCon) on user vectors

    Requires at least 2 distinct users with different labels in the batch.
    Returns 0 gracefully if the batch doesn't meet this condition.
    """

    def __init__(self, temperature: float = 0.1, margin: float = 0.5) -> None:
        super().__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(
        self,
        review_vectors: torch.Tensor,
        user_ids: torch.Tensor,
        user_labels_by_review: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            review_vectors: (batch_size, vector_dim) review-level vectors
            user_ids: (batch_size,) integer user IDs for grouping
            user_labels_by_review: (batch_size,) explicit binary user labels
                from prepared.user_df.user_label, repeated for each review row.

        Returns:
            Scalar loss tensor.
        """
        device = review_vectors.device
        unique_users = user_ids.unique()

        if len(unique_users) < 4:
            return torch.tensor(0.0, device=device, requires_grad=True)

        user_vectors = []
        user_labels = []

        for uid in unique_users:
            mask = user_ids == uid
            user_vec = review_vectors[mask].mean(dim=0)
            labels_for_user = user_labels_by_review[mask]
            if torch.any(labels_for_user != labels_for_user[0]):
                raise ValueError("RouteV regularizer received inconsistent user_label values within a user batch.")
            user_label = labels_for_user[0]
            user_vectors.append(user_vec)
            user_labels.append(user_label)

        user_vectors = torch.stack(user_vectors)  # (n_users, dim)
        user_labels = torch.stack(user_labels)    # (n_users,)

        n_fake = (user_labels == 1).sum()
        n_real = (user_labels == 0).sum()
        if n_fake < 1 or n_real < 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Normalized vectors for cosine similarity
        user_vectors_norm = F.normalize(user_vectors, dim=1)
        sim_matrix = torch.mm(user_vectors_norm, user_vectors_norm.t()) / self.temperature

        n = len(user_labels)
        # Mask: same-class pairs are positives
        label_eq = user_labels.unsqueeze(0) == user_labels.unsqueeze(1)  # (n, n)
        # Exclude self
        self_mask = ~torch.eye(n, dtype=torch.bool, device=device)
        pos_mask = label_eq & self_mask
        neg_mask = ~label_eq & self_mask

        # SupCon loss: for each anchor, pull same-class closer, push different-class away
        loss = torch.tensor(0.0, device=device)
        valid_anchors = 0

        for i in range(n):
            pos_indices = pos_mask[i].nonzero(as_tuple=True)[0]
            neg_indices = neg_mask[i].nonzero(as_tuple=True)[0]
            if len(pos_indices) == 0 or len(neg_indices) == 0:
                continue

            # For each positive pair, contrast against all negatives
            pos_sims = sim_matrix[i, pos_indices]
            neg_sims = sim_matrix[i, neg_indices]

            # Log-sum-exp over negatives
            neg_logsumexp = torch.logsumexp(neg_sims, dim=0)
            # Loss for this anchor: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
            anchor_loss = (-pos_sims + neg_logsumexp).mean()
            loss = loss + anchor_loss
            valid_anchors += 1

        if valid_anchors > 0:
            loss = loss / valid_anchors

        return loss


class TripletUserVectorLoss(nn.Module):
    """Simpler alternative: triplet margin loss on user vectors.

    Samples (anchor, positive, negative) triplets from user vectors within
    the batch. Anchor and positive share the same label; negative has the
    opposite label.
    """

    def __init__(self, margin: float = 0.3) -> None:
        super().__init__()
        self.margin = margin
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(
        self,
        review_vectors: torch.Tensor,
        user_ids: torch.Tensor,
        user_labels_by_review: torch.Tensor,
    ) -> torch.Tensor:
        device = review_vectors.device
        unique_users = user_ids.unique()

        if len(unique_users) < 4:
            return torch.tensor(0.0, device=device, requires_grad=True)

        user_vectors = []
        user_labels = []

        for uid in unique_users:
            mask = user_ids == uid
            user_vec = review_vectors[mask].mean(dim=0)
            labels_for_user = user_labels_by_review[mask]
            if torch.any(labels_for_user != labels_for_user[0]):
                raise ValueError("RouteV triplet regularizer received inconsistent user_label values within a user batch.")
            user_label = labels_for_user[0]
            user_vectors.append(user_vec)
            user_labels.append(user_label)

        user_vectors = torch.stack(user_vectors)
        user_labels = torch.stack(user_labels)

        fake_mask = user_labels == 1
        real_mask = user_labels == 0
        fake_vecs = user_vectors[fake_mask]
        real_vecs = user_vectors[real_mask]

        if len(fake_vecs) < 2 or len(real_vecs) < 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Build triplets: anchor=fake[i], positive=fake[j], negative=real[k]
        n_triplets = min(len(fake_vecs), 8)
        anchors = fake_vecs[:n_triplets]
        positives = fake_vecs[torch.randperm(len(fake_vecs), device=device)[:n_triplets]]
        negatives = real_vecs[torch.randperm(len(real_vecs), device=device)[:n_triplets]]

        return self.triplet_loss(anchors, positives, negatives)
