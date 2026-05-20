from __future__ import annotations

import torch
from torch import nn


class MaskedAttentionPooling(nn.Module):
    def __init__(self, input_dim: int, attention_dim: int = 64) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.attention(x).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class SegmentSequenceAttentionModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        projection_dim: int = 64,
        hidden_dim: int = 64,
        attention_dim: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.segment_projection = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(
            input_size=projection_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        encoded_dim = hidden_dim * 2
        self.pooling = MaskedAttentionPooling(encoded_dim, attention_dim=attention_dim)
        self.classifier = nn.Sequential(
            nn.Linear(encoded_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self.segment_projection(x)
        encoded, _ = self.gru(projected)
        pooled, attention_weights = self.pooling(encoded, mask)
        logits = self.classifier(pooled).squeeze(-1)
        return logits, attention_weights
