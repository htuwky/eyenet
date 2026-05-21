from __future__ import annotations

import torch
from torch import nn


class MaskedAttentionPooling(nn.Module):
    def __init__(self, input_dim: int, attention_dim: int = 64) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.scorer(x).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class EyeMovementEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        projection_dim: int = 64,
        hidden_dim: int = 64,
        attention_dim: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.encoder = nn.GRU(
            input_size=projection_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.output_dim = hidden_dim * 2
        self.pooling = MaskedAttentionPooling(self.output_dim, attention_dim=attention_dim)

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.input_projection(x)
        encoded, _ = self.encoder(projected)
        return encoded

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encode_sequence(x)
        pooled, attention_weights = self.pooling(encoded, mask)
        return pooled, attention_weights


class SupervisedEncoderClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        projection_dim: int = 64,
        hidden_dim: int = 64,
        attention_dim: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.encoder = EyeMovementEncoder(
            input_dim=input_dim,
            projection_dim=projection_dim,
            hidden_dim=hidden_dim,
            attention_dim=attention_dim,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedding, attention_weights = self.encoder(x, mask)
        logits = self.classifier(embedding).squeeze(-1)
        return logits, embedding, attention_weights


class MaskedEventModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        projection_dim: int = 64,
        hidden_dim: int = 64,
        attention_dim: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.encoder = EyeMovementEncoder(
            input_dim=input_dim,
            projection_dim=projection_dim,
            hidden_dim=hidden_dim,
            attention_dim=attention_dim,
            dropout=dropout,
        )
        self.reconstruction_head = nn.Sequential(
            nn.Linear(self.encoder.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder.encode_sequence(x)
        reconstruction = self.reconstruction_head(encoded)
        return reconstruction, encoded
