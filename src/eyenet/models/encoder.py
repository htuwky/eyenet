from __future__ import annotations

import math

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
        encoder_type: str = "bigru_attention",
        projection_dim: int = 64,
        hidden_dim: int = 64,
        attention_dim: int = 64,
        num_layers: int = 1,
        num_heads: int = 4,
        feedforward_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if encoder_type not in {"bigru_attention", "transformer"}:
            raise ValueError("encoder_type must be one of: bigru_attention, transformer")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.encoder_type = encoder_type
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        if encoder_type == "bigru_attention":
            self.encoder = nn.GRU(
                input_size=projection_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.output_dim = hidden_dim * 2
            self.positional_encoding = None
        else:
            if projection_dim % num_heads != 0:
                raise ValueError("projection_dim must be divisible by num_heads for transformer encoders.")
            transformer_layer = nn.TransformerEncoderLayer(
                d_model=projection_dim,
                nhead=num_heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
            self.output_dim = projection_dim
            self.positional_encoding = SinusoidalPositionalEncoding(projection_dim)
        self.pooling = MaskedAttentionPooling(self.output_dim, attention_dim=attention_dim)

    def encode_sequence(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        projected = self.input_projection(x)
        if self.encoder_type == "bigru_attention":
            encoded, _ = self.encoder(projected)
            return encoded
        if self.positional_encoding is not None:
            projected = self.positional_encoding(projected)
        key_padding_mask = None if mask is None else ~mask
        return self.encoder(projected, src_key_padding_mask=key_padding_mask)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encode_sequence(x, mask=mask)
        pooled, attention_weights = self.pooling(encoded, mask)
        return pooled, attention_weights


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 10000) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] > self.pe.shape[1]:
            raise ValueError(f"Sequence length {x.shape[1]} exceeds positional encoding limit {self.pe.shape[1]}.")
        return x + self.pe[:, : x.shape[1]].to(dtype=x.dtype)


class SupervisedEncoderClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        encoder_type: str = "bigru_attention",
        projection_dim: int = 64,
        hidden_dim: int = 64,
        attention_dim: int = 64,
        num_layers: int = 1,
        num_heads: int = 4,
        feedforward_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.encoder = EyeMovementEncoder(
            input_dim=input_dim,
            encoder_type=encoder_type,
            projection_dim=projection_dim,
            hidden_dim=hidden_dim,
            attention_dim=attention_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
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
        encoder_type: str = "bigru_attention",
        projection_dim: int = 64,
        hidden_dim: int = 64,
        attention_dim: int = 64,
        num_layers: int = 1,
        num_heads: int = 4,
        feedforward_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(input_dim))
        self.encoder = EyeMovementEncoder(
            input_dim=input_dim,
            encoder_type=encoder_type,
            projection_dim=projection_dim,
            hidden_dim=hidden_dim,
            attention_dim=attention_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
        )
        self.reconstruction_head = nn.Sequential(
            nn.Linear(self.encoder.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder.encode_sequence(x, mask=mask)
        reconstruction = self.reconstruction_head(encoded)
        return reconstruction, encoded
