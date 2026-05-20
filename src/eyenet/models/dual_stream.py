from __future__ import annotations

import torch
from torch import nn

from eyenet.models.segment_sequence import MaskedAttentionPooling


class DualStreamConcatAttentionModel(nn.Module):
    def __init__(
        self,
        macro_input_dim: int,
        event_input_dim: int,
        projection_dim: int = 64,
        hidden_dim: int = 64,
        attention_dim: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.macro_projection = nn.Sequential(
            nn.Linear(macro_input_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.event_projection = nn.Sequential(
            nn.Linear(event_input_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.macro_gru = nn.GRU(
            input_size=projection_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.event_gru = nn.GRU(
            input_size=projection_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        encoded_dim = hidden_dim * 2
        self.macro_pooling = MaskedAttentionPooling(encoded_dim, attention_dim=attention_dim)
        self.event_pooling = MaskedAttentionPooling(encoded_dim, attention_dim=attention_dim)
        self.classifier = nn.Sequential(
            nn.Linear(encoded_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        macro_x: torch.Tensor,
        macro_mask: torch.Tensor,
        event_x: torch.Tensor,
        event_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        macro_projected = self.macro_projection(macro_x)
        event_projected = self.event_projection(event_x)
        macro_encoded, _ = self.macro_gru(macro_projected)
        event_encoded, _ = self.event_gru(event_projected)
        macro_pooled, macro_attention = self.macro_pooling(macro_encoded, macro_mask)
        event_pooled, event_attention = self.event_pooling(event_encoded, event_mask)
        fused = torch.cat([macro_pooled, event_pooled], dim=-1)
        logits = self.classifier(fused).squeeze(-1)
        return logits, {"macro": macro_attention, "event": event_attention}


class DualStreamGatedAttentionModel(nn.Module):
    def __init__(
        self,
        macro_input_dim: int,
        event_input_dim: int,
        projection_dim: int = 64,
        hidden_dim: int = 64,
        attention_dim: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.macro_projection = nn.Sequential(
            nn.Linear(macro_input_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.event_projection = nn.Sequential(
            nn.Linear(event_input_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.macro_gru = nn.GRU(
            input_size=projection_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.event_gru = nn.GRU(
            input_size=projection_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        encoded_dim = hidden_dim * 2
        self.macro_pooling = MaskedAttentionPooling(encoded_dim, attention_dim=attention_dim)
        self.event_pooling = MaskedAttentionPooling(encoded_dim, attention_dim=attention_dim)
        self.gate = nn.Sequential(
            nn.Linear(encoded_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(encoded_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        macro_x: torch.Tensor,
        macro_mask: torch.Tensor,
        event_x: torch.Tensor,
        event_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        macro_projected = self.macro_projection(macro_x)
        event_projected = self.event_projection(event_x)
        macro_encoded, _ = self.macro_gru(macro_projected)
        event_encoded, _ = self.event_gru(event_projected)
        macro_pooled, macro_attention = self.macro_pooling(macro_encoded, macro_mask)
        event_pooled, event_attention = self.event_pooling(event_encoded, event_mask)
        gate = self.gate(torch.cat([macro_pooled, event_pooled], dim=-1))
        fused = gate * macro_pooled + (1.0 - gate) * event_pooled
        logits = self.classifier(fused).squeeze(-1)
        return logits, {"macro": macro_attention, "event": event_attention, "macro_gate": gate.squeeze(-1)}
