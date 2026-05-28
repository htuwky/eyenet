from __future__ import annotations

import torch
from torch import nn

from eyenet.models.encoder import EyeMovementEncoder
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


class EncoderDualStreamConcatModel(nn.Module):
    def __init__(
        self,
        macro_input_dim: int,
        encoder_input_dim: int,
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
        self.macro_projection = nn.Sequential(
            nn.Linear(macro_input_dim, projection_dim),
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
        macro_dim = hidden_dim * 2
        self.macro_pooling = MaskedAttentionPooling(macro_dim, attention_dim=attention_dim)
        self.event_encoder = EyeMovementEncoder(
            input_dim=encoder_input_dim,
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
            nn.Linear(macro_dim + self.event_encoder.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        macro_x: torch.Tensor,
        macro_mask: torch.Tensor,
        encoder_x: torch.Tensor,
        encoder_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        macro_projected = self.macro_projection(macro_x)
        macro_encoded, _ = self.macro_gru(macro_projected)
        macro_pooled, macro_attention = self.macro_pooling(macro_encoded, macro_mask)
        event_pooled, event_attention = self.event_encoder(encoder_x, encoder_mask)
        fused = torch.cat([macro_pooled, event_pooled], dim=-1)
        logits = self.classifier(fused).squeeze(-1)
        return logits, {"macro": macro_attention, "event": event_attention}


class EncoderDualStreamGatedModel(nn.Module):
    def __init__(
        self,
        macro_input_dim: int,
        encoder_input_dim: int,
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
        self.macro_projection = nn.Sequential(
            nn.Linear(macro_input_dim, projection_dim),
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
        macro_dim = hidden_dim * 2
        fusion_dim = hidden_dim * 2
        self.macro_pooling = MaskedAttentionPooling(macro_dim, attention_dim=attention_dim)
        self.event_encoder = EyeMovementEncoder(
            input_dim=encoder_input_dim,
            encoder_type=encoder_type,
            projection_dim=projection_dim,
            hidden_dim=hidden_dim,
            attention_dim=attention_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
        )
        self.macro_fusion_projection = nn.Linear(macro_dim, fusion_dim)
        self.event_fusion_projection = nn.Linear(self.event_encoder.output_dim, fusion_dim)
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        macro_x: torch.Tensor,
        macro_mask: torch.Tensor,
        encoder_x: torch.Tensor,
        encoder_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        macro_projected = self.macro_projection(macro_x)
        macro_encoded, _ = self.macro_gru(macro_projected)
        macro_pooled, macro_attention = self.macro_pooling(macro_encoded, macro_mask)
        event_pooled, event_attention = self.event_encoder(encoder_x, encoder_mask)
        macro_fused = self.macro_fusion_projection(macro_pooled)
        event_fused = self.event_fusion_projection(event_pooled)
        gate = self.gate(torch.cat([macro_fused, event_fused], dim=-1))
        fused = gate * macro_fused + (1.0 - gate) * event_fused
        logits = self.classifier(fused).squeeze(-1)
        return logits, {"macro": macro_attention, "event": event_attention, "macro_gate": gate.squeeze(-1)}


class SummaryEncoderDualStreamConcatModel(nn.Module):
    def __init__(
        self,
        summary_input_dim: int,
        encoder_input_dim: int,
        encoder_type: str = "bigru_attention",
        projection_dim: int = 64,
        hidden_dim: int = 64,
        attention_dim: int = 64,
        num_layers: int = 1,
        num_heads: int = 4,
        feedforward_dim: int = 256,
        dropout: float = 0.3,
        summary_dim: int | None = None,
    ) -> None:
        super().__init__()
        summary_dim = summary_dim or hidden_dim * 2
        self.summary_encoder = nn.Sequential(
            nn.Linear(summary_input_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, summary_dim),
            nn.LayerNorm(summary_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.event_encoder = EyeMovementEncoder(
            input_dim=encoder_input_dim,
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
            nn.Linear(summary_dim + self.event_encoder.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        summary_x: torch.Tensor,
        encoder_x: torch.Tensor,
        encoder_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        summary_pooled = self.summary_encoder(summary_x)
        event_pooled, event_attention = self.event_encoder(encoder_x, encoder_mask)
        fused = torch.cat([summary_pooled, event_pooled], dim=-1)
        logits = self.classifier(fused).squeeze(-1)
        return logits, {"event": event_attention}


class SummaryEncoderDualStreamGatedModel(nn.Module):
    def __init__(
        self,
        summary_input_dim: int,
        encoder_input_dim: int,
        encoder_type: str = "bigru_attention",
        projection_dim: int = 64,
        hidden_dim: int = 64,
        attention_dim: int = 64,
        num_layers: int = 1,
        num_heads: int = 4,
        feedforward_dim: int = 256,
        dropout: float = 0.3,
        summary_dim: int | None = None,
    ) -> None:
        super().__init__()
        fusion_dim = hidden_dim * 2
        summary_dim = summary_dim or fusion_dim
        self.summary_encoder = nn.Sequential(
            nn.Linear(summary_input_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, summary_dim),
            nn.LayerNorm(summary_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.event_encoder = EyeMovementEncoder(
            input_dim=encoder_input_dim,
            encoder_type=encoder_type,
            projection_dim=projection_dim,
            hidden_dim=hidden_dim,
            attention_dim=attention_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
        )
        self.summary_fusion_projection = nn.Linear(summary_dim, fusion_dim)
        self.event_fusion_projection = nn.Linear(self.event_encoder.output_dim, fusion_dim)
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        summary_x: torch.Tensor,
        encoder_x: torch.Tensor,
        encoder_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        summary_fused = self.summary_fusion_projection(self.summary_encoder(summary_x))
        event_pooled, event_attention = self.event_encoder(encoder_x, encoder_mask)
        event_fused = self.event_fusion_projection(event_pooled)
        gate = self.gate(torch.cat([summary_fused, event_fused], dim=-1))
        fused = gate * summary_fused + (1.0 - gate) * event_fused
        logits = self.classifier(fused).squeeze(-1)
        return logits, {"event": event_attention, "summary_gate": gate.squeeze(-1)}


class SummaryEncoderDualStreamResidualLogitModel(nn.Module):
    def __init__(
        self,
        summary_input_dim: int,
        encoder_input_dim: int,
        encoder_type: str = "bigru_attention",
        projection_dim: int = 64,
        hidden_dim: int = 64,
        attention_dim: int = 64,
        num_layers: int = 1,
        num_heads: int = 4,
        feedforward_dim: int = 256,
        dropout: float = 0.3,
        summary_dim: int | None = 16,
        residual_scale: float = 0.25,
    ) -> None:
        super().__init__()
        summary_dim = summary_dim or 16
        self.residual_scale = float(residual_scale)
        self.summary_encoder = nn.Sequential(
            nn.Linear(summary_input_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, summary_dim),
            nn.LayerNorm(summary_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.event_encoder = EyeMovementEncoder(
            input_dim=encoder_input_dim,
            encoder_type=encoder_type,
            projection_dim=projection_dim,
            hidden_dim=hidden_dim,
            attention_dim=attention_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
        )
        self.encoder_head = nn.Sequential(
            nn.Linear(self.event_encoder.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.summary_head = nn.Sequential(
            nn.Linear(summary_dim, max(summary_dim, 8)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(summary_dim, 8), 1),
        )
        self.raw_summary_alpha = nn.Parameter(torch.tensor(-2.0))

    def forward(
        self,
        summary_x: torch.Tensor,
        encoder_x: torch.Tensor,
        encoder_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        summary_pooled = self.summary_encoder(summary_x)
        event_pooled, event_attention = self.event_encoder(encoder_x, encoder_mask)
        encoder_logit = self.encoder_head(event_pooled).squeeze(-1)
        summary_logit = self.summary_head(summary_pooled).squeeze(-1)
        summary_alpha = self.residual_scale * torch.sigmoid(self.raw_summary_alpha)
        logits = encoder_logit + summary_alpha * summary_logit
        return logits, {
            "event": event_attention,
            "summary_alpha": summary_alpha.expand_as(logits),
            "encoder_logit": encoder_logit,
            "summary_logit": summary_logit,
        }
