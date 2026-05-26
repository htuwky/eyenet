from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from eyenet.data.encoder_dataset import EncoderPreprocessor, build_encoder_dataloaders
from eyenet.models.encoder import MaskedEventModel
from eyenet.training.segment_sequence import set_seed


@dataclass(frozen=True)
class MaskedEventModelingConfig:
    batch_size: int = 8
    max_epochs: int = 50
    patience: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    encoder_type: str = "bigru_attention"
    projection_dim: int = 64
    hidden_dim: int = 64
    attention_dim: int = 64
    num_layers: int = 1
    num_heads: int = 4
    feedforward_dim: int = 256
    dropout: float = 0.3
    random_seed: int = 42
    max_seq_len: int | None = None
    gradient_clip_norm: float = 5.0
    mask_probability: float = 0.15
    min_masked_events: int = 1
    mask_strategy: str = "span"
    min_mask_span_events: int = 2
    max_mask_span_events: int = 8
    require_label: bool = False


def train_masked_event_model(
    events: pd.DataFrame,
    split_subjects: pd.DataFrame,
    feature_columns: list[str],
    cfg: MaskedEventModelingConfig,
    device: str | None = None,
    checkpoint_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    set_seed(cfg.random_seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    loaders, preprocessor = build_encoder_dataloaders(
        events=events,
        split_subjects=split_subjects,
        feature_columns=feature_columns,
        batch_size=cfg.batch_size,
        max_seq_len=cfg.max_seq_len,
        balanced_train_sampler=False,
        require_label=cfg.require_label,
    )
    model = MaskedEventModel(
        input_dim=len(feature_columns),
        encoder_type=cfg.encoder_type,
        projection_dim=cfg.projection_dim,
        hidden_dim=cfg.hidden_dim,
        attention_dim=cfg.attention_dim,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        feedforward_dim=cfg.feedforward_dim,
        dropout=cfg.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    criterion = nn.MSELoss(reduction="none")

    best_valid_loss = np.inf
    best_state = None
    best_epoch = 0
    stopped_epoch = cfg.max_epochs
    epochs_without_improvement = 0
    log_rows: list[dict[str, Any]] = []

    for epoch in range(1, cfg.max_epochs + 1):
        train_loss, train_mask_rate = run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            cfg,
            optimizer=optimizer,
        )
        valid_loss, valid_mask_rate = run_epoch(
            model,
            loaders["valid"],
            criterion,
            device,
            cfg,
            optimizer=None,
        )
        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "train_mask_rate": train_mask_rate,
                "valid_mask_rate": valid_mask_rate,
            }
        )
        print(
            f"[mem] epoch {epoch:03d}/{cfg.max_epochs:03d} "
            f"train_loss={train_loss:.6f} valid_loss={valid_loss:.6f} "
            f"train_mask={train_mask_rate:.4f} valid_mask={valid_mask_rate:.4f}",
            flush=True,
        )
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= cfg.patience:
            stopped_epoch = epoch
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_mask_rate = run_epoch(
        model,
        loaders["test"],
        criterion,
        device,
        cfg,
        optimizer=None,
    )

    checkpoint_path = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if checkpoint_path is not None:
        save_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            preprocessor=preprocessor,
            cfg=cfg,
            feature_columns=feature_columns,
            best_epoch=best_epoch,
            stopped_epoch=stopped_epoch,
            best_valid_loss=float(best_valid_loss),
            test_loss=float(test_loss),
            device=device,
        )

    run_info = {
        "config": asdict(cfg),
        "device": device,
        "feature_columns": feature_columns,
        "n_features": len(feature_columns),
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "best_valid_loss": float(best_valid_loss),
        "test_loss": float(test_loss),
        "test_mask_rate": float(test_mask_rate),
    }
    return pd.DataFrame(log_rows), run_info


def run_epoch(
    model: MaskedEventModel,
    loader,
    criterion,
    device: str,
    cfg: MaskedEventModelingConfig,
    optimizer=None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    mask_rates: list[float] = []
    for batch in loader:
        x = batch["x"].to(device)
        valid_mask = batch["mask"].to(device)
        input_x, reconstruction_mask = apply_event_mask(
            x,
            valid_mask,
            mask_probability=cfg.mask_probability,
            min_masked_events=cfg.min_masked_events,
            mask_strategy=cfg.mask_strategy,
            min_mask_span_events=cfg.min_mask_span_events,
            max_mask_span_events=cfg.max_mask_span_events,
            mask_token=model.mask_token,
        )
        if training:
            optimizer.zero_grad()
        reconstruction, _ = model(input_x, valid_mask)
        loss = masked_reconstruction_loss(reconstruction, x, reconstruction_mask, criterion)
        if training:
            loss.backward()
            if cfg.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_norm)
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
        mask_rates.append(float(reconstruction_mask.float().mean().detach().cpu()))
    return float(np.mean(losses)), float(np.mean(mask_rates))


def apply_event_mask(
    x: torch.Tensor,
    valid_mask: torch.Tensor,
    mask_probability: float,
    min_masked_events: int,
    mask_strategy: str = "span",
    min_mask_span_events: int = 2,
    max_mask_span_events: int = 8,
    mask_token: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mask_strategy == "random":
        random_values = torch.rand(valid_mask.shape, device=x.device)
        reconstruction_mask = (random_values < mask_probability) & valid_mask
    elif mask_strategy == "span":
        reconstruction_mask = build_span_mask(
            valid_mask,
            mask_probability=mask_probability,
            min_masked_events=min_masked_events,
            min_span_events=min_mask_span_events,
            max_span_events=max_mask_span_events,
        )
    else:
        raise ValueError(f"Unsupported mask_strategy: {mask_strategy}. Use 'span' or 'random'.")

    ensure_minimum_masked_events(reconstruction_mask, valid_mask, min_masked_events)
    input_x = x.clone()
    if mask_token is None:
        raise ValueError("mask_token is required for masked event modeling.")
    input_x[reconstruction_mask] = mask_token.to(device=x.device, dtype=x.dtype)
    return input_x, reconstruction_mask


def build_span_mask(
    valid_mask: torch.Tensor,
    mask_probability: float,
    min_masked_events: int,
    min_span_events: int,
    max_span_events: int,
) -> torch.Tensor:
    reconstruction_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
    min_span_events = max(1, int(min_span_events))
    max_span_events = max(min_span_events, int(max_span_events))
    for row in range(valid_mask.shape[0]):
        valid_indices = torch.nonzero(valid_mask[row], as_tuple=False).flatten()
        n_valid = int(len(valid_indices))
        if n_valid == 0:
            continue
        target = max(int(round(n_valid * mask_probability)), int(min_masked_events))
        target = min(target, n_valid)
        attempts = 0
        while int(reconstruction_mask[row].sum().item()) < target and attempts < target * 10:
            span_len = int(
                torch.randint(
                    low=min_span_events,
                    high=max_span_events + 1,
                    size=(1,),
                    device=valid_mask.device,
                ).item()
            )
            start_offset = int(torch.randint(low=0, high=n_valid, size=(1,), device=valid_mask.device).item())
            selected = valid_indices[start_offset : min(start_offset + span_len, n_valid)]
            reconstruction_mask[row, selected] = True
            attempts += 1
    return reconstruction_mask & valid_mask


def ensure_minimum_masked_events(
    reconstruction_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    min_masked_events: int,
) -> None:
    for row in range(reconstruction_mask.shape[0]):
        if int(reconstruction_mask[row].sum().item()) < min_masked_events:
            valid_indices = torch.nonzero(valid_mask[row], as_tuple=False).flatten()
            if len(valid_indices) > 0:
                n_needed = min(int(min_masked_events), len(valid_indices))
                permutation = torch.randperm(len(valid_indices), device=valid_mask.device)[:n_needed]
                reconstruction_mask[row, valid_indices[permutation]] = True


def masked_reconstruction_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    reconstruction_mask: torch.Tensor,
    criterion,
) -> torch.Tensor:
    per_feature_loss = criterion(reconstruction, target).mean(dim=-1)
    masked_loss = per_feature_loss[reconstruction_mask]
    if masked_loss.numel() == 0:
        return per_feature_loss.mean() * 0.0
    return masked_loss.mean()


def save_checkpoint(
    checkpoint_path: Path,
    model: MaskedEventModel,
    preprocessor: EncoderPreprocessor,
    cfg: MaskedEventModelingConfig,
    feature_columns: list[str],
    best_epoch: int,
    stopped_epoch: int,
    best_valid_loss: float,
    test_loss: float,
    device: str,
) -> None:
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    preprocessor.save(checkpoint_path / "preprocessor.joblib")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "encoder_state_dict": model.encoder.state_dict(),
            "config": asdict(cfg),
            "feature_columns": feature_columns,
            "input_dim": len(feature_columns),
            "best_epoch": best_epoch,
            "stopped_epoch": stopped_epoch,
            "best_valid_loss": best_valid_loss,
            "test_loss": test_loss,
            "preprocessor_path": "preprocessor.joblib",
            "device": device,
            "model_type": "masked_event_model",
            "masking": "learnable_mask_token",
        },
        checkpoint_path / "best.pt",
    )


def save_masked_event_modeling_outputs(
    output_dir: str | Path,
    training_log: pd.DataFrame,
    run_info: dict[str, Any],
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    training_log.to_csv(output_path / "training_log.csv", index=False, encoding="utf-8-sig")
    (output_path / "config.json").write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")
