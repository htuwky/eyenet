from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from eyenet.data.encoder_dataset import EncoderPreprocessor, build_encoder_dataloaders
from eyenet.models.encoder import SupervisedEncoderClassifier
from eyenet.training.baseline import compute_metrics
from eyenet.training.segment_sequence import set_seed
from eyenet.training.thresholds import analyze_thresholds, choose_thresholds


@dataclass(frozen=True)
class SupervisedEncoderConfig:
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
    pos_weight: float = 1.5
    max_seq_len: int | None = None
    gradient_clip_norm: float = 5.0
    balanced_train_sampler: bool = True
    pretrained_checkpoint: str | None = None
    freeze_encoder: bool = False


def train_supervised_encoder(
    events: pd.DataFrame,
    split_subjects: pd.DataFrame,
    feature_columns: list[str],
    cfg: SupervisedEncoderConfig,
    device: str | None = None,
    checkpoint_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    set_seed(cfg.random_seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    loaders, preprocessor = build_encoder_dataloaders(
        events=events,
        split_subjects=split_subjects,
        feature_columns=feature_columns,
        batch_size=cfg.batch_size,
        max_seq_len=cfg.max_seq_len,
        balanced_train_sampler=cfg.balanced_train_sampler,
    )
    cfg = cfg_with_pretrained_architecture(cfg, device=device) if cfg.pretrained_checkpoint is not None else cfg
    model = SupervisedEncoderClassifier(
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
    pretrained_loaded = False
    if cfg.pretrained_checkpoint is not None:
        load_pretrained_encoder(
            model=model,
            checkpoint_path=cfg.pretrained_checkpoint,
            feature_columns=feature_columns,
            device=device,
        )
        pretrained_loaded = True

    if cfg.freeze_encoder:
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("No trainable parameters remain after applying freeze settings.")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg.pos_weight, dtype=torch.float32, device=device))

    best_auc = -np.inf
    best_state = None
    best_epoch = 0
    stopped_epoch = cfg.max_epochs
    epochs_without_improvement = 0
    log_rows: list[dict[str, Any]] = []

    for epoch in range(1, cfg.max_epochs + 1):
        train_loss = train_one_epoch(model, loaders["train"], optimizer, criterion, device, cfg.gradient_clip_norm)
        valid_loss, valid_true, valid_prob, _ = evaluate(model, loaders["valid"], criterion, device)
        valid_auc = roc_auc_score(valid_true, valid_prob)
        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "valid_auc": float(valid_auc),
            }
        )
        print(
            f"[supervised] epoch {epoch:03d}/{cfg.max_epochs:03d} "
            f"train_loss={train_loss:.6f} valid_loss={valid_loss:.6f} valid_auc={valid_auc:.6f}",
            flush=True,
        )
        if valid_auc > best_auc:
            best_auc = valid_auc
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
            best_valid_auc=float(best_auc),
            device=device,
            pretrained_loaded=pretrained_loaded,
        )

    valid_loss, valid_true, valid_prob, valid_subjects = evaluate(model, loaders["valid"], criterion, device)
    test_loss, test_true, test_prob, test_subjects = evaluate(model, loaders["test"], criterion, device)

    model_name = f"supervised_{cfg.encoder_type}_encoder"
    valid_predictions_default = make_prediction_frame(
        model_name,
        "valid",
        valid_subjects,
        valid_true,
        valid_prob,
        threshold=0.5,
    )
    valid_threshold_metrics = analyze_thresholds(valid_predictions_default)
    selected_thresholds = choose_thresholds(valid_threshold_metrics)
    threshold_map = build_threshold_map(selected_thresholds)

    prediction_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for split_name, labels, probabilities, subject_ids, loss in [
        ("valid", valid_true, valid_prob, valid_subjects, valid_loss),
        ("test", test_true, test_prob, test_subjects, test_loss),
    ]:
        for threshold_name, threshold in threshold_map.items():
            frame = make_prediction_frame(model_name, split_name, subject_ids, labels, probabilities, threshold=threshold)
            frame["threshold_name"] = threshold_name
            frame["threshold"] = threshold
            prediction_rows.extend(frame.to_dict(orient="records"))
            metrics = compute_metrics(
                frame["label"].to_numpy(dtype=int),
                frame["prediction"].to_numpy(dtype=int),
                frame["probability"].to_numpy(dtype=float),
            )
            metric_rows.append(
                {
                    "model": model_name,
                    "split": split_name,
                    "threshold_name": threshold_name,
                    "threshold": threshold,
                    "loss": loss,
                    "best_epoch": best_epoch,
                    "stopped_epoch": stopped_epoch,
                    "best_valid_auc": float(best_auc),
                    **metrics,
                }
            )

    run_info = {
        "config": asdict(cfg),
        "device": device,
        "feature_columns": feature_columns,
        "n_features": len(feature_columns),
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "best_valid_auc": float(best_auc),
        "pretrained_loaded": pretrained_loaded,
        "freeze_encoder": cfg.freeze_encoder,
        "n_trainable_parameters": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
        "n_total_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
    }
    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(prediction_rows),
        valid_threshold_metrics,
        selected_thresholds,
        pd.DataFrame(log_rows),
        run_info,
    )


def train_one_epoch(model, loader, optimizer, criterion, device: str, gradient_clip_norm: float) -> float:
    model.train()
    losses: list[float] = []
    for batch in loader:
        optimizer.zero_grad()
        x = batch["x"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["label"].to(device).float()
        logits, _, _ = model(x, mask)
        loss = criterion(logits, labels)
        loss.backward()
        if gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


@torch.no_grad()
def evaluate(model, loader, criterion, device: str) -> tuple[float, list[int], list[float], list[str]]:
    model.eval()
    losses: list[float] = []
    y_true: list[int] = []
    y_prob: list[float] = []
    subject_ids: list[str] = []
    for batch in loader:
        x = batch["x"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["label"].to(device).float()
        logits, _, _ = model(x, mask)
        loss = criterion(logits, labels)
        prob = torch.sigmoid(logits)
        losses.append(float(loss.detach().cpu()))
        y_true.extend(labels.cpu().numpy().astype(int).tolist())
        y_prob.extend(prob.cpu().numpy().tolist())
        subject_ids.extend(batch["subject_id"])
    return float(np.mean(losses)), y_true, y_prob, subject_ids


def load_pretrained_encoder(
    model: SupervisedEncoderClassifier,
    checkpoint_path: str | Path,
    feature_columns: list[str],
    device: str,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_features = checkpoint.get("feature_columns")
    if checkpoint_features is not None and list(checkpoint_features) != list(feature_columns):
        raise ValueError(
            "Pretrained checkpoint feature columns do not match current feature columns. "
            "Use the same encoder-ready schema for pretraining and downstream fine-tuning."
        )
    encoder_state = checkpoint.get("encoder_state_dict")
    if encoder_state is None:
        model_state = checkpoint.get("model_state_dict")
        if model_state is None:
            raise ValueError("Checkpoint does not contain encoder_state_dict or model_state_dict.")
        encoder_prefix = "encoder."
        encoder_state = {
            key.removeprefix(encoder_prefix): value
            for key, value in model_state.items()
            if key.startswith(encoder_prefix)
        }
    if not encoder_state:
        raise ValueError("No encoder weights were found in the pretrained checkpoint.")
    model.encoder.load_state_dict(encoder_state, strict=True)


def cfg_with_pretrained_architecture(cfg: SupervisedEncoderConfig, device: str) -> SupervisedEncoderConfig:
    checkpoint = torch.load(cfg.pretrained_checkpoint, map_location=device)
    checkpoint_cfg = checkpoint.get("config") or {}
    updates = {
        "encoder_type": checkpoint_cfg.get("encoder_type", "bigru_attention"),
        "projection_dim": checkpoint_cfg.get("projection_dim", cfg.projection_dim),
        "hidden_dim": checkpoint_cfg.get("hidden_dim", cfg.hidden_dim),
        "attention_dim": checkpoint_cfg.get("attention_dim", cfg.attention_dim),
        "num_layers": checkpoint_cfg.get("num_layers", 1),
        "num_heads": checkpoint_cfg.get("num_heads", cfg.num_heads),
        "feedforward_dim": checkpoint_cfg.get("feedforward_dim", cfg.feedforward_dim),
        "dropout": checkpoint_cfg.get("dropout", cfg.dropout),
    }
    return SupervisedEncoderConfig(**{**asdict(cfg), **updates})


def make_prediction_frame(
    model_name: str,
    split_name: str,
    subject_ids: list[str],
    labels: list[int],
    probabilities: list[float],
    threshold: float,
) -> pd.DataFrame:
    probabilities_array = np.asarray(probabilities, dtype=float)
    return pd.DataFrame(
        {
            "model": model_name,
            "split": split_name,
            "subject_id": subject_ids,
            "label": np.asarray(labels, dtype=int),
            "probability": probabilities_array,
            "prediction": (probabilities_array >= threshold).astype(int),
        }
    )


def build_threshold_map(selected_thresholds: pd.DataFrame) -> dict[str, float]:
    best_balanced = selected_thresholds[selected_thresholds["criterion"] == "best_balanced_accuracy"].iloc[0]
    best_f1 = selected_thresholds[selected_thresholds["criterion"] == "best_f1"].iloc[0]
    screening_candidates = selected_thresholds[
        selected_thresholds["criterion"].str.contains("sensitivity_at_least_0.80", regex=False)
    ]
    screening_threshold = (
        float(screening_candidates.iloc[0]["threshold"])
        if not screening_candidates.empty
        else float(best_balanced["threshold"])
    )
    return {
        "default_0.50": 0.5,
        "valid_best_balanced_accuracy": float(best_balanced["threshold"]),
        "valid_best_f1": float(best_f1["threshold"]),
        "valid_screening_sensitivity_at_least_0.80": screening_threshold,
    }


def save_checkpoint(
    checkpoint_path: Path,
    model: SupervisedEncoderClassifier,
    preprocessor: EncoderPreprocessor,
    cfg: SupervisedEncoderConfig,
    feature_columns: list[str],
    best_epoch: int,
    stopped_epoch: int,
    best_valid_auc: float,
    device: str,
    pretrained_loaded: bool,
) -> None:
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    preprocessor.save(checkpoint_path / "preprocessor.joblib")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(cfg),
            "feature_columns": feature_columns,
            "input_dim": len(feature_columns),
            "best_epoch": best_epoch,
            "stopped_epoch": stopped_epoch,
            "best_valid_auc": best_valid_auc,
            "preprocessor_path": "preprocessor.joblib",
            "device": device,
            "model_type": "supervised_bigru_attention_encoder",
            "pretrained_loaded": pretrained_loaded,
            "freeze_encoder": cfg.freeze_encoder,
            "pretrained_checkpoint": cfg.pretrained_checkpoint,
        },
        checkpoint_path / "best.pt",
    )


def save_supervised_encoder_outputs(
    output_dir: str | Path,
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    valid_threshold_metrics: pd.DataFrame,
    selected_thresholds: pd.DataFrame,
    training_log: pd.DataFrame,
    run_info: dict[str, Any],
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_path / "metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(output_path / "predictions.csv", index=False, encoding="utf-8-sig")
    valid_threshold_metrics.to_csv(output_path / "valid_threshold_metrics.csv", index=False, encoding="utf-8-sig")
    selected_thresholds.to_csv(output_path / "selected_thresholds.csv", index=False, encoding="utf-8-sig")
    training_log.to_csv(output_path / "training_log.csv", index=False, encoding="utf-8-sig")
    (output_path / "config.json").write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")
