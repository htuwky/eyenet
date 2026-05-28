from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from eyenet.data.encoder_dataset import (
    SubjectSequenceDataset,
    add_subject_key,
    align_split_dataset_id,
    fit_encoder_preprocessor,
)
from eyenet.data.subject_summary import (
    EXCLUDE_COLUMNS,
    apply_summary_feature_set,
    fit_subject_summary_preprocessor,
    prepare_subject_summary_table,
    select_summary_feature_columns,
)
from eyenet.models.dual_stream import (
    SummaryEncoderDualStreamConcatModel,
    SummaryEncoderDualStreamGatedModel,
    SummaryEncoderDualStreamResidualLogitModel,
)
from eyenet.training.baseline import compute_metrics
from eyenet.training.encoder_dual_stream import load_pretrained_event_encoder, load_split_subjects
from eyenet.training.segment_sequence import set_seed
from eyenet.training.supervised_encoder import build_threshold_map
from eyenet.training.thresholds import analyze_thresholds, choose_thresholds


@dataclass(frozen=True)
class SummaryEncoderDualStreamConfig:
    batch_size: int = 8
    max_epochs: int = 100
    patience: int = 15
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    fusion: str = "concat"
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
    max_seq_len: int | None = 1500
    gradient_clip_norm: float = 5.0
    pretrained_checkpoint: str | None = None
    freeze_encoder: bool = False
    summary_feature_set: str = "strict"
    summary_dim: int = 16
    summary_residual_scale: float = 0.25
    max_train_missing_rate: float = 0.4


class SummaryEncoderDualStreamDataset(Dataset):
    def __init__(
        self,
        summary_arrays: dict[str, np.ndarray],
        encoder_arrays: dict[str, np.ndarray],
        encoder_masks: dict[str, np.ndarray],
        labels: dict[str, int],
        subject_ids: list[str],
    ) -> None:
        self.summary_arrays = summary_arrays
        self.encoder_arrays = encoder_arrays
        self.encoder_masks = encoder_masks
        self.labels = labels
        self.subject_ids = subject_ids

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        subject_id = self.subject_ids[index]
        return {
            "summary_x": torch.tensor(self.summary_arrays[subject_id], dtype=torch.float32),
            "encoder_x": torch.tensor(self.encoder_arrays[subject_id], dtype=torch.float32),
            "encoder_mask": torch.tensor(self.encoder_masks[subject_id], dtype=torch.bool),
            "label": torch.tensor(self.labels[subject_id], dtype=torch.float32),
            "subject_id": subject_id,
        }


def train_fixed_split(
    subject_summary: pd.DataFrame,
    encoder_events: pd.DataFrame,
    split_subjects: pd.DataFrame,
    encoder_feature_columns: list[str],
    cfg: SummaryEncoderDualStreamConfig,
    device: str | None = None,
    checkpoint_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    set_seed(cfg.random_seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = cfg_with_pretrained_architecture(cfg, device=device) if cfg.pretrained_checkpoint is not None else cfg

    summary_data = prepare_summary_split_data(subject_summary, split_subjects)
    candidate_cols = [col for col in summary_data.columns if col not in EXCLUDE_COLUMNS]
    candidate_cols, feature_set_audit = apply_summary_feature_set(candidate_cols, cfg.summary_feature_set)
    train_summary = summary_data[summary_data["split"] == "train"].copy()
    feature_cols, feature_audit = select_summary_feature_columns(
        train_df=train_summary,
        candidate_cols=candidate_cols,
        max_train_missing_rate=cfg.max_train_missing_rate,
    )
    feature_audit = feature_audit.merge(feature_set_audit, on="feature", how="right")
    summary_preprocessor = fit_subject_summary_preprocessor(train_summary, feature_cols)

    encoder_split_subjects = align_split_dataset_id(encoder_events, split_subjects)
    encoder_split_subjects = add_subject_key(encoder_split_subjects)
    train_subjects = set(encoder_split_subjects.loc[encoder_split_subjects["split"] == "train", "_subject_key"])
    encoder_preprocessor = fit_encoder_preprocessor(
        encoder_events,
        encoder_feature_columns,
        train_subjects=train_subjects,
    )

    datasets = {
        split_name: build_dataset(
            summary_data=summary_data,
            encoder_events=encoder_events,
            split_subjects=encoder_split_subjects,
            split_name=split_name,
            summary_feature_cols=feature_cols,
            summary_preprocessor=summary_preprocessor,
            encoder_feature_cols=encoder_feature_columns,
            encoder_preprocessor=encoder_preprocessor,
            max_seq_len=cfg.max_seq_len,
        )
        for split_name in ["train", "valid", "test"]
    }
    loaders = {
        split_name: DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=split_name == "train",
        )
        for split_name, dataset in datasets.items()
    }

    model = build_model(cfg, summary_input_dim=len(feature_cols), encoder_input_dim=len(encoder_feature_columns)).to(device)
    pretrained_loaded = False
    if cfg.pretrained_checkpoint is not None:
        load_pretrained_event_encoder(
            event_encoder=model.event_encoder,
            checkpoint_path=cfg.pretrained_checkpoint,
            feature_columns=encoder_feature_columns,
            device=device,
        )
        pretrained_loaded = True
    if cfg.freeze_encoder:
        for parameter in model.event_encoder.parameters():
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
        log_rows.append({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss, "valid_auc": float(valid_auc)})
        print(
            f"[summary-encoder-dual-{cfg.fusion}] epoch {epoch:03d}/{cfg.max_epochs:03d} "
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
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "summary_preprocessor": summary_preprocessor,
                "summary_feature_cols": feature_cols,
                "feature_audit": feature_audit,
                "encoder_preprocessor": encoder_preprocessor,
                "encoder_feature_cols": encoder_feature_columns,
            },
            checkpoint_path / "preprocessor.joblib",
        )
        torch.save(
            {
                "best_epoch": best_epoch,
                "stopped_epoch": stopped_epoch,
                "best_valid_auc": float(best_auc),
                "model_state_dict": model.state_dict(),
                "config": asdict(cfg),
                "summary_feature_cols": feature_cols,
                "encoder_feature_cols": encoder_feature_columns,
                "preprocessor_path": "preprocessor.joblib",
                "pretrained_loaded": pretrained_loaded,
                "device": device,
            },
            checkpoint_path / "best.pt",
        )

    valid_loss, valid_true, valid_prob, valid_subjects = evaluate(model, loaders["valid"], criterion, device)
    test_loss, test_true, test_prob, test_subjects = evaluate(model, loaders["test"], criterion, device)
    model_name = f"summary_encoder_dual_stream_{cfg.fusion}_{cfg.summary_feature_set}"
    valid_predictions_default = make_prediction_frame(model_name, "valid", valid_subjects, valid_true, valid_prob, 0.5)
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
            frame = make_prediction_frame(model_name, split_name, subject_ids, labels, probabilities, threshold)
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

    attention_rows = collect_attention_weights(model, loaders["valid"], device, split_name="valid", fusion=cfg.fusion)
    attention_rows.extend(collect_attention_weights(model, loaders["test"], device, split_name="test", fusion=cfg.fusion))
    run_info = {
        "config": asdict(cfg),
        "device": device,
        "summary_feature_cols": feature_cols,
        "encoder_feature_cols": encoder_feature_columns,
        "n_summary_features": len(feature_cols),
        "n_encoder_features": len(encoder_feature_columns),
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
        pd.DataFrame(attention_rows),
        run_info,
    )


def prepare_summary_split_data(subject_summary: pd.DataFrame, split_subjects: pd.DataFrame) -> pd.DataFrame:
    summary = prepare_subject_summary_table(subject_summary)
    split = split_subjects.copy()
    split["subject_id"] = split["subject_id"].astype(str).str.zfill(3)
    data = summary.drop(columns=["split"], errors="ignore").merge(
        split[["subject_id", "split", "official_fold"]],
        on="subject_id",
        how="inner",
    )
    if data["subject_id"].nunique() != split["subject_id"].nunique():
        raise ValueError("Subject summary table and split file have different labeled subject coverage.")
    if data["label"].isna().any():
        missing = data.loc[data["label"].isna(), "subject_id"].tolist()
        raise ValueError(f"Split subjects missing labels in summary table: {missing[:10]}")
    return data


def build_model(cfg: SummaryEncoderDualStreamConfig, summary_input_dim: int, encoder_input_dim: int):
    if cfg.fusion == "residual_logit":
        return SummaryEncoderDualStreamResidualLogitModel(
            summary_input_dim=summary_input_dim,
            encoder_input_dim=encoder_input_dim,
            encoder_type=cfg.encoder_type,
            projection_dim=cfg.projection_dim,
            hidden_dim=cfg.hidden_dim,
            attention_dim=cfg.attention_dim,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            feedforward_dim=cfg.feedforward_dim,
            dropout=cfg.dropout,
            summary_dim=cfg.summary_dim,
            residual_scale=cfg.summary_residual_scale,
        )
    model_cls = SummaryEncoderDualStreamGatedModel if cfg.fusion == "gated" else SummaryEncoderDualStreamConcatModel
    return model_cls(
        summary_input_dim=summary_input_dim,
        encoder_input_dim=encoder_input_dim,
        encoder_type=cfg.encoder_type,
        projection_dim=cfg.projection_dim,
        hidden_dim=cfg.hidden_dim,
        attention_dim=cfg.attention_dim,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        feedforward_dim=cfg.feedforward_dim,
        dropout=cfg.dropout,
        summary_dim=cfg.summary_dim,
    )


def cfg_with_pretrained_architecture(cfg: SummaryEncoderDualStreamConfig, device: str) -> SummaryEncoderDualStreamConfig:
    checkpoint = torch.load(cfg.pretrained_checkpoint, map_location=device)
    checkpoint_cfg = checkpoint.get("config") or {}
    updates = {
        "encoder_type": checkpoint_cfg.get("encoder_type", cfg.encoder_type),
        "projection_dim": checkpoint_cfg.get("projection_dim", cfg.projection_dim),
        "hidden_dim": checkpoint_cfg.get("hidden_dim", cfg.hidden_dim),
        "attention_dim": checkpoint_cfg.get("attention_dim", cfg.attention_dim),
        "num_layers": checkpoint_cfg.get("num_layers", cfg.num_layers),
        "num_heads": checkpoint_cfg.get("num_heads", cfg.num_heads),
        "feedforward_dim": checkpoint_cfg.get("feedforward_dim", cfg.feedforward_dim),
        "dropout": checkpoint_cfg.get("dropout", cfg.dropout),
    }
    return SummaryEncoderDualStreamConfig(**{**asdict(cfg), **updates})


def build_dataset(
    summary_data: pd.DataFrame,
    encoder_events: pd.DataFrame,
    split_subjects: pd.DataFrame,
    split_name: str,
    summary_feature_cols: list[str],
    summary_preprocessor,
    encoder_feature_cols: list[str],
    encoder_preprocessor,
    max_seq_len: int | None,
) -> SummaryEncoderDualStreamDataset:
    summary_split = summary_data[summary_data["split"] == split_name].copy().sort_values("subject_id")
    summary_values = summary_preprocessor.transform(summary_split)
    summary_array_map = dict(zip(summary_split["subject_id"], summary_values, strict=False))
    summary_label_map = dict(zip(summary_split["subject_id"], summary_split["label"].astype(int), strict=False))

    encoder_dataset = SubjectSequenceDataset(
        events=encoder_events,
        feature_columns=encoder_feature_cols,
        preprocessor=encoder_preprocessor,
        split_subjects=split_subjects,
        split_name=split_name,
        max_seq_len=max_seq_len,
        require_label=True,
    )
    encoder_max_len = int(max(sample["features"].shape[0] for sample in encoder_dataset.samples))
    encoder_array_map = {}
    encoder_mask_map = {}
    encoder_label_map = {}
    for sample in encoder_dataset.samples:
        features = sample["features"]
        padded = np.zeros((encoder_max_len, len(encoder_feature_cols)), dtype=np.float32)
        mask = np.zeros(encoder_max_len, dtype=bool)
        length = int(features.shape[0])
        padded[:length] = features
        mask[:length] = True
        encoder_array_map[sample["subject_id"]] = padded
        encoder_mask_map[sample["subject_id"]] = mask
        encoder_label_map[sample["subject_id"]] = int(sample["label"])

    common_subjects = sorted(set(summary_array_map) & set(encoder_array_map))
    if len(common_subjects) != len(set(summary_array_map)) or len(common_subjects) != len(set(encoder_array_map)):
        raise ValueError("Subject summary and encoder streams do not cover the same subjects in this split.")
    for subject_id in common_subjects:
        if int(summary_label_map[subject_id]) != int(encoder_label_map[subject_id]):
            raise ValueError(f"Label mismatch between streams for subject {subject_id}.")
    return SummaryEncoderDualStreamDataset(
        summary_arrays=summary_array_map,
        encoder_arrays=encoder_array_map,
        encoder_masks=encoder_mask_map,
        labels=summary_label_map,
        subject_ids=common_subjects,
    )


def train_one_epoch(model, loader, optimizer, criterion, device: str, gradient_clip_norm: float) -> float:
    model.train()
    losses: list[float] = []
    for batch in loader:
        optimizer.zero_grad()
        logits, _ = model(
            batch["summary_x"].to(device),
            batch["encoder_x"].to(device),
            batch["encoder_mask"].to(device),
        )
        loss = criterion(logits, batch["label"].to(device))
        loss.backward()
        if gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


@torch.no_grad()
def evaluate(model, loader, criterion, device: str):
    model.eval()
    losses: list[float] = []
    y_true: list[int] = []
    y_prob: list[float] = []
    subject_ids: list[str] = []
    for batch in loader:
        logits, _ = model(
            batch["summary_x"].to(device),
            batch["encoder_x"].to(device),
            batch["encoder_mask"].to(device),
        )
        label = batch["label"].to(device)
        loss = criterion(logits, label)
        prob = torch.sigmoid(logits)
        losses.append(float(loss.detach().cpu()))
        y_true.extend(label.cpu().numpy().astype(int).tolist())
        y_prob.extend(prob.cpu().numpy().tolist())
        subject_ids.extend(batch["subject_id"])
    return float(np.mean(losses)), y_true, y_prob, subject_ids


@torch.no_grad()
def collect_attention_weights(model, loader, device: str, split_name: str, fusion: str) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        logits, attention = model(
            batch["summary_x"].to(device),
            batch["encoder_x"].to(device),
            batch["encoder_mask"].to(device),
        )
        probability = torch.sigmoid(logits).cpu().numpy()
        prediction = (probability >= 0.5).astype(int)
        labels = batch["label"].cpu().numpy().astype(int)
        event_attention = attention["event"].cpu().numpy()
        encoder_mask = batch["encoder_mask"].cpu().numpy()
        summary_gate = attention.get("summary_gate")
        summary_gate_array = summary_gate.cpu().numpy() if summary_gate is not None else np.full(len(labels), np.nan)
        summary_alpha = attention.get("summary_alpha")
        summary_alpha_array = (
            summary_alpha.cpu().numpy() if summary_alpha is not None else np.full(len(labels), np.nan)
        )
        encoder_logit = attention.get("encoder_logit")
        encoder_logit_array = (
            encoder_logit.cpu().numpy() if encoder_logit is not None else np.full(len(labels), np.nan)
        )
        summary_logit = attention.get("summary_logit")
        summary_logit_array = (
            summary_logit.cpu().numpy() if summary_logit is not None else np.full(len(labels), np.nan)
        )
        for row_idx, subject_id in enumerate(batch["subject_id"]):
            base = {
                "split": split_name,
                "subject_id": subject_id,
                "label": int(labels[row_idx]),
                "probability": float(probability[row_idx]),
                "prediction": int(prediction[row_idx]),
                "summary_gate": float(summary_gate_array[row_idx]),
                "encoder_gate": float(1.0 - summary_gate_array[row_idx]) if fusion == "gated" else np.nan,
                "summary_alpha": float(summary_alpha_array[row_idx]),
                "encoder_logit": float(encoder_logit_array[row_idx]),
                "summary_logit": float(summary_logit_array[row_idx]),
            }
            for event_pos in range(int(encoder_mask[row_idx].sum())):
                rows.append(
                    {
                        **base,
                        "stream": "encoder_event",
                        "position": event_pos + 1,
                        "attention_weight": float(event_attention[row_idx, event_pos]),
                    }
                )
    return rows


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


def save_summary_encoder_dual_stream_outputs(
    output_dir: str | Path,
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    valid_threshold_metrics: pd.DataFrame,
    selected_thresholds: pd.DataFrame,
    training_log: pd.DataFrame,
    attention_weights: pd.DataFrame,
    run_info: dict[str, Any],
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_path / "metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(output_path / "predictions.csv", index=False, encoding="utf-8-sig")
    valid_threshold_metrics.to_csv(output_path / "valid_threshold_metrics.csv", index=False, encoding="utf-8-sig")
    selected_thresholds.to_csv(output_path / "selected_thresholds.csv", index=False, encoding="utf-8-sig")
    training_log.to_csv(output_path / "training_log.csv", index=False, encoding="utf-8-sig")
    attention_weights.to_csv(output_path / "attention_weights.csv", index=False, encoding="utf-8-sig")
    (output_path / "config.json").write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")


__all__ = [
    "SummaryEncoderDualStreamConfig",
    "load_split_subjects",
    "save_summary_encoder_dual_stream_outputs",
    "train_fixed_split",
]
