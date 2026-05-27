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
from eyenet.models.dual_stream import EncoderDualStreamConcatModel, EncoderDualStreamGatedModel
from eyenet.training.baseline import compute_metrics
from eyenet.training.dual_stream_concat import build_threshold_map
from eyenet.training.fixed_split_baseline import attach_fixed_split
from eyenet.training.segment_sequence import (
    build_subject_sequences,
    get_segment_feature_columns,
    prepare_fixed_split_data as prepare_macro_fixed_split_data,
    set_seed,
)
from eyenet.training.thresholds import analyze_thresholds, choose_thresholds


@dataclass(frozen=True)
class EncoderDualStreamConfig:
    batch_size: int = 8
    max_epochs: int = 100
    patience: int = 15
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    fusion: str = "gated"
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
    use_segment_position: bool = True
    max_seq_len: int | None = 1500
    gradient_clip_norm: float = 5.0
    pretrained_checkpoint: str | None = None
    freeze_encoder: bool = False


class EncoderDualStreamDataset(Dataset):
    def __init__(
        self,
        macro_arrays: dict[str, np.ndarray],
        macro_masks: dict[str, np.ndarray],
        encoder_arrays: dict[str, np.ndarray],
        encoder_masks: dict[str, np.ndarray],
        labels: dict[str, int],
        subject_ids: list[str],
    ) -> None:
        self.macro_arrays = macro_arrays
        self.macro_masks = macro_masks
        self.encoder_arrays = encoder_arrays
        self.encoder_masks = encoder_masks
        self.labels = labels
        self.subject_ids = subject_ids

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        subject_id = self.subject_ids[index]
        return {
            "macro_x": torch.tensor(self.macro_arrays[subject_id], dtype=torch.float32),
            "macro_mask": torch.tensor(self.macro_masks[subject_id], dtype=torch.bool),
            "encoder_x": torch.tensor(self.encoder_arrays[subject_id], dtype=torch.float32),
            "encoder_mask": torch.tensor(self.encoder_masks[subject_id], dtype=torch.bool),
            "label": torch.tensor(self.labels[subject_id], dtype=torch.float32),
            "subject_id": subject_id,
        }


def train_fixed_split(
    macro_segments: pd.DataFrame,
    encoder_events: pd.DataFrame,
    split_subjects: pd.DataFrame,
    encoder_feature_columns: list[str],
    cfg: EncoderDualStreamConfig,
    device: str | None = None,
    checkpoint_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    set_seed(cfg.random_seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = cfg_with_pretrained_architecture(cfg, device=device) if cfg.pretrained_checkpoint is not None else cfg

    macro_data = attach_fixed_split(macro_segments, split_path_from_frame(split_subjects))
    macro_feature_cols = get_segment_feature_columns(macro_data, use_segment_position=cfg.use_segment_position)
    train_macro, valid_macro, test_macro, macro_imputer, macro_scaler = prepare_macro_fixed_split_data(
        macro_data, macro_feature_cols
    )
    max_segments = int(
        max(
            train_macro.groupby("subject_id").size().max(),
            valid_macro.groupby("subject_id").size().max(),
            test_macro.groupby("subject_id").size().max(),
        )
    )

    encoder_split_subjects = align_split_dataset_id(encoder_events, split_subjects)
    encoder_split_subjects = add_subject_key(encoder_split_subjects)
    train_subjects = set(encoder_split_subjects.loc[encoder_split_subjects["split"] == "train", "_subject_key"])
    encoder_preprocessor = fit_encoder_preprocessor(
        encoder_events,
        encoder_feature_columns,
        train_subjects=train_subjects,
    )
    train_dataset = build_encoder_dual_dataset(
        train_macro,
        encoder_events,
        encoder_split_subjects,
        split_name="train",
        macro_feature_cols=macro_feature_cols,
        encoder_feature_cols=encoder_feature_columns,
        encoder_preprocessor=encoder_preprocessor,
        max_segments=max_segments,
        max_seq_len=cfg.max_seq_len,
    )
    valid_dataset = build_encoder_dual_dataset(
        valid_macro,
        encoder_events,
        encoder_split_subjects,
        split_name="valid",
        macro_feature_cols=macro_feature_cols,
        encoder_feature_cols=encoder_feature_columns,
        encoder_preprocessor=encoder_preprocessor,
        max_segments=max_segments,
        max_seq_len=cfg.max_seq_len,
    )
    test_dataset = build_encoder_dual_dataset(
        test_macro,
        encoder_events,
        encoder_split_subjects,
        split_name="test",
        macro_feature_cols=macro_feature_cols,
        encoder_feature_cols=encoder_feature_columns,
        encoder_preprocessor=encoder_preprocessor,
        max_segments=max_segments,
        max_seq_len=cfg.max_seq_len,
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    model = build_model(cfg, macro_input_dim=len(macro_feature_cols), encoder_input_dim=len(encoder_feature_columns)).to(device)
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
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, cfg.gradient_clip_norm)
        valid_loss, valid_true, valid_prob, _ = evaluate(model, valid_loader, criterion, device)
        valid_auc = roc_auc_score(valid_true, valid_prob)
        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "valid_auc": float(valid_auc),
                "max_segments": max_segments,
                "max_seq_len": cfg.max_seq_len,
            }
        )
        print(
            f"[encoder-dual-{cfg.fusion}] epoch {epoch:03d}/{cfg.max_epochs:03d} "
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
                "macro_imputer": macro_imputer,
                "macro_scaler": macro_scaler,
                "macro_feature_cols": macro_feature_cols,
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
                "macro_feature_cols": macro_feature_cols,
                "encoder_feature_cols": encoder_feature_columns,
                "preprocessor_path": "preprocessor.joblib",
                "pretrained_loaded": pretrained_loaded,
                "device": device,
            },
            checkpoint_path / "best.pt",
        )

    valid_loss, valid_true, valid_prob, valid_subjects = evaluate(model, valid_loader, criterion, device)
    test_loss, test_true, test_prob, test_subjects = evaluate(model, test_loader, criterion, device)
    model_name = f"encoder_dual_stream_{cfg.fusion}"
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
                    "max_segments": max_segments,
                    "max_seq_len": cfg.max_seq_len,
                    **metrics,
                }
            )

    attention_rows = collect_attention_weights(model, valid_loader, device, split_name="valid", fusion=cfg.fusion)
    attention_rows.extend(collect_attention_weights(model, test_loader, device, split_name="test", fusion=cfg.fusion))
    run_info = {
        "config": asdict(cfg),
        "device": device,
        "macro_feature_cols": macro_feature_cols,
        "encoder_feature_cols": encoder_feature_columns,
        "n_macro_features": len(macro_feature_cols),
        "n_encoder_features": len(encoder_feature_columns),
        "max_segments": max_segments,
        "max_seq_len": cfg.max_seq_len,
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "best_valid_auc": float(best_auc),
        "pretrained_loaded": pretrained_loaded,
        "freeze_encoder": cfg.freeze_encoder,
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


def split_path_from_frame(split_subjects: pd.DataFrame) -> Path:
    split_path = split_subjects.attrs.get("path")
    if split_path:
        return Path(split_path)
    raise ValueError("split_subjects must include attrs['path']; load it with load_split_subjects().")


def load_split_subjects(path: str | Path) -> pd.DataFrame:
    split = pd.read_csv(path, dtype={"subject_id": str})
    split["subject_id"] = split["subject_id"].astype(str).str.zfill(3)
    split.attrs["path"] = str(path)
    return split


def build_model(cfg: EncoderDualStreamConfig, macro_input_dim: int, encoder_input_dim: int):
    model_cls = EncoderDualStreamGatedModel if cfg.fusion == "gated" else EncoderDualStreamConcatModel
    return model_cls(
        macro_input_dim=macro_input_dim,
        encoder_input_dim=encoder_input_dim,
        encoder_type=cfg.encoder_type,
        projection_dim=cfg.projection_dim,
        hidden_dim=cfg.hidden_dim,
        attention_dim=cfg.attention_dim,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        feedforward_dim=cfg.feedforward_dim,
        dropout=cfg.dropout,
    )


def cfg_with_pretrained_architecture(cfg: EncoderDualStreamConfig, device: str) -> EncoderDualStreamConfig:
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
    return EncoderDualStreamConfig(**{**asdict(cfg), **updates})


def load_pretrained_event_encoder(event_encoder, checkpoint_path: str | Path, feature_columns: list[str], device: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_features = checkpoint.get("feature_columns")
    if checkpoint_features is not None and list(checkpoint_features) != list(feature_columns):
        raise ValueError(
            "Pretrained checkpoint feature columns do not match current feature columns. "
            "Use the same encoder-ready schema for pretraining and downstream fusion."
        )
    encoder_state = checkpoint.get("encoder_state_dict")
    if encoder_state is None:
        model_state = checkpoint.get("model_state_dict")
        if model_state is None:
            raise ValueError("Checkpoint does not contain encoder_state_dict or model_state_dict.")
        encoder_state = {
            key.removeprefix("encoder."): value for key, value in model_state.items() if key.startswith("encoder.")
        }
    if not encoder_state:
        raise ValueError("No encoder weights were found in the pretrained checkpoint.")
    event_encoder.load_state_dict(encoder_state, strict=True)


def build_encoder_dual_dataset(
    macro_segments: pd.DataFrame,
    encoder_events: pd.DataFrame,
    split_subjects: pd.DataFrame,
    split_name: str,
    macro_feature_cols: list[str],
    encoder_feature_cols: list[str],
    encoder_preprocessor,
    max_segments: int,
    max_seq_len: int | None,
) -> EncoderDualStreamDataset:
    macro_arrays, macro_masks, macro_labels, macro_subjects = build_subject_sequences(
        macro_segments,
        macro_feature_cols,
        max_segments=max_segments,
    )
    encoder_dataset = SubjectSequenceDataset(
        events=encoder_events,
        feature_columns=encoder_feature_cols,
        preprocessor=encoder_preprocessor,
        split_subjects=split_subjects,
        split_name=split_name,
        max_seq_len=max_seq_len,
        require_label=True,
    )
    macro_array_map = dict(zip(macro_subjects, macro_arrays))
    macro_mask_map = dict(zip(macro_subjects, macro_masks))
    macro_label_map = dict(zip(macro_subjects, macro_labels))
    encoder_max_len = int(max(sample["features"].shape[0] for sample in encoder_dataset.samples))
    encoder_array_map = {}
    encoder_mask_map = {}
    for sample in encoder_dataset.samples:
        features = sample["features"]
        padded = np.zeros((encoder_max_len, len(encoder_feature_cols)), dtype=np.float32)
        mask = np.zeros(encoder_max_len, dtype=bool)
        length = int(features.shape[0])
        padded[:length] = features
        mask[:length] = True
        encoder_array_map[sample["subject_id"]] = padded
        encoder_mask_map[sample["subject_id"]] = mask
    encoder_label_map = {sample["subject_id"]: sample["label"] for sample in encoder_dataset.samples}
    common_subjects = sorted(set(macro_subjects) & set(encoder_array_map))
    if len(common_subjects) != len(set(macro_subjects)) or len(common_subjects) != len(set(encoder_array_map)):
        raise ValueError("Macro and encoder streams do not cover the same subjects in this split.")
    for subject_id in common_subjects:
        if int(macro_label_map[subject_id]) != int(encoder_label_map[subject_id]):
            raise ValueError(f"Label mismatch between streams for subject {subject_id}.")
    return EncoderDualStreamDataset(
        macro_arrays=macro_array_map,
        macro_masks=macro_mask_map,
        encoder_arrays=encoder_array_map,
        encoder_masks=encoder_mask_map,
        labels=macro_label_map,
        subject_ids=common_subjects,
    )


def train_one_epoch(model, loader, optimizer, criterion, device: str, gradient_clip_norm: float) -> float:
    model.train()
    losses: list[float] = []
    for batch in loader:
        optimizer.zero_grad()
        logits, _ = model(
            batch["macro_x"].to(device),
            batch["macro_mask"].to(device),
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
            batch["macro_x"].to(device),
            batch["macro_mask"].to(device),
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
            batch["macro_x"].to(device),
            batch["macro_mask"].to(device),
            batch["encoder_x"].to(device),
            batch["encoder_mask"].to(device),
        )
        probability = torch.sigmoid(logits).cpu().numpy()
        prediction = (probability >= 0.5).astype(int)
        labels = batch["label"].cpu().numpy().astype(int)
        macro_attention = attention["macro"].cpu().numpy()
        event_attention = attention["event"].cpu().numpy()
        macro_mask = batch["macro_mask"].cpu().numpy()
        encoder_mask = batch["encoder_mask"].cpu().numpy()
        macro_gate = attention.get("macro_gate")
        macro_gate_array = macro_gate.cpu().numpy() if macro_gate is not None else np.full(len(labels), np.nan)
        for row_idx, subject_id in enumerate(batch["subject_id"]):
            base = {
                "split": split_name,
                "subject_id": subject_id,
                "label": int(labels[row_idx]),
                "probability": float(probability[row_idx]),
                "prediction": int(prediction[row_idx]),
                "macro_gate": float(macro_gate_array[row_idx]),
                "event_gate": float(1.0 - macro_gate_array[row_idx]) if fusion == "gated" else np.nan,
            }
            for segment_pos in range(int(macro_mask[row_idx].sum())):
                rows.append(
                    {
                        **base,
                        "stream": "macro",
                        "position": segment_pos + 1,
                        "attention_weight": float(macro_attention[row_idx, segment_pos]),
                    }
                )
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


def save_encoder_dual_stream_outputs(
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
