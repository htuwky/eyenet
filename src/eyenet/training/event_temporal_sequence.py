from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from eyenet.models.event_temporal_sequence import EventTemporalSequenceAttentionModel
from eyenet.training.baseline import compute_metrics
from eyenet.training.fixed_split_baseline import attach_fixed_split
from eyenet.training.segment_sequence import set_seed, summarize_across_seeds
from eyenet.training.thresholds import analyze_thresholds, choose_thresholds


@dataclass(frozen=True)
class EventTemporalSequenceConfig:
    batch_size: int = 8
    max_epochs: int = 100
    patience: int = 15
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    projection_dim: int = 64
    hidden_dim: int = 64
    attention_dim: int = 64
    dropout: float = 0.3
    random_seed: int = 42
    pos_weight: float = 1.0
    max_events: int | None = None
    gradient_clip_norm: float = 5.0


class EventTemporalSequenceDataset(Dataset):
    def __init__(
        self,
        arrays: list[np.ndarray],
        masks: list[np.ndarray],
        labels: list[int],
        subject_ids: list[str],
        event_indices: list[np.ndarray],
        segment_indices: list[np.ndarray],
    ) -> None:
        self.arrays = arrays
        self.masks = masks
        self.labels = labels
        self.subject_ids = subject_ids
        self.event_indices = event_indices
        self.segment_indices = segment_indices

    def __len__(self) -> int:
        return len(self.arrays)

    def __getitem__(self, index: int) -> dict:
        return {
            "x": torch.tensor(self.arrays[index], dtype=torch.float32),
            "mask": torch.tensor(self.masks[index], dtype=torch.bool),
            "label": torch.tensor(self.labels[index], dtype=torch.float32),
            "subject_id": self.subject_ids[index],
            "event_index": torch.tensor(self.event_indices[index], dtype=torch.long),
            "segment_index": torch.tensor(self.segment_indices[index], dtype=torch.long),
        }


def get_event_feature_columns(events: pd.DataFrame) -> list[str]:
    excluded = {
        "dataset_id",
        "subject_id",
        "split",
        "fold",
        "official_fold",
        "label",
        "segment_index",
        "subject_event_index",
        "event_index_in_segment",
    }
    return [col for col in events.columns if col not in excluded]


def prepare_fold_data(
    events: pd.DataFrame,
    valid_fold: str,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_events = events[(events["split"] == "train_valid") & (events["fold"] != valid_fold)].copy()
    valid_events = events[(events["split"] == "train_valid") & (events["fold"] == valid_fold)].copy()

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_values = scaler.fit_transform(imputer.fit_transform(train_events[feature_cols]))
    valid_values = scaler.transform(imputer.transform(valid_events[feature_cols]))
    train_scaled = pd.DataFrame(train_values, columns=feature_cols, index=train_events.index)
    valid_scaled = pd.DataFrame(valid_values, columns=feature_cols, index=valid_events.index)
    train_events = pd.concat([train_events.drop(columns=feature_cols), train_scaled], axis=1)
    valid_events = pd.concat([valid_events.drop(columns=feature_cols), valid_scaled], axis=1)
    return train_events, valid_events


def prepare_fixed_split_data(
    events: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SimpleImputer, StandardScaler]:
    train_events = events[events["split"] == "train"].copy()
    valid_events = events[events["split"] == "valid"].copy()
    test_events = events[events["split"] == "test"].copy()
    if train_events.empty or valid_events.empty or test_events.empty:
        raise ValueError("Fixed split must contain train, valid, and test events.")

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_values = scaler.fit_transform(imputer.fit_transform(train_events[feature_cols]))
    valid_values = scaler.transform(imputer.transform(valid_events[feature_cols]))
    test_values = scaler.transform(imputer.transform(test_events[feature_cols]))

    train_scaled = pd.DataFrame(train_values, columns=feature_cols, index=train_events.index)
    valid_scaled = pd.DataFrame(valid_values, columns=feature_cols, index=valid_events.index)
    test_scaled = pd.DataFrame(test_values, columns=feature_cols, index=test_events.index)
    train_events = pd.concat([train_events.drop(columns=feature_cols), train_scaled], axis=1)
    valid_events = pd.concat([valid_events.drop(columns=feature_cols), valid_scaled], axis=1)
    test_events = pd.concat([test_events.drop(columns=feature_cols), test_scaled], axis=1)
    return train_events, valid_events, test_events, imputer, scaler


def build_subject_event_sequences(events: pd.DataFrame, feature_cols: list[str], max_events: int | None = None):
    grouped = list(events.groupby("subject_id", sort=True))
    if max_events is None:
        max_events = max(len(group) for _, group in grouped)

    arrays: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    labels: list[int] = []
    subject_ids: list[str] = []
    event_indices: list[np.ndarray] = []
    segment_indices: list[np.ndarray] = []

    for subject_id, group in grouped:
        group = group.sort_values("subject_event_index")
        values = group[feature_cols].to_numpy(dtype=np.float32)
        raw_event_indices = group["subject_event_index"].to_numpy(dtype=np.int64)
        raw_segment_indices = group["segment_index"].to_numpy(dtype=np.int64)
        label_values = group["label"].dropna()
        if label_values.empty:
            raise ValueError(f"Subject {subject_id} has no non-missing label for event sequence training.")
        label = int(label_values.iloc[0])
        n_events = min(len(values), max_events)

        padded = np.zeros((max_events, len(feature_cols)), dtype=np.float32)
        mask = np.zeros(max_events, dtype=bool)
        padded_events = np.zeros(max_events, dtype=np.int64)
        padded_segments = np.zeros(max_events, dtype=np.int64)
        padded[:n_events] = values[:n_events]
        mask[:n_events] = True
        padded_events[:n_events] = raw_event_indices[:n_events]
        padded_segments[:n_events] = raw_segment_indices[:n_events]

        arrays.append(padded)
        masks.append(mask)
        labels.append(label)
        subject_ids.append(subject_id)
        event_indices.append(padded_events)
        segment_indices.append(padded_segments)

    return arrays, masks, labels, subject_ids, event_indices, segment_indices


def train_fixed_split(
    events: pd.DataFrame,
    split_path: str | Path,
    cfg: EventTemporalSequenceConfig,
    device: str | None = None,
    checkpoint_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    set_seed(cfg.random_seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    data = attach_fixed_split(events, split_path)
    feature_cols = get_event_feature_columns(data)
    train_events, valid_events, test_events, imputer, scaler = prepare_fixed_split_data(data, feature_cols)

    max_events = int(
        max(
            train_events.groupby("subject_id").size().max(),
            valid_events.groupby("subject_id").size().max(),
            test_events.groupby("subject_id").size().max(),
        )
    )
    if cfg.max_events is not None:
        max_events = min(max_events, int(cfg.max_events))

    train_data = build_subject_event_sequences(train_events, feature_cols, max_events=max_events)
    valid_data = build_subject_event_sequences(valid_events, feature_cols, max_events=max_events)
    test_data = build_subject_event_sequences(test_events, feature_cols, max_events=max_events)

    train_loader = DataLoader(EventTemporalSequenceDataset(*train_data), batch_size=cfg.batch_size, shuffle=True)
    valid_loader = DataLoader(EventTemporalSequenceDataset(*valid_data), batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(EventTemporalSequenceDataset(*test_data), batch_size=cfg.batch_size, shuffle=False)

    model = EventTemporalSequenceAttentionModel(
        input_dim=len(feature_cols),
        projection_dim=cfg.projection_dim,
        hidden_dim=cfg.hidden_dim,
        attention_dim=cfg.attention_dim,
        dropout=cfg.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg.pos_weight, dtype=torch.float32, device=device))

    log_rows: list[dict] = []
    best_auc = -np.inf
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_epoch = cfg.max_epochs

    for epoch in range(1, cfg.max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, cfg.gradient_clip_norm)
        valid_loss, y_true, y_prob, _ = evaluate(model, valid_loader, criterion, device)
        valid_auc = roc_auc_score(y_true, y_prob)
        log_rows.append(
            {
                "split": "valid",
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "valid_auc": float(valid_auc),
                "max_events": int(max_events),
            }
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
                "imputer": imputer,
                "scaler": scaler,
                "feature_cols": feature_cols,
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
                "feature_cols": feature_cols,
                "preprocessor_path": "preprocessor.joblib",
                "device": device,
                "split_path": str(split_path),
            },
            checkpoint_path / "best.pt",
        )

    _, valid_true, valid_prob, valid_subjects = evaluate(model, valid_loader, criterion, device)
    _, test_true, test_prob, test_subjects = evaluate(model, test_loader, criterion, device)

    valid_predictions_default = make_deep_prediction_frame("valid", valid_subjects, valid_true, valid_prob, 0.5)
    valid_threshold_metrics = analyze_thresholds(valid_predictions_default)
    selected_thresholds = choose_thresholds(valid_threshold_metrics)

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
    threshold_map = {
        "default_0.50": 0.5,
        "valid_best_balanced_accuracy": float(best_balanced["threshold"]),
        "valid_best_f1": float(best_f1["threshold"]),
        "valid_screening_sensitivity_at_least_0.80": screening_threshold,
    }

    prediction_rows: list[dict] = []
    metric_rows: list[dict] = []
    for split_name, labels, probabilities, subject_ids in [
        ("valid", valid_true, valid_prob, valid_subjects),
        ("test", test_true, test_prob, test_subjects),
    ]:
        for threshold_name, threshold in threshold_map.items():
            frame = make_deep_prediction_frame(split_name, subject_ids, labels, probabilities, threshold)
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
                    "model": "event_temporal_sequence_bigru_attention",
                    "split": split_name,
                    "threshold_name": threshold_name,
                    "threshold": threshold,
                    "best_epoch": best_epoch,
                    "stopped_epoch": stopped_epoch,
                    "best_valid_auc": float(best_auc),
                    "max_events": int(max_events),
                    **metrics,
                }
            )

    attention_rows = collect_attention_weights(model, valid_loader, device)
    for row in attention_rows:
        row["split"] = "valid"
    test_attention_rows = collect_attention_weights(model, test_loader, device)
    for row in test_attention_rows:
        row["split"] = "test"
    attention_rows.extend(test_attention_rows)

    run_info = {
        "config": asdict(cfg),
        "device": device,
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "split_path": str(split_path),
        "max_events": int(max_events),
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "best_valid_auc": float(best_auc),
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


def train_official_folds(
    events: pd.DataFrame,
    cfg: EventTemporalSequenceConfig,
    device: str | None = None,
    checkpoint_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    set_seed(cfg.random_seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    feature_cols = get_event_feature_columns(events)
    folds = sorted(events.loc[events["split"] == "train_valid", "fold"].unique())

    metric_rows: list[dict] = []
    prediction_rows: list[dict] = []
    log_rows: list[dict] = []
    attention_rows: list[dict] = []
    checkpoint_path = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if checkpoint_path is not None:
        checkpoint_path.mkdir(parents=True, exist_ok=True)

    for fold in folds:
        train_events, valid_events = prepare_fold_data(events, fold, feature_cols)
        fold_max_events = max(
            train_events.groupby("subject_id").size().max(),
            valid_events.groupby("subject_id").size().max(),
        )
        if cfg.max_events is not None:
            fold_max_events = min(fold_max_events, cfg.max_events)

        train_data = build_subject_event_sequences(train_events, feature_cols, max_events=int(fold_max_events))
        valid_data = build_subject_event_sequences(valid_events, feature_cols, max_events=int(fold_max_events))

        train_loader = DataLoader(
            EventTemporalSequenceDataset(*train_data),
            batch_size=cfg.batch_size,
            shuffle=True,
        )
        valid_loader = DataLoader(
            EventTemporalSequenceDataset(*valid_data),
            batch_size=cfg.batch_size,
            shuffle=False,
        )

        model = EventTemporalSequenceAttentionModel(
            input_dim=len(feature_cols),
            projection_dim=cfg.projection_dim,
            hidden_dim=cfg.hidden_dim,
            attention_dim=cfg.attention_dim,
            dropout=cfg.dropout,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg.pos_weight, dtype=torch.float32, device=device))

        best_auc = -np.inf
        best_state = None
        best_epoch = 0
        epochs_without_improvement = 0
        stopped_epoch = cfg.max_epochs

        for epoch in range(1, cfg.max_epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, cfg.gradient_clip_norm)
            valid_loss, y_true, y_prob, _ = evaluate(model, valid_loader, criterion, device)
            valid_auc = roc_auc_score(y_true, y_prob)
            log_rows.append(
                {
                    "fold": fold,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "valid_auc": float(valid_auc),
                    "max_events": int(fold_max_events),
                }
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
        if checkpoint_path is not None:
            torch.save(
                {
                    "fold": fold,
                    "best_epoch": best_epoch,
                    "best_auc": float(best_auc),
                    "model_state_dict": model.state_dict(),
                    "config": asdict(cfg),
                    "feature_cols": feature_cols,
                    "device": device,
                },
                checkpoint_path / f"fold_{fold}_best.pt",
            )

        _, y_true, y_prob, subject_ids = evaluate(model, valid_loader, criterion, device)
        y_pred = (np.asarray(y_prob) >= 0.5).astype(int)
        metrics = compute_metrics(np.asarray(y_true), y_pred, np.asarray(y_prob))
        metric_rows.append(
            {
                "fold": fold,
                "best_epoch": best_epoch,
                "stopped_epoch": stopped_epoch,
                "best_valid_auc": float(best_auc),
                "max_events": int(fold_max_events),
                **metrics,
            }
        )
        for subject_id, label, prob, pred in zip(subject_ids, y_true, y_prob, y_pred):
            prediction_rows.append(
                {
                    "fold": fold,
                    "subject_id": subject_id,
                    "label": int(label),
                    "probability": float(prob),
                    "prediction": int(pred),
                }
            )
        attention_rows.extend(collect_attention_weights(model, valid_loader, device))

    run_info = {
        "config": asdict(cfg),
        "device": device,
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "folds": folds,
    }
    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(prediction_rows),
        pd.DataFrame(log_rows),
        pd.DataFrame(attention_rows),
        run_info,
    )


def make_deep_prediction_frame(
    split_name: str,
    subject_ids: list[str],
    labels: list[int],
    probabilities: list[float],
    threshold: float,
) -> pd.DataFrame:
    probabilities_array = np.asarray(probabilities, dtype=float)
    return pd.DataFrame(
        {
            "model": "event_temporal_sequence_bigru_attention",
            "split": split_name,
            "subject_id": subject_ids,
            "label": np.asarray(labels, dtype=int),
            "probability": probabilities_array,
            "prediction": (probabilities_array >= threshold).astype(int),
        }
    )


def train_one_epoch(model, loader, optimizer, criterion, device: str, gradient_clip_norm: float) -> float:
    model.train()
    losses: list[float] = []
    for batch in loader:
        optimizer.zero_grad()
        x = batch["x"].to(device)
        mask = batch["mask"].to(device)
        label = batch["label"].to(device)
        logits, _ = model(x, mask)
        loss = criterion(logits, label)
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
        x = batch["x"].to(device)
        mask = batch["mask"].to(device)
        label = batch["label"].to(device)
        logits, _ = model(x, mask)
        loss = criterion(logits, label)
        prob = torch.sigmoid(logits)
        losses.append(float(loss.detach().cpu()))
        y_true.extend(label.cpu().numpy().astype(int).tolist())
        y_prob.extend(prob.cpu().numpy().tolist())
        subject_ids.extend(batch["subject_id"])
    return float(np.mean(losses)), y_true, y_prob, subject_ids


@torch.no_grad()
def collect_attention_weights(model, loader, device: str) -> list[dict]:
    model.eval()
    rows: list[dict] = []
    for batch in loader:
        x = batch["x"].to(device)
        mask = batch["mask"].to(device)
        label = batch["label"].cpu().numpy().astype(int)
        logits, attention = model(x, mask)
        probability = torch.sigmoid(logits).cpu().numpy()
        prediction = (probability >= 0.5).astype(int)
        attention_np = attention.cpu().numpy()
        mask_np = batch["mask"].cpu().numpy()
        event_index_np = batch["event_index"].cpu().numpy()
        segment_index_np = batch["segment_index"].cpu().numpy()
        for row_idx, subject_id in enumerate(batch["subject_id"]):
            valid_count = int(mask_np[row_idx].sum())
            for pos in range(valid_count):
                rows.append(
                    {
                        "subject_id": subject_id,
                        "label": int(label[row_idx]),
                        "probability": float(probability[row_idx]),
                        "prediction": int(prediction[row_idx]),
                        "sequence_position": pos + 1,
                        "subject_event_index": int(event_index_np[row_idx, pos]),
                        "segment_index": int(segment_index_np[row_idx, pos]),
                        "attention_weight": float(attention_np[row_idx, pos]),
                    }
                )
    return rows


def summarize_deep_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["auc", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1"]
    row = {"model": "event_temporal_sequence_bigru_attention"}
    for metric in metric_cols:
        row[f"{metric}_mean"] = float(fold_metrics[metric].mean())
        row[f"{metric}_std"] = float(fold_metrics[metric].std())
    return pd.DataFrame([row])


def save_event_temporal_sequence_outputs(
    output_dir: str | Path,
    fold_metrics: pd.DataFrame,
    summary_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    training_log: pd.DataFrame,
    attention_weights: pd.DataFrame,
    run_info: dict,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    fold_metrics.to_csv(output_path / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    summary_metrics.to_csv(output_path / "summary_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(output_path / "predictions.csv", index=False, encoding="utf-8-sig")
    training_log.to_csv(output_path / "training_log.csv", index=False, encoding="utf-8-sig")
    attention_weights.to_csv(output_path / "attention_weights.csv", index=False, encoding="utf-8-sig")
    (output_path / "config.json").write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")


def save_fixed_split_event_temporal_sequence_outputs(
    output_dir: str | Path,
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    valid_threshold_metrics: pd.DataFrame,
    selected_thresholds: pd.DataFrame,
    training_log: pd.DataFrame,
    attention_weights: pd.DataFrame,
    run_info: dict,
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
    "EventTemporalSequenceConfig",
    "train_fixed_split",
    "train_official_folds",
    "summarize_deep_metrics",
    "summarize_across_seeds",
    "save_event_temporal_sequence_outputs",
    "save_fixed_split_event_temporal_sequence_outputs",
]
