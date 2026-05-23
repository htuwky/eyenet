from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import WeightedRandomSampler


@dataclass(frozen=True)
class EncoderPreprocessor:
    feature_columns: list[str]
    imputer: SimpleImputer
    scaler: StandardScaler

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame[self.feature_columns].to_numpy(dtype=np.float32)
        values = self.imputer.transform(values)
        values = self.scaler.transform(values)
        return values.astype(np.float32)

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)


def normalize_subject_ids(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["subject_id"] = out["subject_id"].astype(str).str.zfill(3)
    if "dataset_id" in out.columns:
        out["dataset_id"] = out["dataset_id"].astype(str)
    return out


def add_subject_key(frame: pd.DataFrame) -> pd.DataFrame:
    out = normalize_subject_ids(frame)
    if "dataset_id" in out.columns:
        out["_subject_key"] = out["dataset_id"].astype(str) + "::" + out["subject_id"].astype(str)
    else:
        out["_subject_key"] = out["subject_id"].astype(str)
    return out


def align_split_dataset_id(events: pd.DataFrame, split_subjects: pd.DataFrame) -> pd.DataFrame:
    """Backfill dataset_id for legacy single-dataset split files."""
    split = split_subjects.copy()
    if "dataset_id" in split.columns or "dataset_id" not in events.columns:
        return split
    dataset_ids = events["dataset_id"].dropna().astype(str).unique()
    if len(dataset_ids) == 1:
        split["dataset_id"] = dataset_ids[0]
    return split


def fit_encoder_preprocessor(events: pd.DataFrame, feature_columns: list[str], train_subjects: set[str]) -> EncoderPreprocessor:
    data = events.copy()
    data = add_subject_key(data)
    train = data[data["_subject_key"].isin(train_subjects)].copy()
    if train.empty:
        raise ValueError("No train rows available for encoder preprocessor fitting.")
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_values = train[feature_columns].to_numpy(dtype=np.float32)
    imputed = imputer.fit_transform(train_values)
    scaler.fit(imputed)
    return EncoderPreprocessor(feature_columns=feature_columns, imputer=imputer, scaler=scaler)


class SubjectSequenceDataset(Dataset):
    def __init__(
        self,
        events: pd.DataFrame,
        feature_columns: list[str],
        preprocessor: EncoderPreprocessor,
        split_subjects: pd.DataFrame,
        split_name: str,
        max_seq_len: int | None = None,
        require_label: bool = True,
    ) -> None:
        self.feature_columns = feature_columns
        self.preprocessor = preprocessor
        self.max_seq_len = max_seq_len
        self.require_label = require_label
        split_subjects = align_split_dataset_id(events, split_subjects)
        split_subjects = add_subject_key(split_subjects)
        subject_keys = set(split_subjects.loc[split_subjects["split"] == split_name, "_subject_key"])
        data = add_subject_key(events)
        data = data[data["_subject_key"].isin(subject_keys)].copy()
        data = data.sort_values(["dataset_id", "subject_id", "segment_index", "event_index_in_segment"]).reset_index(drop=True)
        if data.empty:
            raise ValueError(f"No encoder events found for split: {split_name}")

        self.samples: list[dict[str, Any]] = []
        for _, subject_df in data.groupby("_subject_key", sort=True):
            subject_id = str(subject_df["subject_id"].iloc[0])
            features = self.preprocessor.transform(subject_df)
            if self.max_seq_len is not None and len(features) > self.max_seq_len:
                features = features[: self.max_seq_len]
            label_values = pd.to_numeric(subject_df["label"], errors="coerce").dropna()
            if label_values.empty:
                if self.require_label:
                    raise ValueError(
                        f"Subject {subject_id} in split {split_name} has no label. "
                        "Use require_label=False for self-supervised encoder pretraining."
                    )
                label = -1
            else:
                label = int(label_values.iloc[0])
            dataset_id = str(subject_df["dataset_id"].dropna().iloc[0])
            self.samples.append(
                {
                    "subject_id": subject_id,
                    "dataset_id": dataset_id,
                    "label": label,
                    "features": features,
                    "length": int(len(features)),
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]


def collate_subject_sequences(batch: list[dict[str, Any]]) -> dict[str, Any]:
    batch_size = len(batch)
    max_len = max(item["length"] for item in batch)
    feature_dim = int(batch[0]["features"].shape[1])
    x = torch.zeros((batch_size, max_len, feature_dim), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    labels = torch.zeros(batch_size, dtype=torch.long)
    subject_ids: list[str] = []
    dataset_ids: list[str] = []

    for index, item in enumerate(batch):
        length = item["length"]
        x[index, :length] = torch.from_numpy(item["features"])
        mask[index, :length] = True
        labels[index] = int(item["label"])
        subject_ids.append(str(item["subject_id"]))
        dataset_ids.append(str(item["dataset_id"]))

    return {
        "x": x,
        "mask": mask,
        "label": labels,
        "subject_id": subject_ids,
        "dataset_id": dataset_ids,
        "length": torch.tensor([item["length"] for item in batch], dtype=torch.long),
    }


def build_encoder_dataloaders(
    events: pd.DataFrame,
    split_subjects: pd.DataFrame,
    feature_columns: list[str],
    batch_size: int = 8,
    max_seq_len: int | None = None,
    num_workers: int = 0,
    balanced_train_sampler: bool = False,
    require_label: bool = True,
) -> tuple[dict[str, DataLoader], EncoderPreprocessor]:
    split_subjects = align_split_dataset_id(events, split_subjects)
    split_subjects = add_subject_key(split_subjects)
    train_subjects = set(split_subjects.loc[split_subjects["split"] == "train", "_subject_key"])
    preprocessor = fit_encoder_preprocessor(events, feature_columns, train_subjects=train_subjects)
    loaders: dict[str, DataLoader] = {}
    for split_name in ["train", "valid", "test"]:
        dataset = SubjectSequenceDataset(
            events=events,
            feature_columns=feature_columns,
            preprocessor=preprocessor,
            split_subjects=split_subjects,
            split_name=split_name,
            max_seq_len=max_seq_len,
            require_label=require_label,
        )
        sampler = None
        shuffle = split_name == "train"
        if split_name == "train" and balanced_train_sampler:
            if not require_label:
                raise ValueError("balanced_train_sampler requires labels and cannot be used with require_label=False.")
            labels = np.array([sample["label"] for sample in dataset.samples], dtype=int)
            class_counts = np.bincount(labels)
            sample_weights = np.array([1.0 / class_counts[label] for label in labels], dtype=np.float64)
            sampler = WeightedRandomSampler(
                weights=torch.as_tensor(sample_weights, dtype=torch.double),
                num_samples=len(sample_weights),
                replacement=True,
            )
            shuffle = False
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_subject_sequences,
        )
    return loaders, preprocessor
