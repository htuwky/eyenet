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


def fit_encoder_preprocessor(events: pd.DataFrame, feature_columns: list[str], train_subjects: set[str]) -> EncoderPreprocessor:
    data = events.copy()
    data["subject_id"] = data["subject_id"].astype(str).str.zfill(3)
    train = data[data["subject_id"].isin(train_subjects)].copy()
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
    ) -> None:
        self.feature_columns = feature_columns
        self.preprocessor = preprocessor
        self.max_seq_len = max_seq_len
        split_subjects = split_subjects.copy()
        split_subjects["subject_id"] = split_subjects["subject_id"].astype(str).str.zfill(3)
        subject_ids = set(split_subjects.loc[split_subjects["split"] == split_name, "subject_id"])
        data = events.copy()
        data["subject_id"] = data["subject_id"].astype(str).str.zfill(3)
        data = data[data["subject_id"].isin(subject_ids)].copy()
        data = data.sort_values(["subject_id", "segment_index", "event_index_in_segment"]).reset_index(drop=True)
        if data.empty:
            raise ValueError(f"No encoder events found for split: {split_name}")

        self.samples: list[dict[str, Any]] = []
        for subject_id, subject_df in data.groupby("subject_id", sort=True):
            features = self.preprocessor.transform(subject_df)
            if self.max_seq_len is not None and len(features) > self.max_seq_len:
                features = features[: self.max_seq_len]
            label = int(pd.to_numeric(subject_df["label"], errors="coerce").dropna().iloc[0])
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
) -> tuple[dict[str, DataLoader], EncoderPreprocessor]:
    split_subjects = split_subjects.copy()
    split_subjects["subject_id"] = split_subjects["subject_id"].astype(str).str.zfill(3)
    train_subjects = set(split_subjects.loc[split_subjects["split"] == "train", "subject_id"])
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
        )
        sampler = None
        shuffle = split_name == "train"
        if split_name == "train" and balanced_train_sampler:
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
