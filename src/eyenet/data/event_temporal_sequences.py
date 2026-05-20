from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


EVENT_TEMPORAL_FEATURE_COLUMNS = [
    "subject_event_index_norm",
    "x_norm",
    "y_norm",
    "x_dva",
    "y_dva",
    "duration_ms",
    "log_duration_ms",
    "saccade_dx_norm",
    "saccade_dy_norm",
    "saccade_amplitude_norm",
    "saccade_dx_dva",
    "saccade_dy_dva",
    "saccade_amplitude_dva",
    "saccade_angle_sin",
    "saccade_angle_cos",
    "transition_velocity_dva_s_approx",
    "log_transition_velocity_dva_s",
    "transition_missing",
    "is_first_event_in_segment",
    "is_last_event_in_segment",
    "segment_index_norm",
    "event_index_in_segment_norm",
]


def build_event_temporal_sequences(events: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    data = events.copy()
    data["subject_id"] = data["subject_id"].astype(str).str.zfill(3)
    data = data[(data["split"] == "train_valid") & (data["event_type"] == "fixation")].copy()
    data = data.sort_values(["subject_id", "trial_id", "event_index"]).reset_index(drop=True)

    segment_codes = (
        data[["subject_id", "trial_id"]]
        .drop_duplicates()
        .assign(segment_index=lambda df: df.groupby("subject_id").cumcount() + 1)
    )
    data = data.drop(columns=["segment_index"], errors="ignore").merge(
        segment_codes,
        on=["subject_id", "trial_id"],
        how="left",
    )
    data["subject_event_index"] = data.groupby("subject_id").cumcount() + 1
    data["event_index_in_segment"] = data.groupby(["subject_id", "segment_index"]).cumcount() + 1

    segment_sizes = data.groupby(["subject_id", "segment_index"])["event_index_in_segment"].transform("max")
    subject_event_sizes = data.groupby("subject_id")["subject_event_index"].transform("max")
    subject_segment_sizes = data.groupby("subject_id")["segment_index"].transform("max")

    data["is_first_event_in_segment"] = (data["event_index_in_segment"] == 1).astype(int)
    data["is_last_event_in_segment"] = (data["event_index_in_segment"] == segment_sizes).astype(int)
    data["segment_index_norm"] = safe_ratio(data["segment_index"] - 1, subject_segment_sizes - 1)
    data["event_index_in_segment_norm"] = safe_ratio(data["event_index_in_segment"] - 1, segment_sizes - 1)
    data["subject_event_index_norm"] = safe_ratio(data["subject_event_index"] - 1, subject_event_sizes - 1)

    data["transition_missing"] = data["saccade_amplitude_dva"].isna().astype(int)
    fill_zero_cols = [
        "saccade_dx_norm",
        "saccade_dy_norm",
        "saccade_amplitude_norm",
        "saccade_dx_dva",
        "saccade_dy_dva",
        "saccade_amplitude_dva",
        "transition_velocity_dva_s_approx",
    ]
    for col in fill_zero_cols:
        data[col] = data[col].fillna(0.0)

    data["saccade_angle"] = data["saccade_angle"].fillna(0.0)
    data["saccade_angle_sin"] = np.sin(data["saccade_angle"].to_numpy(dtype=float))
    data["saccade_angle_cos"] = np.cos(data["saccade_angle"].to_numpy(dtype=float))
    data["duration_ms"] = data["duration_ms"].clip(lower=1)
    data["log_duration_ms"] = np.log1p(data["duration_ms"])
    data["transition_velocity_dva_s_approx"] = data["transition_velocity_dva_s_approx"].clip(lower=0)
    data["log_transition_velocity_dva_s"] = np.log1p(data["transition_velocity_dva_s_approx"])

    output_columns = [
        "dataset_id",
        "subject_id",
        "split",
        "fold",
        "label",
        "segment_index",
        "subject_event_index",
        "event_index_in_segment",
        *EVENT_TEMPORAL_FEATURE_COLUMNS,
    ]
    event_temporal = data[output_columns].copy()
    summary = summarize_event_temporal_sequences(event_temporal)
    return event_temporal, summary


def summarize_event_temporal_sequences(event_temporal: pd.DataFrame) -> dict:
    subject_counts = event_temporal.groupby("subject_id").size()
    segment_counts = event_temporal.groupby("subject_id")["segment_index"].nunique()
    events_per_segment = event_temporal.groupby(["subject_id", "segment_index"]).size()
    label_counts = event_temporal.drop_duplicates("subject_id")["label"].value_counts().sort_index()
    transition_missing_rate = float(event_temporal["transition_missing"].mean())
    return {
        "n_subjects": int(subject_counts.size),
        "n_events": int(len(event_temporal)),
        "n_features": int(len(EVENT_TEMPORAL_FEATURE_COLUMNS)),
        "label_counts": {str(int(k)): int(v) for k, v in label_counts.items()},
        "events_per_subject": describe_series(subject_counts),
        "segments_per_subject": describe_series(segment_counts),
        "events_per_segment": describe_series(events_per_segment),
        "transition_missing_rate": transition_missing_rate,
        "feature_columns": EVENT_TEMPORAL_FEATURE_COLUMNS,
        "content_fields_used_as_features": [],
        "content_boundary_fields": ["segment_index"],
        "stream_name": "event_temporal_stream",
        "stream_description": "Fixation/saccade event-level temporal stream; not a frequency-domain stream.",
    }


def save_event_temporal_sequence_outputs(output_csv: str | Path, event_temporal: pd.DataFrame, summary: dict) -> None:
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    event_temporal.to_csv(output_path, index=False, encoding="utf-8-sig")
    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    result = numerator / denominator
    return result.fillna(0.0)


def describe_series(series: pd.Series) -> dict:
    values = series.to_numpy(dtype=float)
    return {
        "min": float(np.min(values)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }
