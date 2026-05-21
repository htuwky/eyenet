from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_feature_schema(path: str | Path) -> dict[str, Any]:
    schema_path = Path(path)
    if not schema_path.exists():
        raise FileNotFoundError(f"Feature schema does not exist: {schema_path}")
    return json.loads(schema_path.read_text(encoding="utf-8"))


def build_encoder_ready_table(events: pd.DataFrame, schema: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    metadata_cols = list(schema["metadata_columns"])
    feature_cols = list(schema["feature_columns"])
    required_cols = metadata_cols + feature_cols
    missing = [column for column in required_cols if column not in events.columns]
    if missing:
        raise ValueError(f"Input table is missing required encoder columns: {missing}")

    excluded_present = [column for column in schema.get("excluded_columns", []) if column in feature_cols]
    if excluded_present:
        raise ValueError(f"Excluded columns cannot be used as features: {excluded_present}")

    data = events[required_cols].copy()
    data["subject_id"] = data["subject_id"].astype(str).str.zfill(3)
    data = data.sort_values(["subject_id", "segment_index", "event_index_in_segment"]).reset_index(drop=True)
    for column in feature_cols:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data[feature_cols] = data[feature_cols].replace([np.inf, -np.inf], np.nan)

    summary = summarize_encoder_ready_table(data, schema)
    return data, summary


def summarize_encoder_ready_table(data: pd.DataFrame, schema: dict[str, Any]) -> dict[str, Any]:
    feature_cols = list(schema["feature_columns"])
    subject_counts = data.groupby("subject_id").size()
    segment_counts = data.groupby("subject_id")["segment_index"].nunique()
    events_per_segment = data.groupby(["subject_id", "segment_index"]).size()
    label_counts = data.drop_duplicates("subject_id")["label"].value_counts(dropna=False).sort_index()
    null_counts = data[feature_cols].isna().sum()
    null_rates = (null_counts / len(data)).to_dict()
    return {
        "schema_name": schema.get("name"),
        "schema_version": schema.get("version"),
        "n_subjects": int(subject_counts.size),
        "n_events": int(len(data)),
        "n_features": int(len(feature_cols)),
        "feature_columns": feature_cols,
        "metadata_columns": list(schema["metadata_columns"]),
        "label_counts": {str(key): int(value) for key, value in label_counts.items()},
        "events_per_subject": describe_series(subject_counts),
        "segments_per_subject": describe_series(segment_counts),
        "events_per_segment": describe_series(events_per_segment),
        "feature_null_counts": {column: int(null_counts[column]) for column in feature_cols},
        "feature_null_rates": {column: float(null_rates[column]) for column in feature_cols},
        "features_with_nulls": [column for column in feature_cols if int(null_counts[column]) > 0],
        "content_fields_used_as_features": [],
        "excluded_position_features": [
            column
            for column in schema.get("excluded_columns", [])
            if column in {"subject_event_index_norm", "segment_index_norm", "subject_event_index"}
        ],
    }


def save_encoder_ready_outputs(
    output_dir: str | Path,
    table: pd.DataFrame,
    summary: dict[str, Any],
    schema: dict[str, Any],
    table_name: str = "ems_encoder_events.csv",
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path / table_name, index=False, encoding="utf-8-sig")
    (output_path / "encoder_ready_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_path / "feature_schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def describe_series(series: pd.Series) -> dict[str, float]:
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
