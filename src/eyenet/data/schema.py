from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_EVENT_COLUMNS = [
    "dataset_id",
    "subject_id",
    "label",
    "trial_id",
    "event_index",
    "event_type",
    "x_norm",
    "y_norm",
    "duration_ms",
]

RECOMMENDED_EVENT_COLUMNS = [
    "split",
    "fold",
    "x_dva",
    "y_dva",
    "saccade_dx_norm",
    "saccade_dy_norm",
    "saccade_amplitude_norm",
    "saccade_angle",
    "saccade_dx_dva",
    "saccade_dy_dva",
    "saccade_amplitude_dva",
    "transition_velocity_dva_s_approx",
]

OPTIONAL_EVENT_COLUMNS = [
    "session_id",
    "task_id",
    "timestamp_start_ms",
    "timestamp_end_ms",
    "pupil_optional",
    "age",
    "sex",
    "diagnosis",
    "validity",
]

NUMERIC_COLUMNS = [
    "label",
    "event_index",
    "x_norm",
    "y_norm",
    "duration_ms",
    "x_dva",
    "y_dva",
    "saccade_dx_norm",
    "saccade_dy_norm",
    "saccade_amplitude_norm",
    "saccade_angle",
    "saccade_dx_dva",
    "saccade_dy_dva",
    "saccade_amplitude_dva",
    "transition_velocity_dva_s_approx",
]

NORM_COLUMNS = ["x_norm", "y_norm"]


@dataclass
class SchemaValidationReport:
    table_path: str
    n_rows: int
    n_columns: int
    n_subjects: int | None
    dataset_ids: list[str]
    label_counts: dict[str, int]
    missing_required_columns: list[str]
    missing_recommended_columns: list[str]
    present_optional_columns: list[str]
    columns_with_nulls: dict[str, int]
    structural_errors: list[str]
    quality_warnings: list[str]
    non_numeric_columns: list[str]
    normalized_coordinate_issues: dict[str, int]
    event_type_counts: dict[str, int]
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_event_table(
    path: str | Path,
    require_label: bool = False,
    strict_normalized_range: bool = False,
) -> SchemaValidationReport:
    table_path = Path(path)
    if not table_path.exists():
        raise FileNotFoundError(f"Event table does not exist: {table_path}")

    df = pd.read_csv(table_path, low_memory=False)
    columns = set(df.columns)
    missing_required = [column for column in REQUIRED_EVENT_COLUMNS if column not in columns]
    missing_recommended = [column for column in RECOMMENDED_EVENT_COLUMNS if column not in columns]
    present_optional = [column for column in OPTIONAL_EVENT_COLUMNS if column in columns]

    columns_with_nulls = {
        column: int(count)
        for column, count in df.isna().sum().items()
        if int(count) > 0 and column in REQUIRED_EVENT_COLUMNS + RECOMMENDED_EVENT_COLUMNS
    }

    non_numeric_columns: list[str] = []
    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            converted = pd.to_numeric(df[column], errors="coerce")
            original_non_null = df[column].notna()
            invalid_mask = original_non_null & converted.isna()
            if bool(invalid_mask.any()):
                non_numeric_columns.append(column)

    normalized_coordinate_issues: dict[str, int] = {}
    for column in NORM_COLUMNS:
        if column in df.columns:
            values = pd.to_numeric(df[column], errors="coerce")
            out_of_range = values.notna() & ((values < 0.0) | (values > 1.0))
            if bool(out_of_range.any()):
                normalized_coordinate_issues[column] = int(out_of_range.sum())

    if "dataset_id" in df.columns:
        dataset_ids = sorted(str(value) for value in df["dataset_id"].dropna().unique())
    else:
        dataset_ids = []

    if "label" in df.columns:
        label_counts = {str(key): int(value) for key, value in df["label"].value_counts(dropna=False).sort_index().items()}
    else:
        label_counts = {}

    if "event_type" in df.columns:
        event_type_counts = {
            str(key): int(value) for key, value in df["event_type"].value_counts(dropna=False).sort_index().items()
        }
    else:
        event_type_counts = {}

    structural_errors: list[str] = []
    quality_warnings: list[str] = []
    for column in missing_required:
        structural_errors.append(f"missing_required_column:{column}")
    for column in non_numeric_columns:
        structural_errors.append(f"non_numeric_column:{column}")
    if require_label and "label" in columns_with_nulls:
        structural_errors.append("missing_label_values")
    if strict_normalized_range:
        for column, count in normalized_coordinate_issues.items():
            structural_errors.append(f"normalized_coordinate_out_of_range:{column}:{count}")

    if "label" in columns_with_nulls and not require_label:
        quality_warnings.append(f"missing_label_values:{columns_with_nulls['label']}")
    for column, count in normalized_coordinate_issues.items():
        quality_warnings.append(f"normalized_coordinate_out_of_range:{column}:{count}")
    for column in missing_recommended:
        quality_warnings.append(f"missing_recommended_column:{column}")

    n_subjects = int(df["subject_id"].nunique()) if "subject_id" in df.columns else None
    passed = not structural_errors

    return SchemaValidationReport(
        table_path=str(table_path),
        n_rows=int(len(df)),
        n_columns=int(len(df.columns)),
        n_subjects=n_subjects,
        dataset_ids=dataset_ids,
        label_counts=label_counts,
        missing_required_columns=missing_required,
        missing_recommended_columns=missing_recommended,
        present_optional_columns=present_optional,
        columns_with_nulls=columns_with_nulls,
        structural_errors=structural_errors,
        quality_warnings=quality_warnings,
        non_numeric_columns=non_numeric_columns,
        normalized_coordinate_issues=normalized_coordinate_issues,
        event_type_counts=event_type_counts,
        passed=passed,
    )
