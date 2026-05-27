from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


EXCLUDE_COLUMNS = {"subject_id", "fold", "official_fold", "split", "label"}
STRICT_ALLOWED_TOKENS = (
    "short_fixation_ratio_lt150ms",
    "long_fixation_ratio_gt500ms",
    "fix_duration_ms",
    "duration_ms",
    "log_duration_ms",
    "saccade_amp_norm",
    "saccade_amplitude_norm",
    "transition_angle_entropy_8bin",
    "saccade_angle_entropy_8bin",
)
STRICT_BLOCKED_TOKENS = (
    "summary_n_",
    "n_fixations",
    "events_per_segment",
    "x_norm",
    "y_norm",
    "center_distance",
    "center_bias",
    "spatial_",
    "bbox",
    "coverage",
    "scanpath_length",
    "transition_missing",
)


@dataclass(frozen=True)
class SubjectSummaryPreprocessor:
    feature_columns: list[str]
    imputer: SimpleImputer
    scaler: StandardScaler

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame[self.feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        values = self.imputer.transform(values)
        values = self.scaler.transform(values)
        return values.astype(np.float32)


def normalize_subject_id(value: object) -> str:
    text = str(value)
    return text.zfill(3) if text.isdigit() else text


def apply_summary_feature_set(candidate_cols: list[str], feature_set: str) -> tuple[list[str], pd.DataFrame]:
    if feature_set not in {"full", "strict"}:
        raise ValueError("feature_set must be one of: full, strict")
    rows = []
    selected = []
    for col in candidate_cols:
        if feature_set == "full":
            keep = True
            reason = ""
        else:
            blocked = any(token in col for token in STRICT_BLOCKED_TOKENS)
            allowed = any(token in col for token in STRICT_ALLOWED_TOKENS)
            keep = allowed and not blocked
            reason = "" if keep else "outside_strict_content_agnostic_set"
        rows.append(
            {
                "feature": col,
                "feature_set": feature_set,
                "selected_by_feature_set": keep,
                "feature_set_drop_reason": reason,
            }
        )
        if keep:
            selected.append(col)
    if not selected:
        raise ValueError(f"No candidate columns selected by feature_set={feature_set}.")
    return selected, pd.DataFrame(rows)


def select_summary_feature_columns(
    train_df: pd.DataFrame,
    candidate_cols: list[str],
    max_train_missing_rate: float,
) -> tuple[list[str], pd.DataFrame]:
    train_numeric = train_df[candidate_cols].apply(pd.to_numeric, errors="coerce")
    rows = []
    keep_cols = []
    for col in candidate_cols:
        series = train_numeric[col]
        missing_rate = float(series.isna().mean())
        n_unique = int(series.nunique(dropna=True))
        drop_reason = ""
        if missing_rate > max_train_missing_rate:
            drop_reason = "high_train_missing"
        elif n_unique <= 1:
            drop_reason = "constant_or_all_missing_train"
        else:
            keep_cols.append(col)
        rows.append(
            {
                "feature": col,
                "train_missing_rate": missing_rate,
                "train_n_unique_non_missing": n_unique,
                "selected": col in keep_cols,
                "drop_reason": drop_reason,
            }
        )
    if not keep_cols:
        raise ValueError("No usable subject-summary feature columns after train-only filtering.")
    return keep_cols, pd.DataFrame(rows)


def fit_subject_summary_preprocessor(summary: pd.DataFrame, feature_columns: list[str]) -> SubjectSummaryPreprocessor:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    values = summary[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    imputed = imputer.fit_transform(values)
    scaler.fit(imputed)
    return SubjectSummaryPreprocessor(feature_columns=feature_columns, imputer=imputer, scaler=scaler)


def prepare_subject_summary_table(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out["subject_id"] = out["subject_id"].map(normalize_subject_id)
    if "label" in out.columns:
        out["label"] = pd.to_numeric(out["label"], errors="coerce")
    return out
