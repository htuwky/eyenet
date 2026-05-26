from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from eyenet.data.features import add_descriptive, angle_entropy, bcea, grid_coverage


SEGMENT_BASE_FEATURES = [
    "n_fixations",
    "fix_duration_ms_mean",
    "fix_duration_ms_std",
    "fix_duration_ms_median",
    "fix_duration_ms_iqr",
    "fix_duration_ms_max",
    "short_fixation_ratio_lt150ms",
    "long_fixation_ratio_gt500ms",
    "saccade_amp_norm_mean",
    "saccade_amp_norm_std",
    "saccade_amp_norm_median",
    "saccade_amp_norm_iqr",
    "saccade_amp_dva_mean",
    "saccade_amp_dva_std",
    "saccade_amp_dva_median",
    "saccade_amp_dva_iqr",
    "transition_velocity_dva_s_mean",
    "transition_velocity_dva_s_std",
    "transition_velocity_dva_s_median",
    "transition_velocity_dva_s_iqr",
    "scanpath_length_norm",
    "scanpath_length_dva",
    "spatial_x_range",
    "spatial_y_range",
    "spatial_bbox_area",
    "center_bias_mean",
    "bcea_norm",
    "spatial_coverage_8x8",
    "transition_angle_entropy_8bin",
]


def extract_segment_features(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    group_cols = ["subject_id", "split", "fold", "label", "trial_id"]
    for group_values, segment_events in events.groupby(group_cols, sort=False, dropna=False):
        subject_id, split, fold, label, trial_id = group_values
        row = {
            "subject_id": subject_id,
            "split": split,
            "fold": fold,
            "label": label,
            "segment_id": trial_id,
        }
        row.update(extract_one_segment(segment_events))
        rows.append(row)

    segments = pd.DataFrame(rows)
    segments["segment_index"] = segments.groupby("subject_id").cumcount() + 1
    leading_cols = ["subject_id", "split", "fold", "label", "segment_id", "segment_index"]
    return segments[leading_cols + [col for col in segments.columns if col not in leading_cols]]


def extract_one_segment(df: pd.DataFrame) -> dict:
    transition_df = df[df["saccade_amplitude_norm"].notna()].copy()
    features: dict[str, float] = {
        "n_fixations": float(len(df)),
        "short_fixation_ratio_lt150ms": float((df["duration_ms"] < 150).mean()),
        "long_fixation_ratio_gt500ms": float((df["duration_ms"] > 500).mean()),
    }
    add_descriptive(features, "fix_duration_ms", df["duration_ms"])
    add_descriptive(features, "saccade_amp_norm", transition_df["saccade_amplitude_norm"])
    add_descriptive(features, "saccade_amp_dva", transition_df["saccade_amplitude_dva"])
    add_descriptive(features, "transition_velocity_dva_s", transition_df["transition_velocity_dva_s_approx"])

    features["scanpath_length_norm"] = float(transition_df["saccade_amplitude_norm"].sum())
    features["scanpath_length_dva"] = float(transition_df["saccade_amplitude_dva"].sum())
    features["spatial_x_range"] = float(df["x_norm"].max() - df["x_norm"].min())
    features["spatial_y_range"] = float(df["y_norm"].max() - df["y_norm"].min())
    features["spatial_bbox_area"] = features["spatial_x_range"] * features["spatial_y_range"]
    features["center_bias_mean"] = float(np.hypot(df["x_norm"] - 0.5, df["y_norm"] - 0.5).mean())
    features["bcea_norm"] = bcea(df["x_norm"], df["y_norm"])
    features["spatial_coverage_8x8"] = grid_coverage(df["x_norm"], df["y_norm"], bins=8)
    features["transition_angle_entropy_8bin"] = angle_entropy(transition_df["saccade_angle"], bins=8)
    return features


def aggregate_segment_features(segments: pd.DataFrame, feature_cols: list[str] | None = None) -> pd.DataFrame:
    feature_cols = feature_cols or [col for col in SEGMENT_BASE_FEATURES if col in segments.columns]
    rows: list[dict] = []
    train_valid_segments = segments[segments["split"] == "train_valid"].copy()

    for subject_id, subject_segments in train_valid_segments.groupby("subject_id", sort=True):
        labels = subject_segments["label"].dropna()
        if labels.empty:
            raise ValueError(f"Subject {subject_id} has no non-missing label for segment aggregation.")
        label = labels.iloc[0]
        fold = subject_segments["fold"].iloc[0]
        row = {
            "subject_id": subject_id,
            "fold": fold,
            "label": int(label),
            "n_segments": float(len(subject_segments)),
        }
        for feature in feature_cols:
            values = pd.to_numeric(subject_segments[feature], errors="coerce").dropna()
            add_segment_aggregates(row, feature, values)
        rows.append(row)

    return pd.DataFrame(rows).sort_values("subject_id").reset_index(drop=True)


def add_segment_aggregates(row: dict[str, float], feature: str, values: pd.Series) -> None:
    if values.empty:
        for suffix in ["mean", "std", "median", "min", "max", "iqr", "top10_mean", "bottom10_mean", "trend_slope"]:
            row[f"seg_{feature}_{suffix}"] = np.nan
        return

    row[f"seg_{feature}_mean"] = float(values.mean())
    row[f"seg_{feature}_std"] = float(values.std())
    row[f"seg_{feature}_median"] = float(values.median())
    row[f"seg_{feature}_min"] = float(values.min())
    row[f"seg_{feature}_max"] = float(values.max())
    row[f"seg_{feature}_iqr"] = float(values.quantile(0.75) - values.quantile(0.25))
    row[f"seg_{feature}_top10_mean"] = quantile_tail_mean(values, upper=True)
    row[f"seg_{feature}_bottom10_mean"] = quantile_tail_mean(values, upper=False)
    row[f"seg_{feature}_trend_slope"] = trend_slope(values)


def quantile_tail_mean(values: pd.Series, upper: bool) -> float:
    if values.empty:
        return np.nan
    threshold = values.quantile(0.9 if upper else 0.1)
    tail = values[values >= threshold] if upper else values[values <= threshold]
    return float(tail.mean())


def trend_slope(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(clean) < 2:
        return np.nan
    x = np.arange(len(clean), dtype=float)
    slope, _ = np.polyfit(x, clean, deg=1)
    return float(slope)


def save_segment_feature_outputs(
    output_dir: str | Path,
    segments: pd.DataFrame,
    subject_features: pd.DataFrame,
    suffix: str = "no_pupil",
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    segments.to_csv(output_path / f"ems_segment_features_{suffix}.csv", index=False, encoding="utf-8-sig")
    subject_features.to_csv(
        output_path / f"ems_subject_features_segment_agg_{suffix}.csv",
        index=False,
        encoding="utf-8-sig",
    )
