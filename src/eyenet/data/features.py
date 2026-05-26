from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


def load_events(path: str | Path, train_valid_only: bool = True) -> pd.DataFrame:
    events = pd.read_csv(path, dtype={"subject_id": str}, low_memory=False)
    if train_valid_only:
        events = events[events["split"] == "train_valid"].copy()
    return events


def extract_subject_features(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for subject_id, subject_events in events.groupby("subject_id", sort=True):
        labels = subject_events["label"].dropna()
        if labels.empty:
            raise ValueError(f"Subject {subject_id} has no non-missing label for subject feature extraction.")
        label = labels.iloc[0]
        fold = subject_events["fold"].iloc[0]
        row = {
            "subject_id": subject_id,
            "fold": fold,
            "label": int(label),
        }
        row.update(extract_one_subject(subject_events))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("subject_id").reset_index(drop=True)


def extract_one_subject(df: pd.DataFrame) -> dict:
    transition_df = df[df["saccade_amplitude_norm"].notna()].copy()
    features: dict[str, float] = {}

    features["n_trials"] = float(df["trial_id"].nunique())
    features["n_fixations"] = float(len(df))
    features["fixations_per_trial_mean"] = float(df.groupby("trial_id").size().mean())
    features["fixations_per_trial_std"] = float(df.groupby("trial_id").size().std())

    add_descriptive(features, "fix_duration_ms", df["duration_ms"])
    add_descriptive(features, "saccade_amp_norm", transition_df["saccade_amplitude_norm"])
    add_descriptive(features, "saccade_amp_dva", transition_df["saccade_amplitude_dva"])
    add_descriptive(features, "transition_velocity_dva_s", transition_df["transition_velocity_dva_s_approx"])

    features["scanpath_length_norm_total"] = float(transition_df["saccade_amplitude_norm"].sum())
    features["scanpath_length_norm_per_trial_mean"] = float(
        transition_df.groupby("trial_id")["saccade_amplitude_norm"].sum().mean()
    )
    features["scanpath_length_dva_total"] = float(transition_df["saccade_amplitude_dva"].sum())
    features["scanpath_length_dva_per_trial_mean"] = float(
        transition_df.groupby("trial_id")["saccade_amplitude_dva"].sum().mean()
    )

    features["spatial_x_range"] = float(df["x_norm"].max() - df["x_norm"].min())
    features["spatial_y_range"] = float(df["y_norm"].max() - df["y_norm"].min())
    features["spatial_bbox_area"] = features["spatial_x_range"] * features["spatial_y_range"]
    features["center_bias_mean"] = float(np.hypot(df["x_norm"] - 0.5, df["y_norm"] - 0.5).mean())
    features["center_bias_std"] = float(np.hypot(df["x_norm"] - 0.5, df["y_norm"] - 0.5).std())
    features["bcea_norm"] = bcea(df["x_norm"], df["y_norm"])
    features["spatial_coverage_8x8"] = grid_coverage(df["x_norm"], df["y_norm"], bins=8)
    features["spatial_coverage_16x16"] = grid_coverage(df["x_norm"], df["y_norm"], bins=16)
    features["transition_angle_entropy_8bin"] = angle_entropy(transition_df["saccade_angle"], bins=8)

    features["short_fixation_ratio_lt150ms"] = float((df["duration_ms"] < 150).mean())
    features["long_fixation_ratio_gt500ms"] = float((df["duration_ms"] > 500).mean())
    features["large_saccade_ratio_norm_gt025"] = float((transition_df["saccade_amplitude_norm"] > 0.25).mean())

    return features


def add_descriptive(features: dict[str, float], prefix: str, series: pd.Series) -> None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        for suffix in ["mean", "std", "median", "min", "max", "iqr"]:
            features[f"{prefix}_{suffix}"] = np.nan
        return
    features[f"{prefix}_mean"] = float(clean.mean())
    features[f"{prefix}_std"] = float(clean.std())
    features[f"{prefix}_median"] = float(clean.median())
    features[f"{prefix}_min"] = float(clean.min())
    features[f"{prefix}_max"] = float(clean.max())
    features[f"{prefix}_iqr"] = float(clean.quantile(0.75) - clean.quantile(0.25))


def bcea(x: pd.Series, y: pd.Series, probability: float = 0.6827) -> float:
    if not 0.0 < probability < 1.0:
        raise ValueError(f"BCEA probability must be in (0, 1), got {probability}.")
    x_clean = pd.to_numeric(x, errors="coerce")
    y_clean = pd.to_numeric(y, errors="coerce")
    valid = x_clean.notna() & y_clean.notna()
    x_arr = x_clean[valid].to_numpy()
    y_arr = y_clean[valid].to_numpy()
    if len(x_arr) < 3:
        return np.nan
    sx = np.std(x_arr, ddof=1)
    sy = np.std(y_arr, ddof=1)
    if sx == 0.0 or sy == 0.0:
        return 0.0
    rho = np.corrcoef(x_arr, y_arr)[0, 1]
    rho = 0.0 if not np.isfinite(rho) else rho
    k = -math.log(1.0 - probability)
    return float(2.0 * math.pi * k * sx * sy * math.sqrt(max(0.0, 1.0 - rho**2)))


def grid_coverage(x: pd.Series, y: pd.Series, bins: int) -> float:
    x_clean = pd.to_numeric(x, errors="coerce")
    y_clean = pd.to_numeric(y, errors="coerce")
    valid = x_clean.notna() & y_clean.notna()
    if valid.sum() == 0:
        return np.nan
    x_bin = np.clip((x_clean[valid].to_numpy() * bins).astype(int), 0, bins - 1)
    y_bin = np.clip((y_clean[valid].to_numpy() * bins).astype(int), 0, bins - 1)
    occupied = set(zip(x_bin, y_bin))
    return float(len(occupied) / (bins * bins))


def angle_entropy(angles: pd.Series, bins: int = 8) -> float:
    clean = pd.to_numeric(angles, errors="coerce").dropna()
    if clean.empty:
        return np.nan
    counts, _ = np.histogram(clean.to_numpy(), bins=bins, range=(-math.pi, math.pi))
    probs = counts[counts > 0] / counts.sum()
    return float(-(probs * np.log2(probs)).sum())
