from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScreenGeometry:
    width_px: float
    height_px: float
    diagonal_in: float | None = None
    viewing_distance_cm: float | None = None

    @property
    def width_cm(self) -> float | None:
        if self.diagonal_in is None:
            return None
        diagonal_cm = self.diagonal_in * 2.54
        diagonal_px = math.sqrt(self.width_px**2 + self.height_px**2)
        return diagonal_cm * self.width_px / diagonal_px

    @property
    def height_cm(self) -> float | None:
        if self.diagonal_in is None:
            return None
        diagonal_cm = self.diagonal_in * 2.54
        diagonal_px = math.sqrt(self.width_px**2 + self.height_px**2)
        return diagonal_cm * self.height_px / diagonal_px

    @property
    def can_compute_dva(self) -> bool:
        return self.width_cm is not None and self.height_cm is not None and self.viewing_distance_cm is not None


def normalize_coordinates(df: pd.DataFrame, geometry: ScreenGeometry) -> pd.DataFrame:
    out = df.copy()
    out["x_px"] = pd.to_numeric(out["FIX_X"], errors="coerce")
    out["y_px"] = pd.to_numeric(out["FIX_Y"], errors="coerce")
    out["x_norm"] = out["x_px"] / geometry.width_px
    out["y_norm"] = out["y_px"] / geometry.height_px

    if geometry.can_compute_dva:
        width_cm = float(geometry.width_cm)
        height_cm = float(geometry.height_cm)
        distance_cm = float(geometry.viewing_distance_cm)
        x_cm = (out["x_norm"] - 0.5) * width_cm
        y_cm = (out["y_norm"] - 0.5) * height_cm
        out["x_dva"] = np.degrees(np.arctan2(x_cm, distance_cm))
        out["y_dva"] = np.degrees(np.arctan2(y_cm, distance_cm))
    else:
        out["x_dva"] = np.nan
        out["y_dva"] = np.nan
    return out


def add_transition_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    group_cols = ["subject_id", "trial_id"]
    for col in ["x_norm", "y_norm", "x_dva", "y_dva", "duration_ms"]:
        out[f"next_{col}"] = out.groupby(group_cols)[col].shift(-1)

    out["saccade_dx_norm"] = out["next_x_norm"] - out["x_norm"]
    out["saccade_dy_norm"] = out["next_y_norm"] - out["y_norm"]
    out["saccade_amplitude_norm"] = np.hypot(out["saccade_dx_norm"], out["saccade_dy_norm"])
    out["saccade_angle"] = np.arctan2(out["saccade_dy_norm"], out["saccade_dx_norm"])

    out["saccade_dx_dva"] = out["next_x_dva"] - out["x_dva"]
    out["saccade_dy_dva"] = out["next_y_dva"] - out["y_dva"]
    out["saccade_amplitude_dva"] = np.hypot(out["saccade_dx_dva"], out["saccade_dy_dva"])

    next_duration_s = out["next_duration_ms"] / 1000.0
    out["transition_velocity_dva_s_approx"] = out["saccade_amplitude_dva"] / next_duration_s
    out.loc[next_duration_s <= 0, "transition_velocity_dva_s_approx"] = np.nan

    out = out.drop(
        columns=[
            "next_x_norm",
            "next_y_norm",
            "next_x_dva",
            "next_y_dva",
            "next_duration_ms",
        ]
    )
    return out


def summarize_event_table(events: pd.DataFrame) -> dict:
    train = events[events["split"] == "train_valid"]
    return {
        "n_events": int(len(events)),
        "n_train_valid_events": int(len(train)),
        "n_subjects": int(events["subject_id"].nunique()),
        "n_train_valid_subjects": int(train["subject_id"].nunique()),
        "n_trials": int(events[["subject_id", "trial_id"]].drop_duplicates().shape[0]),
        "n_missing_transitions": int(events["saccade_amplitude_norm"].isna().sum()),
        "duration_ms_mean": float(events["duration_ms"].mean()),
        "duration_ms_std": float(events["duration_ms"].std()),
        "saccade_amplitude_norm_mean": float(events["saccade_amplitude_norm"].mean()),
        "saccade_amplitude_dva_mean": float(events["saccade_amplitude_dva"].mean()),
    }
