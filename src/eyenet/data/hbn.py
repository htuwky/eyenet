from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eyenet.data.fixation_detection import IDTFixationConfig, detect_fixations_idt, prepare_raw_gaze_samples

HBN_FILENAME_RE = re.compile(r"^(?P<subject_id>NDAR[A-Z0-9]+)_(?P<trial_id>.+)\.csv$")


@dataclass(frozen=True)
class HBNAdapterConfig:
    raw_dir: Path
    sampling_rate_hz: float = 120.0
    screen_width_px: float = 800.0
    screen_height_px: float = 600.0
    screen_width_cm: float = 33.8
    screen_height_cm: float = 27.0
    viewing_distance_cm: float = 63.5
    time_unit: str = "sample"
    dispersion_threshold_px: float = 35.0
    min_duration_ms: float = 80.0
    max_gap_ms: float = 75.0
    invalid_zero: bool = True
    max_files: int | None = None


def build_hbn_fixation_events(cfg: HBNAdapterConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    csv_files = sorted(Path(cfg.raw_dir).glob("*.csv"))
    if cfg.max_files is not None:
        csv_files = csv_files[: cfg.max_files]
    if not csv_files:
        raise FileNotFoundError(f"No HBN CSV files found in {cfg.raw_dir}")

    idt_cfg = IDTFixationConfig(
        sampling_rate_hz=cfg.sampling_rate_hz,
        dispersion_threshold_px=cfg.dispersion_threshold_px,
        min_duration_ms=cfg.min_duration_ms,
        max_gap_ms=cfg.max_gap_ms,
    )

    event_tables: list[pd.DataFrame] = []
    file_rows: list[dict[str, Any]] = []
    for path in csv_files:
        subject_id, trial_id = parse_hbn_filename(path.name)
        raw = pd.read_csv(path, low_memory=False)
        samples = prepare_raw_gaze_samples(
            raw,
            x_column="x_pix",
            y_column="y_pix",
            time_column="time",
            screen_width_px=cfg.screen_width_px,
            screen_height_px=cfg.screen_height_px,
            sampling_rate_hz=cfg.sampling_rate_hz,
            time_unit=cfg.time_unit,
            invalid_zero=cfg.invalid_zero,
        )
        fixations = detect_fixations_idt(samples, idt_cfg)
        file_rows.append(
            {
                "file": str(path),
                "subject_id": subject_id,
                "trial_id": trial_id,
                "n_raw_rows": int(len(raw)),
                "n_valid_samples": int(len(samples)),
                "n_fixations": int(len(fixations)),
                "valid_sample_rate": float(len(samples) / len(raw)) if len(raw) else 0.0,
            }
        )
        if fixations.empty:
            continue
        fixations = fixations.assign(
            dataset_id="HBN",
            subject_id=subject_id,
            label=np.nan,
            trial_id=trial_id,
            event_type="fixation",
            split="unassigned",
            fold="unassigned",
            session_id=pd.NA,
            task_id=trial_id,
        )
        event_tables.append(fixations)

    if not event_tables:
        raise ValueError("HBN adapter did not detect any fixation events.")

    events = pd.concat(event_tables, ignore_index=True)
    events = add_hbn_dva(events, cfg)
    events = add_transition_and_encoder_columns(events)
    events = events.sort_values(["subject_id", "trial_id", "event_index"]).reset_index(drop=True)
    events = events[ordered_hbn_event_columns(events)]

    file_report = pd.DataFrame(file_rows)
    summary = {
        "dataset_id": "HBN",
        "n_files": int(len(csv_files)),
        "n_files_with_fixations": int(file_report["n_fixations"].gt(0).sum()),
        "n_subjects": int(events["subject_id"].nunique()),
        "n_trials": int(events[["subject_id", "trial_id"]].drop_duplicates().shape[0]),
        "n_fixation_events": int(len(events)),
        "sampling_rate_hz": float(cfg.sampling_rate_hz),
        "screen_width_px": float(cfg.screen_width_px),
        "screen_height_px": float(cfg.screen_height_px),
        "screen_width_cm": float(cfg.screen_width_cm),
        "screen_height_cm": float(cfg.screen_height_cm),
        "viewing_distance_cm": float(cfg.viewing_distance_cm),
        "idt": {
            "dispersion_threshold_px": float(cfg.dispersion_threshold_px),
            "min_duration_ms": float(cfg.min_duration_ms),
            "max_gap_ms": float(cfg.max_gap_ms),
        },
        "valid_sample_rate_mean": float(file_report["valid_sample_rate"].mean()),
        "fixations_per_file": describe_numeric(file_report["n_fixations"]),
        "duration_ms": describe_numeric(events["duration_ms"]),
    }
    return events, {"summary": summary, "file_report": file_report}


def parse_hbn_filename(filename: str) -> tuple[str, str]:
    match = HBN_FILENAME_RE.match(filename)
    if match is None:
        raise ValueError(f"Unexpected HBN filename format: {filename}")
    return match.group("subject_id"), match.group("trial_id")


def add_hbn_dva(events: pd.DataFrame, cfg: HBNAdapterConfig) -> pd.DataFrame:
    out = events.copy()
    x_cm = (out["x_norm"] - 0.5) * float(cfg.screen_width_cm)
    y_cm = (out["y_norm"] - 0.5) * float(cfg.screen_height_cm)
    out["x_dva"] = np.degrees(np.arctan2(x_cm, float(cfg.viewing_distance_cm)))
    out["y_dva"] = np.degrees(np.arctan2(y_cm, float(cfg.viewing_distance_cm)))
    return out


def add_transition_and_encoder_columns(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    group_cols = ["subject_id", "trial_id"]
    sort_cols = group_cols + ["event_index"]
    out = out.sort_values(sort_cols).reset_index(drop=True)

    for col in ["x_norm", "y_norm", "x_dva", "y_dva", "timestamp_start_ms", "duration_ms"]:
        out[f"next_{col}"] = out.groupby(group_cols)[col].shift(-1)

    out["saccade_dx_norm"] = out["next_x_norm"] - out["x_norm"]
    out["saccade_dy_norm"] = out["next_y_norm"] - out["y_norm"]
    out["saccade_amplitude_norm"] = np.hypot(out["saccade_dx_norm"], out["saccade_dy_norm"])
    out["saccade_angle"] = np.arctan2(out["saccade_dy_norm"], out["saccade_dx_norm"])
    out["saccade_angle_sin"] = np.sin(out["saccade_angle"])
    out["saccade_angle_cos"] = np.cos(out["saccade_angle"])

    out["saccade_dx_dva"] = out["next_x_dva"] - out["x_dva"]
    out["saccade_dy_dva"] = out["next_y_dva"] - out["y_dva"]
    out["saccade_amplitude_dva"] = np.hypot(out["saccade_dx_dva"], out["saccade_dy_dva"])
    transition_time_s = (out["next_timestamp_start_ms"] - out["timestamp_start_ms"]) / 1000.0
    out["transition_velocity_dva_s_approx"] = out["saccade_amplitude_dva"] / transition_time_s
    out.loc[transition_time_s <= 0, "transition_velocity_dva_s_approx"] = np.nan
    out["transition_missing"] = out["saccade_amplitude_norm"].isna().astype(int)

    out["log_duration_ms"] = np.log1p(out["duration_ms"].clip(lower=0))
    out["log_transition_velocity_dva_s"] = np.log1p(out["transition_velocity_dva_s_approx"].clip(lower=0))

    out["segment_index"] = out.groupby("subject_id")["trial_id"].transform(lambda series: pd.factorize(series)[0] + 1)
    out["event_index_in_segment"] = out.groupby(group_cols).cumcount() + 1
    events_per_segment = out.groupby(group_cols)["event_index_in_segment"].transform("max")
    out["event_index_in_segment_norm"] = np.where(
        events_per_segment > 1,
        (out["event_index_in_segment"] - 1) / (events_per_segment - 1),
        0.0,
    )
    out["is_first_event_in_segment"] = (out["event_index_in_segment"] == 1).astype(int)
    out["is_last_event_in_segment"] = (out["event_index_in_segment"] == events_per_segment).astype(int)

    return out.drop(columns=[col for col in out.columns if col.startswith("next_")])


def ordered_hbn_event_columns(events: pd.DataFrame) -> list[str]:
    preferred = [
        "dataset_id",
        "subject_id",
        "split",
        "fold",
        "label",
        "trial_id",
        "segment_index",
        "event_index",
        "event_index_in_segment",
        "event_index_in_segment_norm",
        "event_type",
        "x_px",
        "y_px",
        "x_norm",
        "y_norm",
        "x_dva",
        "y_dva",
        "timestamp_start_ms",
        "timestamp_end_ms",
        "duration_ms",
        "log_duration_ms",
        "saccade_dx_norm",
        "saccade_dy_norm",
        "saccade_amplitude_norm",
        "saccade_angle",
        "saccade_angle_sin",
        "saccade_angle_cos",
        "saccade_dx_dva",
        "saccade_dy_dva",
        "saccade_amplitude_dva",
        "transition_velocity_dva_s_approx",
        "log_transition_velocity_dva_s",
        "transition_missing",
        "is_first_event_in_segment",
        "is_last_event_in_segment",
        "n_raw_samples",
        "session_id",
        "task_id",
    ]
    return [col for col in preferred if col in events.columns] + [col for col in events.columns if col not in preferred]


def describe_numeric(series: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return {"min": np.nan, "median": np.nan, "mean": np.nan, "max": np.nan}
    return {
        "min": float(np.min(values)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p75": float(np.quantile(values, 0.75)),
        "max": float(np.max(values)),
    }
