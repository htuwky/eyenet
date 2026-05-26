from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eyenet.data.hbn import add_transition_and_encoder_columns, describe_numeric


@dataclass(frozen=True)
class OneStopAdapterConfig:
    events_path: Path
    screen_width_px: float = 2560.0
    screen_height_px: float = 1440.0
    exclude_practice_trials: bool = True
    exclude_repeated_reading_trials: bool = True
    regime: str = "all"
    chunksize: int = 250_000
    max_rows: int | None = None


USE_COLUMNS = [
    "participant_id",
    "TRIAL_INDEX",
    "CURRENT_FIX_INDEX",
    "CURRENT_FIX_DURATION",
    "CURRENT_FIX_START",
    "CURRENT_FIX_END",
    "CURRENT_FIX_X",
    "CURRENT_FIX_Y",
    "question_preview",
    "repeated_reading_trial",
    "practice_trial",
]


def build_onestop_fixation_events(cfg: OneStopAdapterConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    if cfg.regime not in {"all", "ordinary", "information_seeking"}:
        raise ValueError("--regime must be one of: all, ordinary, information_seeking")
    if not cfg.events_path.exists():
        raise FileNotFoundError(f"OneStop fixation table does not exist: {cfg.events_path}")

    tables: list[pd.DataFrame] = []
    n_source_rows = 0
    n_rows_after_filters = 0
    reader = pd.read_csv(
        cfg.events_path,
        usecols=USE_COLUMNS,
        dtype={"participant_id": str},
        chunksize=cfg.chunksize,
        low_memory=False,
        na_values=["."],
    )

    for chunk in reader:
        if cfg.max_rows is not None:
            remaining = cfg.max_rows - n_source_rows
            if remaining <= 0:
                break
            chunk = chunk.head(remaining)
        if chunk.empty:
            continue

        n_source_rows += int(len(chunk))
        filtered = filter_onestop_chunk(chunk, cfg)
        n_rows_after_filters += int(len(filtered))
        if filtered.empty:
            continue
        tables.append(convert_onestop_chunk(filtered, cfg))

    if not tables:
        raise ValueError("OneStop adapter produced no fixation events after filtering.")

    events = pd.concat(tables, ignore_index=True)
    events = add_transition_and_encoder_columns(events)
    events = events.sort_values(["subject_id", "trial_id", "event_index"]).reset_index(drop=True)
    events = events[ordered_onestop_event_columns(events)]

    file_report = build_onestop_trial_report(events)
    summary = {
        "dataset_id": "OneStop",
        "source_events": str(cfg.events_path),
        "n_source_rows_read": int(n_source_rows),
        "n_rows_after_filters": int(n_rows_after_filters),
        "n_subjects": int(events["subject_id"].nunique()),
        "n_trials": int(events[["subject_id", "trial_id"]].drop_duplicates().shape[0]),
        "n_fixation_events": int(len(events)),
        "screen_width_px": float(cfg.screen_width_px),
        "screen_height_px": float(cfg.screen_height_px),
        "coordinate_assumption": (
            "OneStop fixation and interest-area coordinates are treated as screen-space pixels. "
            "The default 2560x1440 bounds are inferred from TOP_LEFT=(368,186), "
            "interest-area extents, and low observed out-of-range rate."
        ),
        "out_of_range_coordinate_rate": compute_out_of_range_rate(events),
        "filters": {
            "exclude_practice_trials": bool(cfg.exclude_practice_trials),
            "exclude_repeated_reading_trials": bool(cfg.exclude_repeated_reading_trials),
            "regime": cfg.regime,
        },
        "task_counts_events": {
            str(key): int(value) for key, value in events["task_id"].value_counts(dropna=False).items()
        },
        "fixations_per_trial": describe_numeric(file_report["n_fixations"]),
        "events_per_subject": describe_numeric(events.groupby("subject_id").size()),
        "duration_ms": describe_numeric(events["duration_ms"]),
    }
    return events, {"summary": summary, "file_report": file_report}


def filter_onestop_chunk(chunk: pd.DataFrame, cfg: OneStopAdapterConfig) -> pd.DataFrame:
    out = chunk.copy()
    practice = parse_bool_series(out["practice_trial"])
    repeated = parse_bool_series(out["repeated_reading_trial"])
    question_preview = parse_bool_series(out["question_preview"])

    keep = pd.Series(True, index=out.index)
    if cfg.exclude_practice_trials:
        keep &= ~practice
    if cfg.exclude_repeated_reading_trials:
        keep &= ~repeated
    if cfg.regime == "ordinary":
        keep &= ~question_preview
    elif cfg.regime == "information_seeking":
        keep &= question_preview
    return out.loc[keep].reset_index(drop=True)


def convert_onestop_chunk(chunk: pd.DataFrame, cfg: OneStopAdapterConfig) -> pd.DataFrame:
    out = pd.DataFrame(index=chunk.index)
    out["dataset_id"] = "OneStop"
    out["subject_id"] = chunk["participant_id"].astype(str)
    out["split"] = "unassigned"
    out["fold"] = "unassigned"
    out["label"] = np.nan
    out["trial_id"] = pd.to_numeric(chunk["TRIAL_INDEX"], errors="coerce").astype("Int64").astype(str)
    out["event_index"] = pd.to_numeric(chunk["CURRENT_FIX_INDEX"], errors="coerce").astype("Int64")
    out["event_type"] = "fixation"
    out["x_px"] = pd.to_numeric(chunk["CURRENT_FIX_X"], errors="coerce")
    out["y_px"] = pd.to_numeric(chunk["CURRENT_FIX_Y"], errors="coerce")
    out["x_norm"] = out["x_px"] / float(cfg.screen_width_px)
    out["y_norm"] = out["y_px"] / float(cfg.screen_height_px)
    out["x_dva"] = np.nan
    out["y_dva"] = np.nan
    out["timestamp_start_ms"] = pd.to_numeric(chunk["CURRENT_FIX_START"], errors="coerce")
    out["timestamp_end_ms"] = pd.to_numeric(chunk["CURRENT_FIX_END"], errors="coerce")
    out["duration_ms"] = pd.to_numeric(chunk["CURRENT_FIX_DURATION"], errors="coerce")
    out["n_raw_samples"] = pd.NA
    out["session_id"] = pd.NA

    question_preview = parse_bool_series(chunk["question_preview"])
    out["task_id"] = np.where(question_preview, "information_seeking", "ordinary_reading")
    out["question_preview"] = question_preview.astype(int)
    out["repeated_reading_trial"] = parse_bool_series(chunk["repeated_reading_trial"]).astype(int)
    out["practice_trial"] = parse_bool_series(chunk["practice_trial"]).astype(int)

    return out


def parse_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = series.fillna(False).astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes"})


def compute_out_of_range_rate(events: pd.DataFrame) -> float:
    valid = events["x_norm"].notna() & events["y_norm"].notna()
    if not bool(valid.any()):
        return float("nan")
    out_of_range = valid & (
        ~events["x_norm"].between(0.0, 1.0, inclusive="both")
        | ~events["y_norm"].between(0.0, 1.0, inclusive="both")
    )
    return float(out_of_range.sum() / valid.sum())


def build_onestop_trial_report(events: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["subject_id", "trial_id", "task_id"]
    out = (
        events.groupby(group_cols, dropna=False)
        .agg(n_fixations=("event_index", "size"), duration_ms_median=("duration_ms", "median"))
        .reset_index()
    )
    coordinate_out_of_range = (
        events.assign(
            _coordinate_out_of_range=(
                ~events["x_norm"].between(0.0, 1.0, inclusive="both")
                | ~events["y_norm"].between(0.0, 1.0, inclusive="both")
            )
        )
        .groupby(group_cols, dropna=False)["_coordinate_out_of_range"]
        .mean()
        .reset_index(name="out_of_range_coordinate_rate")
    )
    return out.merge(coordinate_out_of_range, on=group_cols, how="left")


def ordered_onestop_event_columns(events: pd.DataFrame) -> list[str]:
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
        "question_preview",
        "repeated_reading_trial",
        "practice_trial",
    ]
    return [col for col in preferred if col in events.columns] + [col for col in events.columns if col not in preferred]
