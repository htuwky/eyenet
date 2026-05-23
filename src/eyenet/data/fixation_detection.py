from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IDTFixationConfig:
    sampling_rate_hz: float
    dispersion_threshold_px: float = 35.0
    min_duration_ms: float = 80.0
    max_gap_ms: float = 75.0


def prepare_raw_gaze_samples(
    frame: pd.DataFrame,
    *,
    x_column: str,
    y_column: str,
    time_column: str,
    screen_width_px: float,
    screen_height_px: float,
    sampling_rate_hz: float,
    time_unit: str = "sample",
    invalid_zero: bool = True,
) -> pd.DataFrame:
    """Return valid raw gaze samples with normalized coordinates and millisecond timestamps."""
    data = frame[[x_column, y_column, time_column]].copy()
    data = data.rename(columns={x_column: "x_px", y_column: "y_px", time_column: "time_raw"})
    data["x_px"] = pd.to_numeric(data["x_px"], errors="coerce")
    data["y_px"] = pd.to_numeric(data["y_px"], errors="coerce")
    data["time_raw"] = pd.to_numeric(data["time_raw"], errors="coerce")

    if time_unit == "sample":
        data["timestamp_ms"] = data["time_raw"] * (1000.0 / float(sampling_rate_hz))
    elif time_unit == "ms":
        data["timestamp_ms"] = data["time_raw"]
    else:
        raise ValueError(f"Unsupported time_unit: {time_unit}. Use 'sample' or 'ms'.")

    valid = data["x_px"].notna() & data["y_px"].notna() & data["timestamp_ms"].notna()
    valid &= data["x_px"].between(0.0, float(screen_width_px), inclusive="both")
    valid &= data["y_px"].between(0.0, float(screen_height_px), inclusive="both")
    if invalid_zero:
        valid &= ~((data["x_px"] == 0.0) & (data["y_px"] == 0.0))

    data = data.loc[valid].sort_values("timestamp_ms").reset_index(drop=True)
    data["x_norm"] = data["x_px"] / float(screen_width_px)
    data["y_norm"] = data["y_px"] / float(screen_height_px)
    return data


def detect_fixations_idt(samples: pd.DataFrame, cfg: IDTFixationConfig) -> pd.DataFrame:
    """Detect fixation events using a conservative I-DT algorithm on valid gaze samples."""
    if samples.empty:
        return empty_fixation_frame()

    required = {"x_px", "y_px", "x_norm", "y_norm", "timestamp_ms"}
    missing = required.difference(samples.columns)
    if missing:
        raise ValueError(f"Raw gaze samples are missing required columns: {sorted(missing)}")

    blocks = split_contiguous_blocks(samples, max_gap_ms=cfg.max_gap_ms)
    rows: list[dict[str, float | int]] = []
    for block in blocks:
        rows.extend(detect_fixations_in_block(block, cfg))

    if not rows:
        return empty_fixation_frame()

    fixations = pd.DataFrame(rows)
    fixations = fixations.sort_values("timestamp_start_ms").reset_index(drop=True)
    fixations["event_index"] = np.arange(1, len(fixations) + 1, dtype=int)
    return fixations


def split_contiguous_blocks(samples: pd.DataFrame, max_gap_ms: float) -> list[pd.DataFrame]:
    data = samples.sort_values("timestamp_ms").reset_index(drop=True)
    if data.empty:
        return []
    gap = data["timestamp_ms"].diff().fillna(0.0)
    block_id = (gap > max_gap_ms).cumsum()
    return [block.reset_index(drop=True) for _, block in data.groupby(block_id, sort=True) if len(block) > 0]


def detect_fixations_in_block(block: pd.DataFrame, cfg: IDTFixationConfig) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    data = block.reset_index(drop=True)
    sample_period_ms = 1000.0 / cfg.sampling_rate_hz
    min_samples = max(2, int(np.ceil(cfg.min_duration_ms / sample_period_ms)) + 1)
    start = 0
    n_rows = len(data)

    while start + min_samples <= n_rows:
        end = start + min_samples
        window = data.iloc[start:end]
        if duration_ms(window) < cfg.min_duration_ms or dispersion_px(window) > cfg.dispersion_threshold_px:
            start += 1
            continue

        last_valid_end = end
        while end < n_rows:
            candidate = data.iloc[start : end + 1]
            if dispersion_px(candidate) > cfg.dispersion_threshold_px:
                break
            last_valid_end = end + 1
            end += 1

        fixation = data.iloc[start:last_valid_end]
        rows.append(make_fixation_row(fixation))
        start = last_valid_end

    return rows


def dispersion_px(window: pd.DataFrame) -> float:
    return float((window["x_px"].max() - window["x_px"].min()) + (window["y_px"].max() - window["y_px"].min()))


def duration_ms(window: pd.DataFrame) -> float:
    return float(window["timestamp_ms"].iloc[-1] - window["timestamp_ms"].iloc[0])


def make_fixation_row(fixation: pd.DataFrame) -> dict[str, float | int]:
    start_ms = float(fixation["timestamp_ms"].iloc[0])
    end_ms = float(fixation["timestamp_ms"].iloc[-1])
    return {
        "timestamp_start_ms": start_ms,
        "timestamp_end_ms": end_ms,
        "duration_ms": max(0.0, end_ms - start_ms),
        "x_px": float(fixation["x_px"].mean()),
        "y_px": float(fixation["y_px"].mean()),
        "x_norm": float(fixation["x_norm"].mean()),
        "y_norm": float(fixation["y_norm"].mean()),
        "n_raw_samples": int(len(fixation)),
    }


def empty_fixation_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "event_index",
            "timestamp_start_ms",
            "timestamp_end_ms",
            "duration_ms",
            "x_px",
            "y_px",
            "x_norm",
            "y_norm",
            "n_raw_samples",
        ]
    )
