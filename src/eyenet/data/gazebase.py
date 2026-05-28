from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eyenet.data.fixation_detection import IDTFixationConfig, detect_fixations_idt
from eyenet.data.hbn import add_transition_and_encoder_columns, describe_numeric

SUBJECT_ZIP_RE = re.compile(r"Round_(?P<round_id>\d+)/Subject_(?P<round_subject_id>\d+)\.zip$")
CSV_RE = re.compile(r"S(?P<session_id>\d+)/.+/S_\d+_S\d+_(?P<task_name>[A-Z0-9]+)\.csv$")


@dataclass(frozen=True)
class GazeBaseAdapterConfig:
    raw_root: Path
    sampling_rate_hz: float = 1000.0
    screen_width_px: float = 1680.0
    screen_height_px: float = 1050.0
    screen_width_cm: float = 47.4
    screen_height_cm: float = 29.7
    viewing_distance_cm: float = 55.0
    dispersion_threshold_dva: float = 1.0
    min_duration_ms: float = 80.0
    max_gap_ms: float = 75.0
    tasks: tuple[str, ...] | None = ("VD1", "VD2")
    max_subject_zips: int | None = None


def build_gazebase_fixation_events(cfg: GazeBaseAdapterConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    subject_zips = find_subject_zips(cfg.raw_root)
    if cfg.max_subject_zips is not None:
        subject_zips = subject_zips[: cfg.max_subject_zips]
    if not subject_zips:
        raise FileNotFoundError(f"No GazeBase subject ZIP files found under {cfg.raw_root}")

    idt_cfg = IDTFixationConfig(
        sampling_rate_hz=cfg.sampling_rate_hz,
        dispersion_threshold_px=cfg.dispersion_threshold_dva,
        min_duration_ms=cfg.min_duration_ms,
        max_gap_ms=cfg.max_gap_ms,
    )

    event_tables: list[pd.DataFrame] = []
    file_rows: list[dict[str, Any]] = []
    task_filter = {task.upper() for task in cfg.tasks} if cfg.tasks else None
    for zip_path in subject_zips:
        round_id, round_subject_id, participant_id = parse_subject_zip(zip_path)
        with zipfile.ZipFile(zip_path) as archive:
            for member in sorted(archive.namelist()):
                parsed = parse_member_name(member)
                if parsed is None:
                    continue
                session_id, task_name = parsed
                if task_filter is not None and task_name not in task_filter:
                    continue

                raw = pd.read_csv(archive.open(member), low_memory=False)
                samples = prepare_gazebase_samples(raw, cfg)
                fixations = detect_fixations_idt(samples, idt_cfg)
                trial_id = f"R{round_id}_S{session_id}_{task_name}"
                file_rows.append(
                    {
                        "zip_file": str(zip_path),
                        "member": member,
                        "subject_id": participant_id,
                        "round_id": round_id,
                        "session_id": session_id,
                        "task_id": task_name,
                        "trial_id": trial_id,
                        "n_raw_rows": int(len(raw)),
                        "n_valid_samples": int(len(samples)),
                        "n_fixations": int(len(fixations)),
                        "valid_sample_rate": float(len(samples) / len(raw)) if len(raw) else 0.0,
                    }
                )
                if fixations.empty:
                    continue

                fixations = finalize_fixation_coordinates(fixations, cfg)
                fixations = fixations.assign(
                    dataset_id="GazeBase",
                    subject_id=participant_id,
                    label=np.nan,
                    trial_id=trial_id,
                    round_id=round_id,
                    event_type="fixation",
                    split="unassigned",
                    fold="unassigned",
                    session_id=f"R{round_id}_S{session_id}",
                    task_id=task_name,
                )
                event_tables.append(fixations)

    if not event_tables:
        raise ValueError("GazeBase adapter did not detect any fixation events.")

    events = pd.concat(event_tables, ignore_index=True)
    events = add_transition_and_encoder_columns(events)
    events = events.sort_values(["subject_id", "trial_id", "event_index"]).reset_index(drop=True)
    events = events[ordered_gazebase_event_columns(events)]

    file_report = pd.DataFrame(file_rows)
    summary = {
        "dataset_id": "GazeBase",
        "n_subject_zips": int(len(subject_zips)),
        "n_files_processed": int(len(file_report)),
        "n_files_with_fixations": int(file_report["n_fixations"].gt(0).sum()),
        "n_subjects": int(events["subject_id"].nunique()),
        "n_trials": int(events[["subject_id", "trial_id"]].drop_duplicates().shape[0]),
        "n_fixation_events": int(len(events)),
        "tasks": sorted(events["task_id"].dropna().astype(str).unique().tolist()),
        "sampling_rate_hz": float(cfg.sampling_rate_hz),
        "screen_width_px": float(cfg.screen_width_px),
        "screen_height_px": float(cfg.screen_height_px),
        "screen_width_cm": float(cfg.screen_width_cm),
        "screen_height_cm": float(cfg.screen_height_cm),
        "viewing_distance_cm": float(cfg.viewing_distance_cm),
        "idt": {
            "dispersion_threshold_dva": float(cfg.dispersion_threshold_dva),
            "min_duration_ms": float(cfg.min_duration_ms),
            "max_gap_ms": float(cfg.max_gap_ms),
        },
        "valid_sample_rate_mean": float(file_report["valid_sample_rate"].mean()),
        "fixations_per_file": describe_numeric(file_report["n_fixations"]),
        "duration_ms": describe_numeric(events["duration_ms"]),
    }
    return events, {"summary": summary, "file_report": file_report}


def find_subject_zips(root: Path) -> list[Path]:
    base = Path(root)
    candidates = [
        base / "downloads" / "GazeBase_v2_0",
        base / "GazeBase_v2_0",
        base,
    ]
    for candidate in candidates:
        if candidate.exists():
            files = sorted(candidate.glob("Round_*/Subject_*.zip"))
            if files:
                return files
    return []


def parse_subject_zip(path: Path) -> tuple[str, str, str]:
    normalized = path.as_posix()
    match = SUBJECT_ZIP_RE.search(normalized)
    if match is None:
        raise ValueError(f"Unexpected GazeBase subject ZIP path: {path}")
    round_id = match.group("round_id")
    round_subject_id = match.group("round_subject_id")
    participant_id = round_subject_id[1:] if round_subject_id.startswith(round_id) else round_subject_id
    return round_id, round_subject_id, participant_id.zfill(3)


def parse_member_name(member: str) -> tuple[str, str] | None:
    match = CSV_RE.match(member)
    if match is None:
        return None
    return match.group("session_id"), match.group("task_name")


def prepare_gazebase_samples(frame: pd.DataFrame, cfg: GazeBaseAdapterConfig) -> pd.DataFrame:
    data = frame[["n", "x", "y", "val"]].copy()
    data = data.rename(columns={"n": "timestamp_ms", "x": "x_dva", "y": "y_dva", "val": "validity"})
    for column in ["timestamp_ms", "x_dva", "y_dva", "validity"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    x_half_dva = half_screen_dva(cfg.screen_width_cm, cfg.viewing_distance_cm)
    y_half_dva = half_screen_dva(cfg.screen_height_cm, cfg.viewing_distance_cm)
    data["x_norm"] = (data["x_dva"] / (2.0 * x_half_dva)) + 0.5
    data["y_norm"] = (data["y_dva"] / (2.0 * y_half_dva)) + 0.5

    valid = data["timestamp_ms"].notna() & data["x_dva"].notna() & data["y_dva"].notna()
    valid &= data["validity"].eq(0)
    valid &= data["x_norm"].between(0.0, 1.0, inclusive="both")
    valid &= data["y_norm"].between(0.0, 1.0, inclusive="both")
    data = data.loc[valid].sort_values("timestamp_ms").reset_index(drop=True)

    # The shared I-DT implementation uses x_px/y_px names for dispersion. For
    # GazeBase these columns intentionally carry DVA coordinates during detection.
    data["x_px"] = data["x_dva"]
    data["y_px"] = data["y_dva"]
    return data[["timestamp_ms", "x_px", "y_px", "x_norm", "y_norm", "x_dva", "y_dva"]]


def finalize_fixation_coordinates(fixations: pd.DataFrame, cfg: GazeBaseAdapterConfig) -> pd.DataFrame:
    out = fixations.rename(columns={"x_px": "x_dva", "y_px": "y_dva"}).copy()
    out["x_px"] = out["x_norm"] * float(cfg.screen_width_px)
    out["y_px"] = out["y_norm"] * float(cfg.screen_height_px)
    return out


def half_screen_dva(screen_size_cm: float, viewing_distance_cm: float) -> float:
    return float(np.degrees(np.arctan2(screen_size_cm / 2.0, viewing_distance_cm)))


def ordered_gazebase_event_columns(events: pd.DataFrame) -> list[str]:
    preferred = [
        "dataset_id",
        "subject_id",
        "split",
        "fold",
        "label",
        "trial_id",
        "round_id",
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
