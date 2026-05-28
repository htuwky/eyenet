from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eyenet.data.fixation_detection import IDTFixationConfig, detect_fixations_idt, prepare_raw_gaze_samples
from eyenet.data.hbn import add_transition_and_encoder_columns, describe_numeric

EYE_FILE_RE = re.compile(
    r"^CRCNS-DataShare/(?P<experiment>data-orig|data-mtv)/(?P<subject_dir>[^/]+)/(?P<clip_stem>[^/]+)\.e-ceyeS$"
)


@dataclass(frozen=True)
class CRCNSEye1AdapterConfig:
    raw_root: Path
    screen_width_px: float = 640.0
    screen_height_px: float = 480.0
    center_x_px: float = 319.0
    center_y_px: float = 239.0
    sampling_rate_hz: float = 240.0
    pixels_per_degree: float = 23.0
    trash_samples: int = 260
    dispersion_threshold_dva: float = 1.0
    min_duration_ms: float = 80.0
    max_gap_ms: float = 75.0
    max_files: int | None = None


def build_crcns_eye1_fixation_events(cfg: CRCNSEye1AdapterConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    eye_files = find_eye_files(cfg.raw_root)
    if cfg.max_files is not None:
        eye_files = eye_files[: cfg.max_files]
    if not eye_files:
        raise FileNotFoundError(f"No CRCNS eye-1 .e-ceyeS files found under {cfg.raw_root}")

    event_tables: list[pd.DataFrame] = []
    file_rows: list[dict[str, Any]] = []
    for zip_path, member in eye_files:
        parsed = parse_eye_member(member)
        raw, metadata = read_eye_file(zip_path, member)
        sampling_rate_hz = float(metadata.get("sampling_rate_hz", cfg.sampling_rate_hz))
        pixels_per_degree = float(metadata.get("pixels_per_degree", cfg.pixels_per_degree))
        trash_samples = int(metadata.get("trash_samples", cfg.trash_samples))

        samples = prepare_crcns_samples(raw, cfg, sampling_rate_hz=sampling_rate_hz, trash_samples=trash_samples)
        idt_cfg = IDTFixationConfig(
            sampling_rate_hz=sampling_rate_hz,
            dispersion_threshold_px=float(cfg.dispersion_threshold_dva) * pixels_per_degree,
            min_duration_ms=cfg.min_duration_ms,
            max_gap_ms=cfg.max_gap_ms,
        )
        fixations = detect_fixations_idt(samples, idt_cfg)
        experiment = parsed["experiment"]
        subject_dir = parsed["subject_dir"]
        clip_stem = parsed["clip_stem"]
        experiment_id = "orig" if experiment == "data-orig" else "mtv"
        subject_id = f"{experiment_id}_{subject_dir}"
        trial_id = f"{experiment_id}_{clip_stem}"

        file_rows.append(
            {
                "zip_file": str(zip_path),
                "member": member,
                "subject_id": subject_id,
                "experiment_id": experiment_id,
                "clip_stem": clip_stem,
                "trial_id": trial_id,
                "n_raw_rows": int(len(raw)),
                "n_valid_samples": int(len(samples)),
                "n_fixations": int(len(fixations)),
                "valid_sample_rate": float(len(samples) / len(raw)) if len(raw) else 0.0,
                "sampling_rate_hz": sampling_rate_hz,
                "pixels_per_degree": pixels_per_degree,
                "trash_samples": trash_samples,
            }
        )
        if fixations.empty:
            continue

        fixations = add_crcns_dva(fixations, cfg, pixels_per_degree=pixels_per_degree)
        fixations = fixations.assign(
            dataset_id="CRCNS_eye1",
            subject_id=subject_id,
            label=np.nan,
            trial_id=trial_id,
            experiment_id=experiment_id,
            clip_stem=clip_stem,
            event_type="fixation",
            split="unassigned",
            fold="unassigned",
            session_id=experiment_id,
            task_id="natural_video_viewing" if experiment_id == "orig" else "mtv_video_viewing",
        )
        event_tables.append(fixations)

    if not event_tables:
        raise ValueError("CRCNS eye-1 adapter did not detect any fixation events.")

    events = pd.concat(event_tables, ignore_index=True)
    events = add_transition_and_encoder_columns(events)
    events = events.sort_values(["subject_id", "trial_id", "event_index"]).reset_index(drop=True)
    events = events[ordered_crcns_event_columns(events)]

    file_report = pd.DataFrame(file_rows)
    summary = {
        "dataset_id": "CRCNS_eye1",
        "n_eye_files": int(len(eye_files)),
        "n_files_with_fixations": int(file_report["n_fixations"].gt(0).sum()),
        "n_subjects": int(events["subject_id"].nunique()),
        "n_trials": int(events[["subject_id", "trial_id"]].drop_duplicates().shape[0]),
        "n_fixation_events": int(len(events)),
        "screen_width_px": float(cfg.screen_width_px),
        "screen_height_px": float(cfg.screen_height_px),
        "sampling_rate_hz": float(file_report["sampling_rate_hz"].median()),
        "pixels_per_degree": float(file_report["pixels_per_degree"].median()),
        "trash_samples": int(file_report["trash_samples"].median()),
        "idt": {
            "dispersion_threshold_dva": float(cfg.dispersion_threshold_dva),
            "dispersion_threshold_px": float(cfg.dispersion_threshold_dva * file_report["pixels_per_degree"].median()),
            "min_duration_ms": float(cfg.min_duration_ms),
            "max_gap_ms": float(cfg.max_gap_ms),
        },
        "experiment_counts_files": {
            str(key): int(value) for key, value in file_report["experiment_id"].value_counts().items()
        },
        "experiment_counts_events": {
            str(key): int(value) for key, value in events["experiment_id"].value_counts().items()
        },
        "valid_sample_rate_mean": float(file_report["valid_sample_rate"].mean()),
        "fixations_per_file": describe_numeric(file_report["n_fixations"]),
        "events_per_subject": describe_numeric(events.groupby("subject_id").size()),
        "duration_ms": describe_numeric(events["duration_ms"]),
    }
    return events, {"summary": summary, "file_report": file_report}


def find_eye_files(raw_root: Path) -> list[tuple[Path, str]]:
    root = Path(raw_root)
    candidates = [root / "downloads", root]
    zip_paths: list[Path] = []
    for candidate in candidates:
        if candidate.exists():
            zip_paths = sorted(candidate.glob("crcns-eye1-*.zip"))
            if zip_paths:
                break

    eye_files: list[tuple[Path, str]] = []
    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.namelist():
                if EYE_FILE_RE.match(member):
                    eye_files.append((zip_path, member))
    return sorted(eye_files, key=lambda item: (item[0].name, item[1]))


def parse_eye_member(member: str) -> dict[str, str]:
    match = EYE_FILE_RE.match(member)
    if match is None:
        raise ValueError(f"Unexpected CRCNS eye file path: {member}")
    return match.groupdict()


def read_eye_file(zip_path: Path, member: str) -> tuple[pd.DataFrame, dict[str, float | int]]:
    with zipfile.ZipFile(zip_path) as archive:
        text = archive.read(member).decode("latin-1", errors="replace")

    lines = text.splitlines()
    metadata = parse_eye_header(lines[:3])
    payload = "\n".join(lines[3:])
    raw = pd.read_csv(
        io.StringIO(payload),
        sep=r"\s+",
        names=["x_px", "y_px", "unknown_1", "flag", "aux_x", "aux_y", "aux_value", "aux_flag"],
        engine="python",
    )
    return raw, metadata


def parse_eye_header(lines: list[str]) -> dict[str, float | int]:
    metadata: dict[str, float | int] = {}
    for line in lines:
        if "=" not in line:
            continue
        key, value = [part.strip() for part in line.split("=", maxsplit=1)]
        match = re.search(r"-?\d+(?:\.\d+)?", value)
        if match is None:
            continue
        number = float(match.group(0))
        if key == "period":
            metadata["sampling_rate_hz"] = number
        elif key == "ppd":
            metadata["pixels_per_degree"] = number
        elif key == "trash":
            metadata["trash_samples"] = int(number)
    return metadata


def prepare_crcns_samples(
    raw: pd.DataFrame,
    cfg: CRCNSEye1AdapterConfig,
    *,
    sampling_rate_hz: float,
    trash_samples: int,
) -> pd.DataFrame:
    data = raw.iloc[trash_samples:].reset_index(drop=True).copy()
    data["sample_index"] = np.arange(len(data), dtype=int)
    return prepare_raw_gaze_samples(
        data,
        x_column="x_px",
        y_column="y_px",
        time_column="sample_index",
        screen_width_px=cfg.screen_width_px,
        screen_height_px=cfg.screen_height_px,
        sampling_rate_hz=sampling_rate_hz,
        time_unit="sample",
        invalid_zero=True,
    )


def add_crcns_dva(events: pd.DataFrame, cfg: CRCNSEye1AdapterConfig, *, pixels_per_degree: float) -> pd.DataFrame:
    out = events.copy()
    out["x_dva"] = (out["x_px"] - float(cfg.center_x_px)) / pixels_per_degree
    out["y_dva"] = (out["y_px"] - float(cfg.center_y_px)) / pixels_per_degree
    return out


def ordered_crcns_event_columns(events: pd.DataFrame) -> list[str]:
    preferred = [
        "dataset_id",
        "subject_id",
        "split",
        "fold",
        "label",
        "trial_id",
        "experiment_id",
        "clip_stem",
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
