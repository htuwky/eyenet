from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from eyenet.data.ems import (
    build_test_manifest,
    build_train_manifest,
    load_ems_config,
    read_fixation_file,
    read_simple_yaml_mapping,
)
from eyenet.data.events import ScreenGeometry, add_transition_features, normalize_coordinates, summarize_event_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the EMS event-level table.")
    parser.add_argument("--config", default="configs/datasets/ems.yaml")
    parser.add_argument("--output-dir", default="data/processed/EMS")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_ems_config(args.config)
    raw_cfg = read_simple_yaml_mapping(Path(args.config))
    stimulus_cfg = raw_cfg["stimulus"]
    geometry = ScreenGeometry(
        width_px=float(stimulus_cfg["image_width_px"]),
        height_px=float(stimulus_cfg["image_height_px"]),
        diagonal_in=float(stimulus_cfg["screen_diagonal_in"]),
        viewing_distance_cm=float(stimulus_cfg["viewing_distance_cm"]),
    )

    manifest = pd.concat([build_train_manifest(cfg), build_test_manifest(cfg)], ignore_index=True)
    events = build_events_from_manifest(manifest, geometry)
    summary = summarize_event_table(events)
    summary["geometry"] = {
        "image_width_px": geometry.width_px,
        "image_height_px": geometry.height_px,
        "screen_diagonal_in": geometry.diagonal_in,
        "viewing_distance_cm": geometry.viewing_distance_cm,
        "screen_width_cm": geometry.width_cm,
        "screen_height_cm": geometry.height_cm,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    events.to_csv(output_dir / "ems_events.csv", index=False, encoding="utf-8-sig")
    (output_dir / "ems_events_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_events_from_manifest(manifest: pd.DataFrame, geometry: ScreenGeometry) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for row in manifest.itertuples(index=False):
        fixations = read_fixation_file(Path(row.file_path))
        subject_events = build_subject_events(fixations, row, geometry)
        parts.append(subject_events)
    return pd.concat(parts, ignore_index=True)


def build_subject_events(fixations: pd.DataFrame, manifest_row, geometry: ScreenGeometry) -> pd.DataFrame:
    df = fixations.copy()
    df = df.rename(
        columns={
            "IMAGE": "trial_id",
            "FIX_INDEX": "event_index",
            "FIX_DURATION": "duration_ms",
            "FIX_PUPIL": "pupil_optional",
        }
    )
    df["subject_id"] = manifest_row.subject_id
    df["split"] = manifest_row.split
    df["fold"] = manifest_row.fold
    df["label"] = manifest_row.label
    df["dataset_id"] = "EMS"
    df["event_type"] = "fixation"
    df["event_index"] = pd.to_numeric(df["event_index"], errors="coerce").astype("Int64")
    df["duration_ms"] = pd.to_numeric(df["duration_ms"], errors="coerce")
    df["pupil_optional"] = pd.to_numeric(df["pupil_optional"], errors="coerce")
    df = normalize_coordinates(df, geometry)
    df = df.sort_values(["subject_id", "trial_id", "event_index"]).reset_index(drop=True)
    df = add_transition_features(df)

    columns = [
        "dataset_id",
        "subject_id",
        "split",
        "fold",
        "label",
        "trial_id",
        "event_index",
        "event_type",
        "x_px",
        "y_px",
        "x_norm",
        "y_norm",
        "x_dva",
        "y_dva",
        "duration_ms",
        "pupil_optional",
        "saccade_dx_norm",
        "saccade_dy_norm",
        "saccade_amplitude_norm",
        "saccade_angle",
        "saccade_dx_dva",
        "saccade_dy_dva",
        "saccade_amplitude_dva",
        "transition_velocity_dva_s_approx",
    ]
    return df[columns]


if __name__ == "__main__":
    main()
