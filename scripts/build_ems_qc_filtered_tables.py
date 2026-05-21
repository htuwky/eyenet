from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build strict-QC and clipped-QC EMS event tables.")
    parser.add_argument("--events", default="data/processed/EMS/ems_events.csv")
    parser.add_argument("--qc-report", default="data/processed/EMS/qc/subject_qc_report.csv")
    parser.add_argument("--output-root", default="data/processed/EMS/filtered")
    parser.add_argument("--strict-name", default="strict_qc")
    parser.add_argument("--clipped-name", default="clipped_qc")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events = pd.read_csv(args.events, dtype={"subject_id": str}, low_memory=False)
    qc = pd.read_csv(args.qc_report, dtype={"subject_id": str})
    output_root = Path(args.output_root)

    strict_subjects = set(qc.loc[qc["usable_for_supervised_training"], "subject_id"].astype(str))
    labeled_subjects = set(qc.loc[qc["has_label"], "subject_id"].astype(str))

    strict_events = events[events["subject_id"].astype(str).isin(strict_subjects)].copy()
    strict_events = keep_in_range_coordinates_and_recompute_transitions(strict_events)
    strict_summary = summarize_variant(strict_events, "strict_qc")
    save_variant(output_root / args.strict_name, strict_events, strict_summary)

    clipped_events = events[events["subject_id"].astype(str).isin(labeled_subjects)].copy()
    clipped_events = clip_coordinates_and_recompute_transitions(clipped_events)
    clipped_summary = summarize_variant(clipped_events, "clipped_qc")
    save_variant(output_root / args.clipped_name, clipped_events, clipped_summary)

    print(json.dumps({"strict_qc": strict_summary, "clipped_qc": clipped_summary}, ensure_ascii=False, indent=2))


def clip_coordinates_and_recompute_transitions(events: pd.DataFrame) -> pd.DataFrame:
    data = events.copy()
    for column in ["x_norm", "y_norm", "x_dva", "y_dva", "duration_ms"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data["x_norm_original"] = data["x_norm"]
    data["y_norm_original"] = data["y_norm"]
    data["coordinate_was_clipped"] = (
        data["x_norm"].notna()
        & data["y_norm"].notna()
        & (~data["x_norm"].between(0.0, 1.0, inclusive="both") | ~data["y_norm"].between(0.0, 1.0, inclusive="both"))
    ).astype(int)

    data["x_norm"] = data["x_norm"].clip(lower=0.0, upper=1.0)
    data["y_norm"] = data["y_norm"].clip(lower=0.0, upper=1.0)

    # EMS DVA values are derived from normalized coordinates and fixed screen geometry.
    # Recompute them from the clipped coordinates to keep spatial and transition features consistent.
    width_px = 1024.0
    height_px = 768.0
    diagonal_in = 19.0
    viewing_distance_cm = 60.0
    diagonal_cm = diagonal_in * 2.54
    diagonal_px = np.hypot(width_px, height_px)
    width_cm = diagonal_cm * width_px / diagonal_px
    height_cm = diagonal_cm * height_px / diagonal_px
    x_cm = (data["x_norm"] - 0.5) * width_cm
    y_cm = (data["y_norm"] - 0.5) * height_cm
    data["x_dva"] = np.degrees(np.arctan2(x_cm, viewing_distance_cm))
    data["y_dva"] = np.degrees(np.arctan2(y_cm, viewing_distance_cm))

    return recompute_transitions_from_existing_coordinates(data)


def keep_in_range_coordinates_and_recompute_transitions(events: pd.DataFrame) -> pd.DataFrame:
    data = events.copy()
    data["x_norm"] = pd.to_numeric(data["x_norm"], errors="coerce")
    data["y_norm"] = pd.to_numeric(data["y_norm"], errors="coerce")
    in_range = (
        data["x_norm"].notna()
        & data["y_norm"].notna()
        & data["x_norm"].between(0.0, 1.0, inclusive="both")
        & data["y_norm"].between(0.0, 1.0, inclusive="both")
    )
    data = data.loc[in_range].copy()
    return recompute_transitions_from_existing_coordinates(data)


def recompute_transitions_from_existing_coordinates(events: pd.DataFrame) -> pd.DataFrame:
    data = events.copy()
    for column in ["x_norm", "y_norm", "x_dva", "y_dva", "duration_ms"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.sort_values(["subject_id", "trial_id", "event_index"]).reset_index(drop=True)
    group_cols = ["subject_id", "trial_id"]
    next_x_norm = data.groupby(group_cols)["x_norm"].shift(-1)
    next_y_norm = data.groupby(group_cols)["y_norm"].shift(-1)
    next_x_dva = data.groupby(group_cols)["x_dva"].shift(-1)
    next_y_dva = data.groupby(group_cols)["y_dva"].shift(-1)
    next_duration_ms = data.groupby(group_cols)["duration_ms"].shift(-1)

    data["saccade_dx_norm"] = next_x_norm - data["x_norm"]
    data["saccade_dy_norm"] = next_y_norm - data["y_norm"]
    data["saccade_amplitude_norm"] = np.hypot(data["saccade_dx_norm"], data["saccade_dy_norm"])
    data["saccade_angle"] = np.arctan2(data["saccade_dy_norm"], data["saccade_dx_norm"])
    data["saccade_dx_dva"] = next_x_dva - data["x_dva"]
    data["saccade_dy_dva"] = next_y_dva - data["y_dva"]
    data["saccade_amplitude_dva"] = np.hypot(data["saccade_dx_dva"], data["saccade_dy_dva"])
    next_duration_s = next_duration_ms / 1000.0
    data["transition_velocity_dva_s_approx"] = data["saccade_amplitude_dva"] / next_duration_s
    data.loc[next_duration_s <= 0, "transition_velocity_dva_s_approx"] = np.nan
    return data


def summarize_variant(events: pd.DataFrame, variant: str) -> dict:
    label_counts = events.drop_duplicates("subject_id")["label"].value_counts(dropna=False).sort_index()
    out_of_range = (
        events["x_norm"].notna()
        & events["y_norm"].notna()
        & (~events["x_norm"].between(0.0, 1.0, inclusive="both") | ~events["y_norm"].between(0.0, 1.0, inclusive="both"))
    )
    summary = {
        "variant": variant,
        "n_events": int(len(events)),
        "n_subjects": int(events["subject_id"].nunique()),
        "label_counts": {str(key): int(value) for key, value in label_counts.items()},
        "out_of_range_coordinate_events": int(out_of_range.sum()),
    }
    if "coordinate_was_clipped" in events.columns:
        summary["clipped_coordinate_events"] = int(events["coordinate_was_clipped"].sum())
    return summary


def save_variant(output_dir: Path, events: pd.DataFrame, summary: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    events.to_csv(output_dir / "ems_events.csv", index=False, encoding="utf-8-sig")
    (output_dir / "ems_events_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
