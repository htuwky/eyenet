from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SEGMENT_FEATURE_COLUMNS = [
    "n_fixations",
    "short_fixation_ratio_lt150ms",
    "long_fixation_ratio_gt500ms",
    "fix_duration_ms_mean",
    "fix_duration_ms_std",
    "fix_duration_ms_median",
    "fix_duration_ms_iqr",
    "saccade_amp_norm_mean",
    "saccade_amp_norm_std",
    "saccade_amp_norm_median",
    "saccade_amp_norm_iqr",
    "scanpath_length_norm",
    "spatial_x_range",
    "spatial_y_range",
    "spatial_bbox_area",
    "center_bias_mean",
    "bcea_norm",
    "spatial_coverage_8x8",
    "transition_angle_entropy_8bin",
]

EVENT_FEATURE_COLUMNS = [
    "duration_ms",
    "log_duration_ms",
    "saccade_amplitude_norm",
    "transition_missing",
    "x_norm",
    "y_norm",
]

AGGREGATIONS = ["mean", "std", "median", "iqr", "p10", "p90"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build content-agnostic EMS subject-level summary features for the phase-2 "
            "encoder + summary-stream experiments."
        )
    )
    parser.add_argument("--segments", default="data/processed/EMS/ems_segment_features_no_pupil.csv")
    parser.add_argument(
        "--events",
        default="data/processed/EMS/encoder_ready/clipped_qc_no_position/ems_encoder_events.csv",
    )
    parser.add_argument("--output", default="data/processed/EMS/ems_subject_summary_features.csv")
    parser.add_argument(
        "--labeled-only",
        action="store_true",
        help="Keep only subjects with a non-missing label. Default keeps unlabeled rows with empty labels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    segments = pd.read_csv(args.segments, dtype={"subject_id": str}, low_memory=False)
    events = pd.read_csv(args.events, dtype={"subject_id": str}, low_memory=False)
    segments["subject_id"] = segments["subject_id"].map(normalize_subject_id)
    events["subject_id"] = events["subject_id"].map(normalize_subject_id)

    segment_summary = summarize_segments(segments)
    event_summary = summarize_events(events)
    labels = collect_subject_labels(segments, events)

    summary = labels.merge(segment_summary, on="subject_id", how="outer")
    summary = summary.merge(event_summary, on="subject_id", how="outer")
    summary = summary.sort_values("subject_id").reset_index(drop=True)
    if args.labeled_only:
        summary = summary[summary["label"].notna()].reset_index(drop=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False, encoding="utf-8-sig")

    feature_columns = [col for col in summary.columns if col not in {"subject_id", "label"}]
    print(f"wrote: {output_path}")
    print(f"subjects: {summary['subject_id'].nunique()}")
    print(f"labeled_subjects: {summary['label'].notna().sum()}")
    print(f"feature_columns: {len(feature_columns)}")
    print("first_columns:")
    for col in summary.columns[:20]:
        print(f"  {col}")


def normalize_subject_id(value: object) -> str:
    text = str(value)
    return text.zfill(3) if text.isdigit() else text


def summarize_segments(segments: pd.DataFrame) -> pd.DataFrame:
    available = [col for col in SEGMENT_FEATURE_COLUMNS if col in segments.columns]
    rows: list[dict[str, float | str | int]] = []
    for subject_id, subject_df in segments.groupby("subject_id", sort=True):
        row: dict[str, float | str | int] = {
            "subject_id": subject_id,
            "summary_n_segments": int(len(subject_df)),
        }
        for col in available:
            values = pd.to_numeric(subject_df[col], errors="coerce").to_numpy(dtype=float)
            add_aggregates(row, f"seg__{col}", values)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_events(events: pd.DataFrame) -> pd.DataFrame:
    events = events.copy()
    if "x_norm" in events.columns and "y_norm" in events.columns:
        x = pd.to_numeric(events["x_norm"], errors="coerce")
        y = pd.to_numeric(events["y_norm"], errors="coerce")
        events["center_distance_norm"] = np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)
    if "saccade_angle_sin" in events.columns and "saccade_angle_cos" in events.columns:
        sin = pd.to_numeric(events["saccade_angle_sin"], errors="coerce")
        cos = pd.to_numeric(events["saccade_angle_cos"], errors="coerce")
        events["saccade_angle_rad"] = np.arctan2(sin, cos)

    numeric_columns = [col for col in EVENT_FEATURE_COLUMNS if col in events.columns]
    if "center_distance_norm" in events.columns:
        numeric_columns.append("center_distance_norm")

    rows: list[dict[str, float | str | int]] = []
    for subject_id, subject_df in events.groupby("subject_id", sort=True):
        row: dict[str, float | str | int] = {
            "subject_id": subject_id,
            "summary_n_events": int(len(subject_df)),
            "summary_n_event_segments": int(subject_df["segment_index"].nunique())
            if "segment_index" in subject_df.columns
            else np.nan,
        }
        if "segment_index" in subject_df.columns:
            counts = subject_df.groupby("segment_index").size().to_numpy(dtype=float)
            add_aggregates(row, "event__events_per_segment", counts)
        if "saccade_angle_rad" in subject_df.columns:
            angles = pd.to_numeric(subject_df["saccade_angle_rad"], errors="coerce").dropna().to_numpy(dtype=float)
            row["event__saccade_angle_entropy_8bin"] = angle_entropy(angles, bins=8)
        for col in numeric_columns:
            values = pd.to_numeric(subject_df[col], errors="coerce").to_numpy(dtype=float)
            add_aggregates(row, f"event__{col}", values)
        rows.append(row)
    return pd.DataFrame(rows)


def collect_subject_labels(segments: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    label_rows = []
    subject_ids = sorted(set(segments["subject_id"]) | set(events["subject_id"]))
    for subject_id in subject_ids:
        labels = []
        for frame in [segments, events]:
            if "label" not in frame.columns:
                continue
            values = pd.to_numeric(
                frame.loc[frame["subject_id"] == subject_id, "label"],
                errors="coerce",
            ).dropna()
            labels.extend(values.astype(int).unique().tolist())
        unique = sorted(set(labels))
        if len(unique) > 1:
            raise ValueError(f"Subject {subject_id} has conflicting labels: {unique}")
        label_rows.append({"subject_id": subject_id, "label": unique[0] if unique else np.nan})
    return pd.DataFrame(label_rows)


def add_aggregates(row: dict[str, float | str | int], prefix: str, values: np.ndarray) -> None:
    values = values[np.isfinite(values)]
    if values.size == 0:
        for agg in AGGREGATIONS:
            row[f"{prefix}__{agg}"] = np.nan
        return
    row[f"{prefix}__mean"] = float(np.mean(values))
    row[f"{prefix}__std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    row[f"{prefix}__median"] = float(np.median(values))
    row[f"{prefix}__iqr"] = float(np.percentile(values, 75) - np.percentile(values, 25))
    row[f"{prefix}__p10"] = float(np.percentile(values, 10))
    row[f"{prefix}__p90"] = float(np.percentile(values, 90))


def angle_entropy(angles: np.ndarray, bins: int = 8) -> float:
    if angles.size == 0:
        return np.nan
    counts, _ = np.histogram(angles, bins=bins, range=(-np.pi, np.pi))
    probs = counts.astype(float) / max(counts.sum(), 1)
    probs = probs[probs > 0]
    if probs.size == 0:
        return np.nan
    return float(-(probs * np.log2(probs)).sum())


if __name__ == "__main__":
    main()
