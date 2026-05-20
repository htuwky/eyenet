from __future__ import annotations

import argparse
import json
from pathlib import Path

from eyenet.data.features import load_events
from eyenet.data.segment_features import (
    aggregate_segment_features,
    extract_segment_features,
    save_segment_feature_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract content-agnostic EMS segment features.")
    parser.add_argument("--events", default="data/processed/EMS/ems_events.csv")
    parser.add_argument("--output-dir", default="data/processed/EMS")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events = load_events(args.events, train_valid_only=False)
    segments = extract_segment_features(events)
    subject_features = aggregate_segment_features(segments)
    save_segment_feature_outputs(args.output_dir, segments, subject_features, suffix="no_pupil")

    summary = {
        "n_segments_total": int(len(segments)),
        "n_train_valid_segments": int((segments["split"] == "train_valid").sum()),
        "n_subjects": int(subject_features["subject_id"].nunique()),
        "n_subject_features": int(subject_features.shape[1] - 3),
        "segment_features": str(Path(args.output_dir) / "ems_segment_features_no_pupil.csv"),
        "subject_features": str(Path(args.output_dir) / "ems_subject_features_segment_agg_no_pupil.csv"),
    }
    (Path(args.output_dir) / "ems_segment_features_no_pupil_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
