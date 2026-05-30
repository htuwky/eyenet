from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from eyenet.data.encoder_ready import describe_series, load_feature_schema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine multiple encoder-ready fixation event tables.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Encoder-ready CSV files to combine.")
    parser.add_argument("--schema", default="configs/features/encoder_original_13feature_core.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    schema = load_feature_schema(args.schema)
    required_cols = list(schema["metadata_columns"]) + list(schema["feature_columns"])
    frames = []
    for input_path in args.inputs:
        frame = pd.read_csv(input_path, dtype={"subject_id": str}, low_memory=False)
        missing = [column for column in required_cols if column not in frame.columns]
        if missing:
            raise ValueError(f"{input_path} is missing required columns: {missing}")
        frames.append(frame[required_cols].copy())

    combined = pd.concat(frames, ignore_index=True)
    combined["dataset_id"] = combined["dataset_id"].astype(str)
    combined["subject_id"] = combined["subject_id"].astype(str)
    combined = combined.sort_values(["dataset_id", "subject_id", "segment_index", "event_index_in_segment"]).reset_index(drop=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False, encoding="utf-8-sig")

    summary = summarize_combined(combined, schema)
    summary_path = Path(args.summary) if args.summary else output_path.with_name(output_path.stem + "_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def summarize_combined(data: pd.DataFrame, schema: dict) -> dict:
    feature_cols = list(schema["feature_columns"])
    subject_table = data[["dataset_id", "subject_id", "label"]].drop_duplicates(["dataset_id", "subject_id"])
    subject_counts = data.groupby(["dataset_id", "subject_id"]).size()
    return {
        "schema_name": schema.get("name"),
        "schema_version": schema.get("version"),
        "n_datasets": int(data["dataset_id"].nunique()),
        "dataset_counts_subjects": {
            str(key): int(value) for key, value in subject_table["dataset_id"].value_counts().sort_index().items()
        },
        "dataset_counts_events": {
            str(key): int(value) for key, value in data["dataset_id"].value_counts().sort_index().items()
        },
        "n_subjects": int(len(subject_table)),
        "n_events": int(len(data)),
        "n_features": int(len(feature_cols)),
        "feature_columns": feature_cols,
        "label_counts": {str(key): int(value) for key, value in subject_table["label"].value_counts(dropna=False).sort_index().items()},
        "events_per_subject": describe_series(subject_counts),
        "content_fields_used_as_features": [],
    }


if __name__ == "__main__":
    main()
