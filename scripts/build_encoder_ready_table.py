from __future__ import annotations

import argparse
import json

import pandas as pd

from eyenet.data.encoder_ready import (
    build_encoder_ready_table,
    load_feature_schema,
    save_encoder_ready_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an encoder-ready event table from an event-temporal sequence table.")
    parser.add_argument("--input", default="data/processed/EMS/filtered/clipped_qc/ems_event_temporal_sequences_no_pupil.csv")
    parser.add_argument("--schema", default="configs/features/encoder_original_13feature_core.json")
    parser.add_argument("--output-dir", default="data/processed/EMS/encoder_ready/clipped_qc_no_position")
    parser.add_argument("--table-name", default="ems_encoder_events.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    schema = load_feature_schema(args.schema)
    events = pd.read_csv(args.input, dtype={"subject_id": str}, low_memory=False)
    table, summary = build_encoder_ready_table(events, schema)
    save_encoder_ready_outputs(args.output_dir, table, summary, schema, table_name=args.table_name)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
