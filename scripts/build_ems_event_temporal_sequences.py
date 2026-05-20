from __future__ import annotations

import argparse
import json

import pandas as pd

from eyenet.data.event_temporal_sequences import (
    build_event_temporal_sequences,
    save_event_temporal_sequence_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build EMS event-temporal fixation/saccade sequences without stimulus content.")
    parser.add_argument("--events", default="data/processed/EMS/ems_events.csv")
    parser.add_argument("--output", default="data/processed/EMS/ems_event_temporal_sequences_no_pupil.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events = pd.read_csv(args.events, dtype={"subject_id": str})
    event_temporal, summary = build_event_temporal_sequences(events)
    save_event_temporal_sequence_outputs(args.output, event_temporal, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
