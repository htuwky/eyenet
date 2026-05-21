from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from eyenet.data.qc import (
    SubjectQCConfig,
    build_subject_qc_report,
    save_subject_qc_outputs,
    summarize_subject_qc,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build subject-level QC reports from an EyeNet event table.")
    parser.add_argument("--events", default="data/processed/EMS/ems_events.csv")
    parser.add_argument("--output-dir", default="data/processed/EMS/qc")
    parser.add_argument("--min-events", type=int, default=100)
    parser.add_argument("--min-trials", type=int, default=5)
    parser.add_argument("--min-valid-coordinate-rate", type=float, default=0.95)
    parser.add_argument("--max-out-of-range-coordinate-rate", type=float, default=0.05)
    parser.add_argument("--max-missing-transition-rate", type=float, default=0.25)
    parser.add_argument("--max-nonpositive-duration-rate", type=float, default=0.01)
    parser.add_argument("--min-median-duration-ms", type=float, default=50.0)
    parser.add_argument("--max-median-duration-ms", type=float, default=2000.0)
    parser.add_argument("--allow-unlabeled-supervised", action="store_true")
    parser.add_argument("--require-label-for-self-supervised", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SubjectQCConfig(
        min_events=args.min_events,
        min_trials=args.min_trials,
        min_valid_coordinate_rate=args.min_valid_coordinate_rate,
        max_out_of_range_coordinate_rate=args.max_out_of_range_coordinate_rate,
        max_missing_transition_rate=args.max_missing_transition_rate,
        max_nonpositive_duration_rate=args.max_nonpositive_duration_rate,
        min_median_duration_ms=args.min_median_duration_ms,
        max_median_duration_ms=args.max_median_duration_ms,
        require_label_for_supervised=not args.allow_unlabeled_supervised,
        require_label_for_self_supervised=args.require_label_for_self_supervised,
    )
    events = pd.read_csv(args.events, low_memory=False)
    report = build_subject_qc_report(events, cfg)
    summary = summarize_subject_qc(report, cfg)
    save_subject_qc_outputs(args.output_dir, report, summary)

    print("Subject-level QC summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print()
    print("Subjects failing hard QC")
    failed = report[~report["hard_qc_pass"]]
    columns = [
        "dataset_id",
        "subject_id",
        "label",
        "n_events",
        "n_trials",
        "valid_coordinate_rate",
        "out_of_range_coordinate_rate",
        "missing_transition_rate",
        "qc_reasons",
    ]
    if failed.empty:
        print("None")
    else:
        print(failed[columns].to_string(index=False))
    print()
    print(f"Outputs written to: {Path(args.output_dir)}")


if __name__ == "__main__":
    main()
