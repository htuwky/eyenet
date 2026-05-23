from __future__ import annotations

import argparse
import json
from pathlib import Path

from eyenet.data.hbn import HBNAdapterConfig, build_hbn_fixation_events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert HBN raw gaze CSV files into EyeNet fixation event tables.")
    parser.add_argument("--raw-dir", default="data/raw/HBN/downloads/data")
    parser.add_argument("--output", default="data/processed/HBN/hbn_fixation_events.csv")
    parser.add_argument("--summary", default="data/processed/HBN/hbn_fixation_events_summary.json")
    parser.add_argument("--file-report", default="data/processed/HBN/hbn_file_report.csv")
    parser.add_argument("--sampling-rate-hz", type=float, default=120.0)
    parser.add_argument("--screen-width-px", type=float, default=800.0)
    parser.add_argument("--screen-height-px", type=float, default=600.0)
    parser.add_argument("--screen-width-cm", type=float, default=33.8)
    parser.add_argument("--screen-height-cm", type=float, default=27.0)
    parser.add_argument("--viewing-distance-cm", type=float, default=63.5)
    parser.add_argument("--time-unit", choices=["sample", "ms"], default="sample")
    parser.add_argument("--dispersion-threshold-px", type=float, default=35.0)
    parser.add_argument("--min-duration-ms", type=float, default=80.0)
    parser.add_argument("--max-gap-ms", type=float, default=75.0)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--keep-zero-zero", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = HBNAdapterConfig(
        raw_dir=Path(args.raw_dir),
        sampling_rate_hz=args.sampling_rate_hz,
        screen_width_px=args.screen_width_px,
        screen_height_px=args.screen_height_px,
        screen_width_cm=args.screen_width_cm,
        screen_height_cm=args.screen_height_cm,
        viewing_distance_cm=args.viewing_distance_cm,
        time_unit=args.time_unit,
        dispersion_threshold_px=args.dispersion_threshold_px,
        min_duration_ms=args.min_duration_ms,
        max_gap_ms=args.max_gap_ms,
        invalid_zero=not args.keep_zero_zero,
        max_files=args.max_files,
    )
    events, outputs = build_hbn_fixation_events(cfg)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(output_path, index=False, encoding="utf-8-sig")

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(outputs["summary"], ensure_ascii=False, indent=2), encoding="utf-8")

    file_report_path = Path(args.file_report)
    file_report_path.parent.mkdir(parents=True, exist_ok=True)
    outputs["file_report"].to_csv(file_report_path, index=False, encoding="utf-8-sig")

    print(json.dumps(outputs["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
