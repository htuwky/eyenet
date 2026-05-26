from __future__ import annotations

import argparse
import json
from pathlib import Path

from eyenet.data.crcns_eye1 import CRCNSEye1AdapterConfig, build_crcns_eye1_fixation_events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert CRCNS eye-1 raw gaze traces into EyeNet fixation events.")
    parser.add_argument("--raw-root", default="data/raw/CRCNS_eye1")
    parser.add_argument("--output", default="data/processed/CRCNS_eye1/crcns_eye1_fixation_events.csv")
    parser.add_argument("--summary", default="data/processed/CRCNS_eye1/crcns_eye1_fixation_events_summary.json")
    parser.add_argument("--file-report", default="data/processed/CRCNS_eye1/crcns_eye1_file_report.csv")
    parser.add_argument("--screen-width-px", type=float, default=640.0)
    parser.add_argument("--screen-height-px", type=float, default=480.0)
    parser.add_argument("--center-x-px", type=float, default=319.0)
    parser.add_argument("--center-y-px", type=float, default=239.0)
    parser.add_argument("--sampling-rate-hz", type=float, default=240.0)
    parser.add_argument("--pixels-per-degree", type=float, default=23.0)
    parser.add_argument("--trash-samples", type=int, default=260)
    parser.add_argument("--dispersion-threshold-dva", type=float, default=1.0)
    parser.add_argument("--min-duration-ms", type=float, default=80.0)
    parser.add_argument("--max-gap-ms", type=float, default=75.0)
    parser.add_argument("--max-files", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = CRCNSEye1AdapterConfig(
        raw_root=Path(args.raw_root),
        screen_width_px=args.screen_width_px,
        screen_height_px=args.screen_height_px,
        center_x_px=args.center_x_px,
        center_y_px=args.center_y_px,
        sampling_rate_hz=args.sampling_rate_hz,
        pixels_per_degree=args.pixels_per_degree,
        trash_samples=args.trash_samples,
        dispersion_threshold_dva=args.dispersion_threshold_dva,
        min_duration_ms=args.min_duration_ms,
        max_gap_ms=args.max_gap_ms,
        max_files=args.max_files,
    )
    events, outputs = build_crcns_eye1_fixation_events(cfg)

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
