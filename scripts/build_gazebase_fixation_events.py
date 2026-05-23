from __future__ import annotations

import argparse
import json
from pathlib import Path

from eyenet.data.gazebase import GazeBaseAdapterConfig, build_gazebase_fixation_events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert GazeBase raw DVA gaze samples into EyeNet fixation events.")
    parser.add_argument("--raw-root", default="data/raw/GazeBase")
    parser.add_argument("--output", default="data/processed/GazeBase/gazebase_fixation_events.csv")
    parser.add_argument("--summary", default="data/processed/GazeBase/gazebase_fixation_events_summary.json")
    parser.add_argument("--file-report", default="data/processed/GazeBase/gazebase_file_report.csv")
    parser.add_argument("--sampling-rate-hz", type=float, default=1000.0)
    parser.add_argument("--screen-width-px", type=float, default=1680.0)
    parser.add_argument("--screen-height-px", type=float, default=1050.0)
    parser.add_argument("--screen-width-cm", type=float, default=47.4)
    parser.add_argument("--screen-height-cm", type=float, default=29.7)
    parser.add_argument("--viewing-distance-cm", type=float, default=55.0)
    parser.add_argument("--dispersion-threshold-dva", type=float, default=1.0)
    parser.add_argument("--min-duration-ms", type=float, default=80.0)
    parser.add_argument("--max-gap-ms", type=float, default=75.0)
    parser.add_argument("--tasks", default="VD1,VD2", help="Comma-separated task codes, or ALL.")
    parser.add_argument("--max-subject-zips", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = GazeBaseAdapterConfig(
        raw_root=Path(args.raw_root),
        sampling_rate_hz=args.sampling_rate_hz,
        screen_width_px=args.screen_width_px,
        screen_height_px=args.screen_height_px,
        screen_width_cm=args.screen_width_cm,
        screen_height_cm=args.screen_height_cm,
        viewing_distance_cm=args.viewing_distance_cm,
        dispersion_threshold_dva=args.dispersion_threshold_dva,
        min_duration_ms=args.min_duration_ms,
        max_gap_ms=args.max_gap_ms,
        tasks=parse_tasks(args.tasks),
        max_subject_zips=args.max_subject_zips,
    )
    events, outputs = build_gazebase_fixation_events(cfg)

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


def parse_tasks(value: str) -> tuple[str, ...] | None:
    if value.strip().upper() == "ALL":
        return None
    tasks = tuple(task.strip().upper() for task in value.split(",") if task.strip())
    if not tasks:
        raise ValueError("At least one task must be provided, or use --tasks ALL.")
    return tasks


if __name__ == "__main__":
    main()
