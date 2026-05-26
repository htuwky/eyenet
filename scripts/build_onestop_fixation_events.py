from __future__ import annotations

import argparse
import json
from pathlib import Path

from eyenet.data.onestop import OneStopAdapterConfig, build_onestop_fixation_events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert OneStop precomputed fixations into EyeNet event tables.")
    parser.add_argument("--events", default="data/raw/OneStop/precomputed_events/fixations_Paragraph.csv")
    parser.add_argument("--output", default="data/processed/OneStop/onestop_fixation_events.csv")
    parser.add_argument("--summary", default="data/processed/OneStop/onestop_fixation_events_summary.json")
    parser.add_argument("--file-report", default="data/processed/OneStop/onestop_trial_report.csv")
    parser.add_argument("--screen-width-px", type=float, default=2560.0)
    parser.add_argument("--screen-height-px", type=float, default=1440.0)
    parser.add_argument("--include-practice-trials", action="store_true")
    parser.add_argument("--include-repeated-reading-trials", action="store_true")
    parser.add_argument("--regime", choices=["all", "ordinary", "information_seeking"], default="all")
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OneStopAdapterConfig(
        events_path=Path(args.events),
        screen_width_px=args.screen_width_px,
        screen_height_px=args.screen_height_px,
        exclude_practice_trials=not args.include_practice_trials,
        exclude_repeated_reading_trials=not args.include_repeated_reading_trials,
        regime=args.regime,
        chunksize=args.chunksize,
        max_rows=args.max_rows,
    )
    events, outputs = build_onestop_fixation_events(cfg)

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
