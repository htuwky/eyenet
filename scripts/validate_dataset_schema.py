from __future__ import annotations

import argparse
import json
from pathlib import Path

from eyenet.data.schema import validate_event_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a processed event table against the EyeNet schema.")
    parser.add_argument("--events", default="data/processed/EMS/ems_events.csv")
    parser.add_argument("--output", default="data/processed/EMS/ems_events_schema_report.json")
    parser.add_argument("--require-label", action="store_true", help="Fail if labels contain missing values.")
    parser.add_argument(
        "--strict-normalized-range",
        action="store_true",
        help="Fail if x_norm/y_norm contain values outside [0, 1].",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_event_table(
        args.events,
        require_label=args.require_label,
        strict_normalized_range=args.strict_normalized_range,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
