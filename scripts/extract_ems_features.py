from __future__ import annotations

import argparse
import json
from pathlib import Path

from eyenet.data.features import extract_subject_features, load_events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract subject-level EMS baseline features.")
    parser.add_argument("--events", default="data/processed/EMS/ems_events.csv")
    parser.add_argument("--output-dir", default="data/processed/EMS")
    parser.add_argument("--include-pupil", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events = load_events(args.events, train_valid_only=True)
    features = extract_subject_features(events, include_pupil=args.include_pupil)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "with_pupil" if args.include_pupil else "no_pupil"
    output_path = output_dir / f"ems_subject_features_{suffix}.csv"
    features.to_csv(output_path, index=False, encoding="utf-8-sig")

    summary = {
        "n_subjects": int(len(features)),
        "n_features": int(features.shape[1] - 3),
        "include_pupil": bool(args.include_pupil),
        "output": str(output_path),
    }
    (output_dir / f"ems_subject_features_{suffix}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
