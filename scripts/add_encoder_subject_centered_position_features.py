from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DERIVED_COLUMNS = [
    "x_subject_centered_norm",
    "y_subject_centered_norm",
    "subject_centered_position_radius_norm",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Append subject-centered position features to encoder-ready event tables. "
            "The original x_norm/y_norm columns are preserved in the output so schemas can decide whether to use them."
        )
    )
    parser.add_argument("--input", required=True, help="Input encoder-ready events CSV.")
    parser.add_argument("--output", required=True, help="Output CSV with appended derived columns.")
    parser.add_argument(
        "--group-columns",
        default="dataset_id,subject_id",
        help="Comma-separated grouping columns used to compute the position center.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite the output file if it already exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    group_columns = [column.strip() for column in args.group_columns.split(",") if column.strip()]

    if not input_path.exists():
        raise FileNotFoundError(f"Input events not found: {input_path}")
    if output_path.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {output_path}. Use --force to overwrite.")

    events = pd.read_csv(input_path, dtype={"subject_id": str}, low_memory=False)
    add_subject_centered_position_features(events, group_columns)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(events):,} rows with {len(events.columns)} columns to {output_path}")
    print("Added columns: " + ", ".join(DERIVED_COLUMNS))


def add_subject_centered_position_features(events: pd.DataFrame, group_columns: list[str]) -> None:
    required_columns = {"x_norm", "y_norm", *group_columns}
    missing = sorted(required_columns - set(events.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    centers = events.groupby(group_columns, dropna=False)[["x_norm", "y_norm"]].transform("median")
    events["x_subject_centered_norm"] = events["x_norm"] - centers["x_norm"]
    events["y_subject_centered_norm"] = events["y_norm"] - centers["y_norm"]
    events["subject_centered_position_radius_norm"] = np.sqrt(
        events["x_subject_centered_norm"].pow(2) + events["y_subject_centered_norm"].pow(2)
    )


if __name__ == "__main__":
    main()
