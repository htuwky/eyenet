from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_INPUT = "data/raw/OneStop/precomputed_events/fixations_Paragraph.csv"
DEFAULT_OUTPUT = "data/processed/OneStop/onestop_fixation_audit.json"

USE_COLUMNS = [
    "participant_id",
    "TRIAL_INDEX",
    "CURRENT_FIX_INDEX",
    "CURRENT_FIX_DURATION",
    "CURRENT_FIX_START",
    "CURRENT_FIX_END",
    "CURRENT_FIX_X",
    "CURRENT_FIX_Y",
    "CURRENT_FIX_X_RESOLUTION",
    "CURRENT_FIX_Y_RESOLUTION",
    "CURRENT_FIX_INTEREST_AREA_DATA",
    "ANSWER_LOCATIONS",
    "TOP_LEFT",
    "question_preview",
    "repeated_reading_trial",
    "practice_trial",
]

NUMERIC_COLUMNS = [
    "CURRENT_FIX_DURATION",
    "CURRENT_FIX_START",
    "CURRENT_FIX_END",
    "CURRENT_FIX_X",
    "CURRENT_FIX_Y",
    "CURRENT_FIX_X_RESOLUTION",
    "CURRENT_FIX_Y_RESOLUTION",
]

REGIME_COLUMNS = ["question_preview", "repeated_reading_trial", "practice_trial"]
GEOMETRY_COLUMNS = ["CURRENT_FIX_INTEREST_AREA_DATA", "ANSWER_LOCATIONS", "TOP_LEFT"]

NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit OneStop precomputed fixation fields for EyeNet conversion.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"OneStop fixation table does not exist: {input_path}")

    audit = audit_onestop_fixations(
        input_path=input_path,
        chunksize=args.chunksize,
        max_rows=args.max_rows,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def audit_onestop_fixations(input_path: Path, chunksize: int, max_rows: int | None = None) -> dict[str, Any]:
    n_rows = 0
    participants: set[str] = set()
    trials: set[tuple[str, str]] = set()
    missing_counts = Counter()
    numeric_values: dict[str, list[np.ndarray]] = {column: [] for column in NUMERIC_COLUMNS}
    resolution_pair_counts: Counter[tuple[str, str]] = Counter()
    regime_counts: dict[str, Counter[str]] = {column: Counter() for column in REGIME_COLUMNS}
    top_left_counts: Counter[str] = Counter()
    interest_area_bounds = BoundsAccumulator()
    answer_location_bounds = BoundsAccumulator()
    nonpositive_duration_count = 0
    inconsistent_time_count = 0

    reader = pd.read_csv(
        input_path,
        usecols=USE_COLUMNS,
        dtype={"participant_id": str},
        chunksize=chunksize,
        low_memory=False,
        na_values=["."],
    )

    for chunk in reader:
        if max_rows is not None:
            remaining = max_rows - n_rows
            if remaining <= 0:
                break
            chunk = chunk.head(remaining)
        if chunk.empty:
            continue

        n_rows += int(len(chunk))
        participants.update(chunk["participant_id"].dropna().astype(str).unique().tolist())
        trial_pairs = chunk[["participant_id", "TRIAL_INDEX"]].dropna().astype(str).drop_duplicates()
        trials.update((row.participant_id, row.TRIAL_INDEX) for row in trial_pairs.itertuples(index=False))

        for column in USE_COLUMNS:
            missing_counts[column] += int(chunk[column].isna().sum())

        numeric = chunk[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
        for column in NUMERIC_COLUMNS:
            values = numeric[column].dropna().to_numpy(dtype=float)
            if values.size:
                numeric_values[column].append(values)

        duration = numeric["CURRENT_FIX_DURATION"]
        nonpositive_duration_count += int((duration.dropna() <= 0.0).sum())

        start = numeric["CURRENT_FIX_START"]
        end = numeric["CURRENT_FIX_END"]
        time_valid = start.notna() & end.notna() & duration.notna()
        inconsistent_time_count += int(((end - start - duration).abs()[time_valid] > 1.0).sum())

        resolution_frame = numeric[["CURRENT_FIX_X_RESOLUTION", "CURRENT_FIX_Y_RESOLUTION"]].dropna().drop_duplicates()
        for row in resolution_frame.itertuples(index=False):
            resolution_pair_counts[(str(row.CURRENT_FIX_X_RESOLUTION), str(row.CURRENT_FIX_Y_RESOLUTION))] += 1

        top_left_counts.update(chunk["TOP_LEFT"].dropna().astype(str).tolist())
        for value in chunk["CURRENT_FIX_INTEREST_AREA_DATA"].dropna().astype(str).drop_duplicates():
            rect = parse_interest_area_rectangle(value)
            if rect is not None:
                x0, y0, x1, y1 = rect
                interest_area_bounds.update_points([(x0, y0), (x1, y1)])
        for value in chunk["ANSWER_LOCATIONS"].dropna().astype(str).drop_duplicates():
            answer_location_bounds.update_points(parse_points(value))

        for column in REGIME_COLUMNS:
            values = chunk[column].fillna("<missing>").astype(str)
            regime_counts[column].update(values.tolist())

    numeric_summary = {column: summarize_arrays(arrays) for column, arrays in numeric_values.items()}
    return {
        "input": str(input_path),
        "n_rows": n_rows,
        "n_participants": len(participants),
        "n_trials": len(trials),
        "missing_counts": dict(sorted(missing_counts.items())),
        "missing_rates": {
            column: (float(count) / n_rows if n_rows else np.nan)
            for column, count in sorted(missing_counts.items())
        },
        "numeric_summary": numeric_summary,
        "geometry_summary": {
            "interest_area_rectangle_bounds": interest_area_bounds.to_dict(),
            "answer_location_bounds": answer_location_bounds.to_dict(),
            "top_left_counts_top20": dict(top_left_counts.most_common(20)),
            "coordinate_normalization_note": (
                "CURRENT_FIX_X_RESOLUTION and CURRENT_FIX_Y_RESOLUTION are resolution/precision fields, "
                "not screen width/height. Do not use them as coordinate bounds."
            ),
        },
        "nonpositive_duration_rate": float(nonpositive_duration_count / n_rows) if n_rows else np.nan,
        "inconsistent_time_rate": float(inconsistent_time_count / n_rows) if n_rows else np.nan,
        "resolution_pair_counts_top20": {
            f"{key[0]}x{key[1]}": value for key, value in resolution_pair_counts.most_common(20)
        },
        "regime_counts": {
            column: dict(counter.most_common())
            for column, counter in regime_counts.items()
        },
        "columns_excluded_from_encoder_features": [
            "paragraph",
            "question",
            "answers_order",
            "answer_1",
            "answer_2",
            "answer_3",
            "answer_4",
            "selected_answer",
            "is_correct",
            "IA_LABEL",
            "universal_pos",
            "ptb_pos",
            "dependency_relation",
            "gpt2_surprisal",
            "wordfreq_frequency",
            "subtlex_frequency",
        ],
    }


class BoundsAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.min_x = np.inf
        self.max_x = -np.inf
        self.min_y = np.inf
        self.max_y = -np.inf

    def update_points(self, points: list[tuple[float, float]]) -> None:
        for x, y in points:
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            self.count += 1
            self.min_x = min(self.min_x, x)
            self.max_x = max(self.max_x, x)
            self.min_y = min(self.min_y, y)
            self.max_y = max(self.max_y, y)

    def to_dict(self) -> dict[str, float | int | None]:
        if self.count == 0:
            return {"count": 0, "min_x": None, "max_x": None, "min_y": None, "max_y": None}
        return {
            "count": self.count,
            "min_x": float(self.min_x),
            "max_x": float(self.max_x),
            "min_y": float(self.min_y),
            "max_y": float(self.max_y),
        }


def parse_interest_area_rectangle(value: str) -> tuple[float, float, float, float] | None:
    # Example: [STATIC, RECTANGLE, 358.0, 153.0, 510.0, 264.0]
    if "RECTANGLE" not in value:
        return None
    numbers = [float(item) for item in NUMBER_PATTERN.findall(value)]
    if len(numbers) < 4:
        return None
    return tuple(numbers[-4:])  # type: ignore[return-value]


def parse_points(value: str) -> list[tuple[float, float]]:
    # Example: [(930,414), (180,756), (1680,756), (930,1098)]
    numbers = [float(item) for item in NUMBER_PATTERN.findall(value)]
    return list(zip(numbers[0::2], numbers[1::2]))


def summarize_arrays(arrays: list[np.ndarray]) -> dict[str, float | int | None]:
    if not arrays:
        return {"count": 0, "min": None, "p25": None, "median": None, "mean": None, "p75": None, "max": None}
    values = np.concatenate(arrays)
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.5)),
        "mean": float(np.mean(values)),
        "p75": float(np.quantile(values, 0.75)),
        "max": float(np.max(values)),
    }


if __name__ == "__main__":
    main()
