from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a fixed subject-level split for self-supervised encoder pretraining.")
    parser.add_argument("--events", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=float, default=0.80)
    parser.add_argument("--valid-size", type=float, default=0.10)
    parser.add_argument("--test-size", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_ratios(args.train_size, args.valid_size, args.test_size)
    events = pd.read_csv(args.events, dtype={"subject_id": str}, low_memory=False)
    subject_table = (
        events[["dataset_id", "subject_id", "label"]]
        .drop_duplicates(["dataset_id", "subject_id"])
        .sort_values(["dataset_id", "subject_id"])
        .reset_index(drop=True)
    )
    stratify = subject_table["dataset_id"] if subject_table["dataset_id"].nunique() > 1 else None
    train_df, temp_df = train_test_split(
        subject_table,
        train_size=args.train_size,
        random_state=args.seed,
        stratify=stratify,
        shuffle=True,
    )
    valid_fraction = args.valid_size / (args.valid_size + args.test_size)
    temp_stratify = temp_df["dataset_id"] if temp_df["dataset_id"].nunique() > 1 else None
    valid_df, test_df = train_test_split(
        temp_df,
        train_size=valid_fraction,
        random_state=args.seed,
        stratify=temp_stratify,
        shuffle=True,
    )

    split = pd.concat(
        [
            train_df.assign(split="train"),
            valid_df.assign(split="valid"),
            test_df.assign(split="test"),
        ],
        ignore_index=True,
    ).sort_values(["split", "dataset_id", "subject_id"])
    split["fold"] = "fixed_self_supervised"
    split = split[["dataset_id", "subject_id", "label", "fold", "split"]]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(output_path, index=False, encoding="utf-8-sig")
    summary = build_summary(split, args)
    output_path.with_name(output_path.stem + "_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def validate_ratios(train_size: float, valid_size: float, test_size: float) -> None:
    total = train_size + valid_size + test_size
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")
    if min(train_size, valid_size, test_size) <= 0:
        raise ValueError("Split ratios must be positive.")


def build_summary(split: pd.DataFrame, args: argparse.Namespace) -> dict:
    rows = []
    for split_name, group in split.groupby("split", sort=False):
        rows.append(
            {
                "split": split_name,
                "n_subjects": int(len(group)),
                "dataset_counts": {str(k): int(v) for k, v in group["dataset_id"].value_counts().sort_index().items()},
            }
        )
    return {
        "name": "self_supervised_subject_split",
        "seed": int(args.seed),
        "ratios": {
            "train": float(args.train_size),
            "valid": float(args.valid_size),
            "test": float(args.test_size),
        },
        "n_subjects": int(split[["dataset_id", "subject_id"]].drop_duplicates().shape[0]),
        "split_summary": rows,
        "protocol_notes": [
            "Subject-level split.",
            "Labels may be missing and are not required for self-supervised pretraining.",
            "For multi-dataset inputs, split is stratified by dataset_id.",
        ],
    }


if __name__ == "__main__":
    main()
