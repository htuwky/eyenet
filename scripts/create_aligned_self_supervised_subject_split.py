from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a self-supervised subject split while preserving an existing "
            "downstream split for one anchor dataset."
        )
    )
    parser.add_argument("--events", required=True)
    parser.add_argument("--anchor-split", required=True)
    parser.add_argument("--anchor-dataset", default="EMS")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=float, default=0.60)
    parser.add_argument("--valid-size", type=float, default=0.20)
    parser.add_argument("--test-size", type=float, default=0.20)
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
    subject_table["_subject_key"] = subject_table.apply(subject_key, axis=1)

    anchor_split = pd.read_csv(args.anchor_split, dtype={"subject_id": str}, low_memory=False)
    if "dataset_id" not in anchor_split.columns:
        anchor_split["dataset_id"] = args.anchor_dataset
    anchor_split = anchor_split[["dataset_id", "subject_id", "split"]].copy()
    anchor_split["_subject_key"] = anchor_split.apply(subject_key, axis=1)

    split_parts = [build_anchor_split(subject_table, anchor_split, args.anchor_dataset)]
    non_anchor = subject_table[subject_table["dataset_id"] != args.anchor_dataset].copy()
    if not non_anchor.empty:
        split_parts.append(split_non_anchor(non_anchor, args))

    split = (
        pd.concat(split_parts, ignore_index=True)
        .sort_values(["split", "dataset_id", "subject_id"])
        .reset_index(drop=True)
    )
    split["fold"] = "aligned_self_supervised"
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


def subject_key(row: pd.Series) -> str:
    subject_id = str(row["subject_id"]).strip()
    if subject_id.isdigit():
        subject_id = subject_id.zfill(3)
    return f"{row['dataset_id']}::{subject_id}"


def build_anchor_split(subject_table: pd.DataFrame, anchor_split: pd.DataFrame, anchor_dataset: str) -> pd.DataFrame:
    anchor_subjects = subject_table[subject_table["dataset_id"] == anchor_dataset].copy()
    merged = anchor_subjects.merge(
        anchor_split[["_subject_key", "split"]],
        on="_subject_key",
        how="left",
        validate="one_to_one",
    )
    if merged["split"].isna().any():
        missing = merged.loc[merged["split"].isna(), "subject_id"].tolist()
        raise ValueError(f"Anchor split is missing {len(missing)} {anchor_dataset} subjects: {missing[:10]}")
    return merged[["dataset_id", "subject_id", "label", "split"]]


def split_non_anchor(subject_table: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
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
    return pd.concat(
        [
            train_df.assign(split="train"),
            valid_df.assign(split="valid"),
            test_df.assign(split="test"),
        ],
        ignore_index=True,
    )[["dataset_id", "subject_id", "label", "split"]]


def build_summary(split: pd.DataFrame, args: argparse.Namespace) -> dict:
    rows = []
    for split_name, group in split.groupby("split", sort=False):
        rows.append(
            {
                "split": str(split_name),
                "n_subjects": int(len(group)),
                "dataset_counts": {str(k): int(v) for k, v in group["dataset_id"].value_counts().sort_index().items()},
            }
        )
    return {
        "name": "aligned_self_supervised_subject_split",
        "seed": int(args.seed),
        "anchor_dataset": args.anchor_dataset,
        "anchor_split": args.anchor_split,
        "ratios_for_non_anchor_datasets": {
            "train": float(args.train_size),
            "valid": float(args.valid_size),
            "test": float(args.test_size),
        },
        "n_subjects": int(split[["dataset_id", "subject_id"]].drop_duplicates().shape[0]),
        "split_summary": rows,
        "protocol_notes": [
            "Subject-level split.",
            "Anchor dataset split is copied from the downstream split file.",
            "Non-anchor datasets are split by subject and stratified by dataset_id when applicable.",
            "Use this split when target-dataset test subjects must stay unseen during self-supervised pretraining.",
        ],
    }


if __name__ == "__main__":
    main()
