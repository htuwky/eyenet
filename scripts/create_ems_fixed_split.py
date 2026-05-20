from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a fixed subject-level stratified EMS split.")
    parser.add_argument("--subjects", default="data/processed/EMS/ems_subject_features_segment_agg_no_pupil.csv")
    parser.add_argument("--output", default="data/splits/EMS/ems_subject_split_60_20_20_seed42.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=float, default=0.60)
    parser.add_argument("--valid-size", type=float, default=0.20)
    parser.add_argument("--test-size", type=float, default=0.20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_ratios(args.train_size, args.valid_size, args.test_size)

    subjects = pd.read_csv(args.subjects, dtype={"subject_id": str})
    subject_table = (
        subjects[["subject_id", "fold", "label"]]
        .drop_duplicates("subject_id")
        .rename(columns={"fold": "official_fold"})
        .sort_values("subject_id")
        .reset_index(drop=True)
    )
    subject_table["subject_id"] = subject_table["subject_id"].astype(str).str.zfill(3)
    subject_table["label"] = subject_table["label"].astype(int)

    train_df, temp_df = train_test_split(
        subject_table,
        train_size=args.train_size,
        random_state=args.seed,
        stratify=subject_table["label"],
        shuffle=True,
    )
    valid_fraction_of_temp = args.valid_size / (args.valid_size + args.test_size)
    valid_df, test_df = train_test_split(
        temp_df,
        train_size=valid_fraction_of_temp,
        random_state=args.seed,
        stratify=temp_df["label"],
        shuffle=True,
    )

    train_df = train_df.assign(split="train")
    valid_df = valid_df.assign(split="valid")
    test_df = test_df.assign(split="test")
    split_df = (
        pd.concat([train_df, valid_df, test_df], ignore_index=True)
        .sort_values(["split", "label", "subject_id"])
        .reset_index(drop=True)
    )
    split_df = split_df[["subject_id", "label", "official_fold", "split"]]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    summary = build_summary(split_df, args)
    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


def validate_ratios(train_size: float, valid_size: float, test_size: float) -> None:
    total = train_size + valid_size + test_size
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")
    if min(train_size, valid_size, test_size) <= 0:
        raise ValueError("Split ratios must be positive.")


def build_summary(split_df: pd.DataFrame, args: argparse.Namespace) -> dict:
    rows = []
    for split, group in split_df.groupby("split", sort=False):
        rows.append(
            {
                "split": split,
                "n_subjects": int(len(group)),
                "n_hc": int((group["label"] == 0).sum()),
                "n_sz": int((group["label"] == 1).sum()),
                "sz_rate": float((group["label"] == 1).mean()),
                "official_fold_counts": {
                    str(k): int(v) for k, v in group["official_fold"].value_counts().sort_index().items()
                },
            }
        )
    return {
        "name": "ems_subject_split_60_20_20",
        "seed": int(args.seed),
        "ratios": {
            "train": float(args.train_size),
            "valid": float(args.valid_size),
            "test": float(args.test_size),
        },
        "n_subjects": int(split_df["subject_id"].nunique()),
        "label_definition": {"0": "healthy_control", "1": "schizophrenia"},
        "split_summary": rows,
        "protocol_notes": [
            "Subject-level split.",
            "Stratified by label.",
            "Official EMS folds are retained only as metadata.",
            "Use train for fitting, valid for early stopping/threshold/calibration/model selection, and test only once for final reporting.",
        ],
    }


if __name__ == "__main__":
    main()
