from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from eyenet.training.statistics import (
    DEFAULT_FEATURES,
    build_subject_level_feature_table,
    compare_groups,
    save_statistical_validation_outputs,
    summarize_error_groups,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Statistical validation for high-attention eye-movement features.")
    parser.add_argument(
        "--attention-segment-features",
        default="experiments/ems_segment_sequence_pos1_5/attention_analysis/attention_segment_features.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/ems_segment_sequence_pos1_5/attention_analysis/statistics",
    )
    parser.add_argument("--attention-group", default="top", choices=["top", "rest", "all"])
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--features", nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = args.features or DEFAULT_FEATURES
    attention_segments = pd.read_csv(args.attention_segment_features, dtype={"subject_id": str})
    subject_feature_table = build_subject_level_feature_table(
        attention_segments,
        features=features,
        attention_group=args.attention_group,
    )
    group_comparison = compare_groups(
        subject_feature_table,
        features=features,
        n_bootstrap=args.n_bootstrap,
        random_seed=args.seed,
    )
    error_group_summary = summarize_error_groups(subject_feature_table, features=features)
    save_statistical_validation_outputs(
        output_dir=Path(args.output_dir),
        subject_feature_table=subject_feature_table,
        group_comparison=group_comparison,
        error_group_summary=error_group_summary,
    )

    print("Statistical validation outputs")
    print(f"output_dir: {args.output_dir}")
    print(f"attention_group: {args.attention_group}")
    print(f"n_subjects: {subject_feature_table['subject_id'].nunique()}")
    print("\nHC vs SZ tests on subject-level high-attention features")
    print(group_comparison.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
