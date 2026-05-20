from __future__ import annotations

import argparse

import pandas as pd

from eyenet.training.baseline import (
    run_official_fold_baseline,
    save_baseline_outputs,
    summarize_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EMS segment-aggregation baseline models.")
    parser.add_argument("--features", default="data/processed/EMS/ems_subject_features_segment_agg_no_pupil.csv")
    parser.add_argument("--output-dir", default="experiments/ems_segment_baseline")
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = pd.read_csv(args.features, dtype={"subject_id": str})
    fold_metrics, predictions = run_official_fold_baseline(features, random_seed=args.random_seed)
    summary = summarize_metrics(fold_metrics)
    save_baseline_outputs(args.output_dir, fold_metrics, summary, predictions)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
