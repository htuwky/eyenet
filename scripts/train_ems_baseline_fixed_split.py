from __future__ import annotations

import argparse

import pandas as pd

from eyenet.training.fixed_split_baseline import (
    run_fixed_split_baseline,
    save_fixed_split_baseline_outputs,
)
from eyenet.utils.config import attach_arg_defaults, cfg_arg, load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EMS baseline models with fixed train/valid/test split.")
    parser.add_argument("--config", default="configs/experiments/ems_baseline.yaml")
    parser.add_argument("--features", default="data/processed/EMS/ems_subject_features_segment_agg_no_pupil.csv")
    parser.add_argument("--split", default="data/splits/EMS/ems_subject_split_60_20_20_seed42.csv")
    parser.add_argument("--output-dir", default="experiments/ems_fixed_split/baseline")
    parser.add_argument("--random-seed", type=int, default=42)
    return attach_arg_defaults(parser, parser.parse_args())


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    args.features = cfg_arg(args, config, "features", "data.features")
    args.split = cfg_arg(args, config, "split", "data.split")
    args.output_dir = cfg_arg(args, config, "output_dir", "experiment.output_dir")
    args.random_seed = cfg_arg(args, config, "random_seed", "experiment.random_seed")
    features = pd.read_csv(args.features, dtype={"subject_id": str})
    metrics, predictions, valid_threshold_metrics, selected_thresholds, split_subjects = run_fixed_split_baseline(
        features=features,
        split_path=args.split,
        random_seed=args.random_seed,
    )
    save_fixed_split_baseline_outputs(
        output_dir=args.output_dir,
        metrics=metrics,
        predictions=predictions,
        valid_threshold_metrics=valid_threshold_metrics,
        selected_thresholds=selected_thresholds,
        split_subjects=split_subjects,
    )

    test_rows = metrics[(metrics["split"] == "test") & (metrics["threshold_name"] == "valid_best_balanced_accuracy")]
    print("Fixed-split test metrics using validation-selected best balanced-accuracy threshold")
    print(test_rows.sort_values("auc", ascending=False).to_string(index=False))
    print("\nValidation-selected thresholds")
    print(selected_thresholds.to_string(index=False))


if __name__ == "__main__":
    main()
