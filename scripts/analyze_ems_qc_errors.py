from __future__ import annotations

import argparse

import pandas as pd

from eyenet.training.qc_analysis import (
    add_qc_flags,
    build_error_group_feature_differences,
    build_fp_fn_summary,
    build_misclassified_profiles,
    build_qc_flag_summary,
    build_subject_prediction_summary,
    save_qc_error_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile EMS baseline errors and QC flags.")
    parser.add_argument("--features", default="data/processed/EMS/ems_subject_features_no_pupil.csv")
    parser.add_argument("--experiment-dir", default="experiments/ems_baseline")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = pd.read_csv(args.features, dtype={"subject_id": str})
    predictions = pd.read_csv(f"{args.experiment_dir}/predictions.csv", dtype={"subject_id": str})

    prediction_summary = build_subject_prediction_summary(predictions)
    profiles = build_misclassified_profiles(features, prediction_summary)
    profiles = add_qc_flags(profiles)
    feature_differences = build_error_group_feature_differences(profiles)
    qc_flag_summary = build_qc_flag_summary(profiles)
    fp_fn_summary = build_fp_fn_summary(profiles)
    save_qc_error_outputs(args.experiment_dir, profiles, feature_differences, qc_flag_summary, fp_fn_summary)

    stable_errors = profiles[profiles["stable_error"]].copy()
    print("Stable misclassified subjects")
    print(stable_errors[["subject_id", "fold", "label", "error_group", "mean_probability", "qc_flag_count"]].to_string(index=False))
    print()
    print("QC flag summary by error group")
    print(qc_flag_summary.to_string(index=False))
    print()
    print("FP/FN summary by fold")
    print(fp_fn_summary.to_string(index=False))


if __name__ == "__main__":
    main()
