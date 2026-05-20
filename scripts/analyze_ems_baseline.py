from __future__ import annotations

import argparse

import pandas as pd

from eyenet.training.analysis import (
    build_error_analysis,
    build_fold_error_summary,
    compute_permutation_importance_by_fold,
    make_paper_table,
    save_analysis_outputs,
    summarize_importance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze EMS baseline model outputs.")
    parser.add_argument("--features", default="data/processed/EMS/ems_subject_features_no_pupil.csv")
    parser.add_argument("--experiment-dir", default="experiments/ems_baseline")
    parser.add_argument("--model-name", default="random_forest")
    parser.add_argument("--n-repeats", type=int, default=30)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir
    features = pd.read_csv(args.features, dtype={"subject_id": str})
    fold_metrics = pd.read_csv(f"{experiment_dir}/fold_metrics.csv")
    predictions = pd.read_csv(f"{experiment_dir}/predictions.csv", dtype={"subject_id": str})

    baseline_table = make_paper_table(fold_metrics)
    error_analysis = build_error_analysis(predictions)
    fold_error_summary = build_fold_error_summary(predictions)
    permutation_importance_df = compute_permutation_importance_by_fold(
        features,
        model_name=args.model_name,
        random_seed=args.random_seed,
        n_repeats=args.n_repeats,
    )
    top_features = summarize_importance(permutation_importance_df).head(args.top_k)
    save_analysis_outputs(
        experiment_dir,
        baseline_table,
        error_analysis,
        fold_error_summary,
        permutation_importance_df,
        top_features,
    )

    print("Baseline table")
    print(baseline_table.to_string(index=False))
    print()
    print(f"Top {args.top_k} features by permutation importance ({args.model_name})")
    print(top_features.to_string(index=False))
    print()
    print("Most frequently misclassified subjects")
    print(error_analysis.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
