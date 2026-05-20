from __future__ import annotations

import argparse
from pathlib import Path

from eyenet.training.model_comparison import (
    align_prediction_tables,
    bootstrap_model_comparison,
    build_model_metric_table,
    load_named_predictions,
    make_publication_table,
    save_model_comparison_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap comparison between deep ensemble and ML baseline.")
    parser.add_argument("--baseline-predictions", default="experiments/ems_segment_baseline/predictions.csv")
    parser.add_argument("--baseline-model", default="svm_rbf")
    parser.add_argument("--baseline-threshold", type=float, default=0.5)
    parser.add_argument("--deep-predictions", default="experiments/ems_segment_sequence_pos1_5/ensemble_predictions.csv")
    parser.add_argument("--deep-threshold", type=float, default=0.45)
    parser.add_argument("--output-dir", default="experiments/ems_model_comparison")
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_name = f"baseline_{args.baseline_model}"
    deep_name = "deep_segment_sequence_ensemble"
    baseline = load_named_predictions(
        args.baseline_predictions,
        model_name=baseline_name,
        threshold=args.baseline_threshold,
        model_filter=args.baseline_model,
    )
    deep = load_named_predictions(
        args.deep_predictions,
        model_name=deep_name,
        threshold=args.deep_threshold,
    )
    aligned = align_prediction_tables(
        reference=baseline,
        candidate=deep,
        reference_name=baseline_name,
        candidate_name=deep_name,
    )
    point_metrics = build_model_metric_table(aligned, model_names=[baseline_name, deep_name])
    metric_ci, difference_ci = bootstrap_model_comparison(
        aligned,
        reference_name=baseline_name,
        candidate_name=deep_name,
        n_bootstrap=args.n_bootstrap,
        random_seed=args.seed,
    )
    publication_table = make_publication_table(point_metrics, metric_ci)
    save_model_comparison_outputs(
        output_dir=Path(args.output_dir),
        aligned_predictions=aligned,
        point_metrics=point_metrics,
        bootstrap_metric_ci=metric_ci,
        bootstrap_difference_ci=difference_ci,
        publication_table=publication_table,
    )

    print("Model comparison outputs")
    print(f"output_dir: {args.output_dir}")
    print(f"n_subjects: {aligned['subject_id'].nunique()}")
    print("\nPoint metrics")
    print(point_metrics.to_string(index=False))
    print("\nPublication metric table")
    print(publication_table.to_string(index=False))
    print("\nBootstrap differences: deep minus baseline")
    print(difference_ci.to_string(index=False))


if __name__ == "__main__":
    main()
