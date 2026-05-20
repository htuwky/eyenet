from __future__ import annotations

import argparse
from pathlib import Path

from eyenet.training.event_temporal_diagnostics import (
    build_default_metric_recheck,
    build_fold_probability_summary,
    build_fold_threshold_summary,
    build_misclassified_subjects,
    build_pooled_auc_diagnostics,
    build_training_curve_summary,
    load_event_temporal_experiment,
    save_event_temporal_diagnostics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose Event-Temporal Stream fold instability and calibration.")
    parser.add_argument("--experiment-dir", default="experiments/ems_event_temporal_sequence_pos1_0")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir)
    output_dir = Path(args.output_dir) if args.output_dir else experiment_dir / "diagnostics"

    predictions, fold_metrics, training_log = load_event_temporal_experiment(experiment_dir)
    fold_probability_summary = build_fold_probability_summary(predictions)
    fold_threshold_summary = build_fold_threshold_summary(predictions)
    training_curve_summary = build_training_curve_summary(training_log, fold_metrics)
    pooled_auc_diagnostics = build_pooled_auc_diagnostics(predictions)
    misclassified_subjects = build_misclassified_subjects(predictions, threshold=args.threshold)
    default_metric_recheck = build_default_metric_recheck(predictions)

    save_event_temporal_diagnostics(
        output_dir=output_dir,
        fold_probability_summary=fold_probability_summary,
        fold_threshold_summary=fold_threshold_summary,
        training_curve_summary=training_curve_summary,
        pooled_auc_diagnostics=pooled_auc_diagnostics,
        misclassified_subjects=misclassified_subjects,
        default_metric_recheck=default_metric_recheck,
    )

    print("Event-Temporal diagnostics outputs")
    print(f"output_dir: {output_dir}")
    print("\nFold probability summary")
    print(fold_probability_summary.to_string(index=False))
    print("\nPooled AUC diagnostics")
    print(pooled_auc_diagnostics.to_string(index=False))
    print("\nTraining curve summary")
    print(training_curve_summary.to_string(index=False))
    print("\nDefault threshold metric recheck")
    print(default_metric_recheck.to_string(index=False))


if __name__ == "__main__":
    main()
