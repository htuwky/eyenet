from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

EXPERIMENTS = [
    ("ml_logistic_regression", "baseline", "logistic_regression"),
    ("ml_svm_rbf", "baseline", "svm_rbf"),
    ("ml_random_forest", "baseline", "random_forest"),
    ("ml_hist_gradient_boosting", "baseline", "hist_gradient_boosting"),
    ("ml_mlp", "baseline", "mlp"),
    ("macro_behavior_stream", "macro_behavior_sequence", "segment_sequence_bigru_attention"),
    ("event_temporal_stream", "event_temporal_sequence", "event_temporal_sequence_bigru_attention"),
    ("dual_stream_concat", "dual_stream_concat", "dual_stream_concat_attention"),
    ("dual_stream_gated", "dual_stream_gated", "dual_stream_gated_attention"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize EMS fixed-split experiment metrics for paper tables.")
    parser.add_argument("--experiment-root", default="experiments/ems_fixed_split")
    parser.add_argument("--output-dir", default="experiments/ems_fixed_split/summary")
    parser.add_argument("--primary-threshold", default="valid_best_balanced_accuracy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.experiment_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[pd.DataFrame] = []
    missing: list[dict] = []
    for display_name, experiment_dir, model_name in EXPERIMENTS:
        metrics_path = root / experiment_dir / "metrics.csv"
        if not metrics_path.exists():
            missing.append({"display_name": display_name, "metrics_path": str(metrics_path)})
            continue
        metrics = pd.read_csv(metrics_path)
        rows = metrics[(metrics["split"] == "test") & (metrics["model"] == model_name)].copy()
        if rows.empty:
            missing.append({"display_name": display_name, "metrics_path": str(metrics_path)})
            continue
        rows.insert(0, "display_name", display_name)
        rows.insert(1, "experiment_dir", experiment_dir)
        all_rows.append(rows)

    if not all_rows:
        raise ValueError("No fixed-split test metric rows were found.")

    all_test_metrics = pd.concat(all_rows, ignore_index=True)
    all_test_metrics = all_test_metrics.sort_values(["display_name", "threshold_name"]).reset_index(drop=True)
    primary = all_test_metrics[all_test_metrics["threshold_name"] == args.primary_threshold].copy()
    primary = primary.sort_values(["auc", "balanced_accuracy", "sensitivity"], ascending=False).reset_index(drop=True)

    metric_cols = ["auc", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1"]
    publication = primary[["display_name", "model", "threshold"] + metric_cols + ["tp", "tn", "fp", "fn"]].copy()
    for col in metric_cols:
        publication[col] = publication[col].map(lambda value: f"{value:.3f}")
    publication["confusion_matrix"] = publication.apply(
        lambda row: f"TP={int(row['tp'])}, TN={int(row['tn'])}, FP={int(row['fp'])}, FN={int(row['fn'])}",
        axis=1,
    )
    publication = publication.drop(columns=["tp", "tn", "fp", "fn"])

    all_test_metrics.to_csv(output_dir / "fixed_split_all_test_metrics.csv", index=False, encoding="utf-8-sig")
    primary.to_csv(output_dir / "fixed_split_primary_test_metrics.csv", index=False, encoding="utf-8-sig")
    publication.to_csv(output_dir / "fixed_split_publication_table.csv", index=False, encoding="utf-8-sig")
    if missing:
        pd.DataFrame(missing).to_csv(output_dir / "missing_experiments.csv", index=False, encoding="utf-8-sig")

    print("Primary fixed-split test metrics")
    print(primary[["display_name", "threshold", *metric_cols, "tp", "tn", "fp", "fn"]].to_string(index=False))
    print()
    print("Outputs")
    print(f"all_test_metrics: {output_dir / 'fixed_split_all_test_metrics.csv'}")
    print(f"primary_metrics: {output_dir / 'fixed_split_primary_test_metrics.csv'}")
    print(f"publication_table: {output_dir / 'fixed_split_publication_table.csv'}")
    if missing:
        print(f"missing_experiments: {output_dir / 'missing_experiments.csv'}")


if __name__ == "__main__":
    main()
