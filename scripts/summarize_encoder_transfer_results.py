from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_RUNS = {
    "from_scratch_supervised": "experiments/encoder_smoke/ems_clipped_qc_no_position",
    "masked_pretrained_finetune": "experiments/encoder_smoke/ems_clipped_qc_no_position_masked_pretrained_finetune",
    "masked_pretrained_frozen": "experiments/encoder_smoke/ems_clipped_qc_no_position_masked_pretrained_frozen",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize supervised encoder transfer experiments.")
    parser.add_argument("--output", default="experiments/encoder_smoke/encoder_transfer_summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for experiment_name, experiment_dir in DEFAULT_RUNS.items():
        metrics_path = Path(experiment_dir) / "metrics.csv"
        config_path = Path(experiment_dir) / "config.json"
        if not metrics_path.exists():
            continue
        metrics = pd.read_csv(metrics_path)
        test_rows = metrics[
            (metrics["split"] == "test")
            & (metrics["threshold_name"] == "valid_best_balanced_accuracy")
        ].copy()
        if test_rows.empty:
            continue
        row = test_rows.iloc[0].to_dict()
        row = {
            "experiment": experiment_name,
            "experiment_dir": experiment_dir,
            "config_path": str(config_path) if config_path.exists() else "",
            **row,
        }
        rows.append(row)

    summary = pd.DataFrame(rows)
    ordered_columns = [
        "experiment",
        "experiment_dir",
        "auc",
        "accuracy",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1",
        "threshold",
        "tp",
        "tn",
        "fp",
        "fn",
        "best_epoch",
        "stopped_epoch",
        "best_valid_auc",
        "loss",
        "config_path",
    ]
    summary = summary[[column for column in ordered_columns if column in summary.columns]]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False, encoding="utf-8-sig")
    print("Encoder transfer summary")
    print(summary.to_string(index=False))
    print(f"\nOutput: {output_path}")


if __name__ == "__main__":
    main()
