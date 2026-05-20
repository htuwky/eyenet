from __future__ import annotations

import argparse

import pandas as pd

from eyenet.training.thresholds import analyze_thresholds, choose_thresholds, save_threshold_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze classification thresholds for screening use.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = pd.read_csv(args.predictions, dtype={"subject_id": str})
    threshold_metrics = analyze_thresholds(predictions)
    selected_thresholds = choose_thresholds(threshold_metrics)
    save_threshold_outputs(args.output_dir, threshold_metrics, selected_thresholds)
    print("Selected thresholds")
    print(selected_thresholds.to_string(index=False))


if __name__ == "__main__":
    main()
