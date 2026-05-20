from __future__ import annotations

import argparse

from eyenet.training.ensemble import build_ensemble_predictions, load_seed_predictions, save_ensemble_predictions
from eyenet.training.thresholds import analyze_thresholds, choose_thresholds, save_threshold_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Average multi-seed predictions and analyze thresholds.")
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_predictions = load_seed_predictions(args.experiment_dir)
    ensemble = build_ensemble_predictions(seed_predictions, threshold=args.threshold)
    threshold_metrics = analyze_thresholds(ensemble)
    selected_thresholds = choose_thresholds(threshold_metrics)
    save_ensemble_predictions(args.experiment_dir, seed_predictions, ensemble)
    save_threshold_outputs(args.experiment_dir, threshold_metrics, selected_thresholds)

    print("Ensemble selected thresholds")
    print(selected_thresholds.to_string(index=False))


if __name__ == "__main__":
    main()
