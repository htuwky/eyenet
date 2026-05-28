from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from eyenet.training.baseline import compute_metrics
from eyenet.training.ensemble import build_ensemble_predictions, load_seed_predictions, save_ensemble_predictions
from eyenet.training.thresholds import analyze_thresholds, choose_thresholds, save_threshold_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Average multi-seed predictions and analyze thresholds. "
            "Current fixed-split runs are grouped by split + subject_id + label; "
            "fold is optional legacy metadata."
        )
    )
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--contains", default=None, help="Match existing experiment directories such as name_seed0.")
    parser.add_argument("--mode", default="finetune", help="Subdirectory containing predictions.csv when using --contains.")
    parser.add_argument("--split", default="test", help="Prediction split to analyze; use 'all' to keep all rows.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--require-complete-seeds",
        action="store_true",
        help="Require every ensemble row in the selected split to have predictions from all discovered seeds.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def fixed_threshold_row(predictions, threshold: float) -> dict:
    y_true = predictions["label"].to_numpy(dtype=int)
    y_prob = predictions["probability"].to_numpy(dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    metrics = compute_metrics(y_true, y_pred, y_prob)
    return {"criterion": f"fixed_{threshold:.2f}", "threshold": threshold, **metrics}


def main() -> None:
    args = parse_args()
    mode = args.mode if args.contains else None
    seed_predictions = load_seed_predictions(args.experiment_dir, contains=args.contains, mode=mode)
    if args.split != "all" and "split" in seed_predictions.columns:
        seed_predictions = seed_predictions[seed_predictions["split"] == args.split].copy()
        if seed_predictions.empty:
            raise ValueError(f"No predictions found for split={args.split!r}")

    n_input_seeds = seed_predictions["seed"].nunique()
    ensemble = build_ensemble_predictions(seed_predictions, threshold=args.threshold)
    seed_coverage = (
        ensemble["n_seeds"]
        .value_counts()
        .rename_axis("n_seeds")
        .reset_index(name="n_rows")
        .sort_values("n_seeds")
    )
    if args.require_complete_seeds:
        ensemble = ensemble[ensemble["n_seeds"] == n_input_seeds].copy()
        if ensemble.empty:
            raise ValueError(f"No rows have predictions from all {n_input_seeds} seeds.")

    threshold_metrics = analyze_thresholds(ensemble)
    selected_thresholds = choose_thresholds(threshold_metrics)
    selected_thresholds = pd.concat(
        [selected_thresholds, pd.DataFrame([fixed_threshold_row(ensemble, args.threshold)])],
        ignore_index=True,
    )

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.experiment_dir)
    if args.contains and args.output_dir is None:
        output_dir = output_dir / f"{args.contains}_ensemble"
    save_ensemble_predictions(output_dir, seed_predictions, ensemble)
    save_threshold_outputs(output_dir, threshold_metrics, selected_thresholds)
    seed_coverage.to_csv(output_dir / "seed_coverage.csv", index=False, encoding="utf-8-sig")

    if seed_coverage["n_seeds"].max() < n_input_seeds:
        print(f"WARNING: no ensemble rows include all {n_input_seeds} seeds.")
    elif (ensemble["n_seeds"] < n_input_seeds).any():
        print(f"WARNING: some ensemble rows include fewer than {n_input_seeds} seeds.")
    print("Seed coverage")
    print(seed_coverage.to_string(index=False))
    print("Ensemble selected thresholds")
    print(selected_thresholds.to_string(index=False))


if __name__ == "__main__":
    main()
