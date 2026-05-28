from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

METRICS = ["auc", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize multi-seed supervised encoder downstream metrics.")
    parser.add_argument("--root", default="experiments/encoder_downstream/multiseed")
    parser.add_argument("--output", default="experiments/encoder_downstream/multiseed/summary.csv")
    parser.add_argument("--threshold-name", default="valid_best_balanced_accuracy")
    parser.add_argument("--split", default="test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect_rows(Path(args.root), args.split, args.threshold_name)
    if not rows:
        raise FileNotFoundError(f"No metrics.csv files found under {args.root}")

    seed_metrics = pd.DataFrame(rows).sort_values(["experiment", "seed"])
    summary = summarize(seed_metrics)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seed_metrics.to_csv(output_path.with_name(output_path.stem + "_per_seed.csv"), index=False, encoding="utf-8-sig")
    summary.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("Per-seed metrics")
    print(seed_metrics.to_string(index=False))
    print("\nSummary")
    print(summary.to_string(index=False))


def collect_rows(root: Path, split: str, threshold_name: str) -> list[dict]:
    rows: list[dict] = []
    for metrics_path in sorted(root.glob("*/*/metrics.csv")):
        experiment = metrics_path.parent.parent.name
        seed_name = metrics_path.parent.name
        if not seed_name.startswith("seed_"):
            continue
        metrics = pd.read_csv(metrics_path)
        selected = metrics[(metrics["split"] == split) & (metrics["threshold_name"] == threshold_name)]
        if selected.empty:
            continue
        row = selected.iloc[0].to_dict()
        row["experiment"] = experiment
        row["seed"] = int(seed_name.replace("seed_", ""))
        rows.append(row)
    return rows


def summarize(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for experiment, group in seed_metrics.groupby("experiment", sort=True):
        row = {"experiment": experiment, "n_seeds": int(group["seed"].nunique())}
        for metric in METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
            row[f"{metric}_min"] = float(group[metric].min())
            row[f"{metric}_max"] = float(group[metric].max())
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
