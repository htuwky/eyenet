from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


METRIC_COLUMNS = ["auc", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize EMS subject-summary baseline multiseed results.")
    parser.add_argument("--root", default="experiments/ems_subject_summary_baseline")
    parser.add_argument("--output", default="experiments/ems_subject_summary_baseline/summary.csv")
    return parser.parse_args()


def seed_from_path(path: Path) -> int | None:
    for part in path.parts:
        match = re.fullmatch(r"seed(\d+)", part)
        if match:
            return int(match.group(1))
    return None


def main() -> None:
    args = parse_args()
    rows = []
    for metrics_path in sorted(Path(args.root).glob("seed*/metrics.csv")):
        metrics = pd.read_csv(metrics_path)
        selected = metrics[
            (metrics["split"] == "test")
            & (metrics["threshold_name"] == "valid_best_balanced_accuracy")
        ].copy()
        selected["seed"] = seed_from_path(metrics_path)  # type: ignore[assignment]
        selected["metrics_path"] = str(metrics_path)
        rows.append(selected)

    if not rows:
        raise FileNotFoundError(f"No seed*/metrics.csv files found under {args.root}")

    per_seed = pd.concat(rows, ignore_index=True)
    summary_rows = []
    for model, group in per_seed.groupby("model"):
        row = {
            "model": model,
            "n_seeds": int(group["seed"].nunique()),
            "seeds": ",".join(str(int(seed)) for seed in sorted(group["seed"].dropna().unique())),
        }
        for metric in METRIC_COLUMNS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std())
            row[f"{metric}_min"] = float(group[metric].min())
            row[f"{metric}_max"] = float(group[metric].max())
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values(
        ["balanced_accuracy_mean", "auc_mean"], ascending=False
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False, encoding="utf-8-sig")
    per_seed.to_csv(output.with_name(output.stem + "_per_seed.csv"), index=False, encoding="utf-8-sig")

    print("EMS subject-summary baseline per-seed metrics")
    print(per_seed.sort_values(["model", "seed"]).to_string(index=False))
    print("\nSummary")
    print(summary.to_string(index=False))
    print(f"\nwrote: {output}")


if __name__ == "__main__":
    main()
