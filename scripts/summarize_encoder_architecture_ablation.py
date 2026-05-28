from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

METRICS = ["auc", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1"]
SEED_PATTERN = re.compile(r"_seed(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize encoder architecture ablation metrics.")
    parser.add_argument("--root", default="experiments/encoder_downstream/architecture_ablation")
    parser.add_argument("--output", default="experiments/encoder_downstream/architecture_ablation/summary.csv")
    parser.add_argument("--contains", default="aligned", help="Only include experiment names containing this text. Use empty string for all.")
    parser.add_argument("--exclude", default="", help="Comma-separated experiment-name substrings to exclude.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--threshold-name", default="valid_best_balanced_accuracy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_metrics = collect_rows(Path(args.root), args)
    if seed_metrics.empty:
        raise FileNotFoundError(f"No matching metrics.csv files found under {args.root}")

    seed_metrics = seed_metrics.sort_values(["experiment_group", "mode", "seed", "experiment"])
    summary = summarize(seed_metrics)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seed_metrics.to_csv(output_path.with_name(output_path.stem + "_per_seed.csv"), index=False, encoding="utf-8-sig")
    summary.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("Per-seed metrics")
    print(seed_metrics.to_string(index=False))
    print("\nSummary")
    print(summary.to_string(index=False))


def collect_rows(root: Path, args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict] = []
    excludes = [item.strip() for item in args.exclude.split(",") if item.strip()]
    for metrics_path in sorted(root.glob("*/*/metrics.csv")):
        experiment = metrics_path.parent.parent.name
        mode = metrics_path.parent.name
        if args.contains and args.contains not in experiment:
            continue
        if any(excluded in experiment for excluded in excludes):
            continue
        seed = parse_seed(experiment)
        if seed is None:
            continue
        metrics = pd.read_csv(metrics_path)
        selected = metrics[
            (metrics["split"] == args.split)
            & (metrics["threshold_name"] == args.threshold_name)
        ]
        if selected.empty:
            continue
        row = selected.iloc[0].to_dict()
        row["experiment"] = experiment
        row["experiment_group"] = strip_seed_suffix(experiment)
        row["mode"] = mode
        row["seed"] = seed
        rows.append(row)
    return pd.DataFrame(rows)


def parse_seed(name: str) -> int | None:
    match = SEED_PATTERN.search(name)
    if match is None:
        return None
    return int(match.group(1))


def strip_seed_suffix(name: str) -> str:
    return SEED_PATTERN.sub("", name)


def summarize(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (experiment_group, mode), group in seed_metrics.groupby(["experiment_group", "mode"], sort=True):
        row = {
            "experiment_group": experiment_group,
            "mode": mode,
            "n_seeds": int(group["seed"].nunique()),
            "seeds": ",".join(str(seed) for seed in sorted(group["seed"].unique())),
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
            row[f"{metric}_min"] = float(group[metric].min())
            row[f"{metric}_max"] = float(group[metric].max())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["mode", "balanced_accuracy_mean", "auc_mean"],
        ascending=[True, False, False],
    )


if __name__ == "__main__":
    main()
