from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

METRICS = ["auc", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1"]
SEED_PATTERN = re.compile(r"_seed(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize test metrics across validation-selected threshold strategies.")
    parser.add_argument("--root", default="experiments/encoder_downstream/architecture_ablation")
    parser.add_argument("--contains", default="")
    parser.add_argument("--exclude", default="")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default="experiments/encoder_downstream/architecture_ablation/threshold_strategy_summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    per_seed = collect_rows(Path(args.root), args)
    if per_seed.empty:
        raise FileNotFoundError(f"No matching metrics.csv files found under {args.root}")

    per_seed = per_seed.sort_values(["experiment_group", "mode", "threshold_name", "seed", "experiment"])
    summary = summarize(per_seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(output_path.with_name(output_path.stem + "_per_seed.csv"), index=False, encoding="utf-8-sig")
    summary.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("Per-seed threshold metrics")
    print(per_seed.to_string(index=False))
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
        selected = metrics[metrics["split"] == args.split].copy()
        if selected.empty:
            continue
        selected["experiment"] = experiment
        selected["experiment_group"] = strip_seed_suffix(experiment)
        selected["mode"] = mode
        selected["seed"] = seed
        rows.extend(selected.to_dict(orient="records"))
    return pd.DataFrame(rows)


def parse_seed(name: str) -> int | None:
    match = SEED_PATTERN.search(name)
    if match is None:
        return None
    return int(match.group(1))


def strip_seed_suffix(name: str) -> str:
    return SEED_PATTERN.sub("", name)


def summarize(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = ["experiment_group", "mode", "threshold_name"]
    for (experiment_group, mode, threshold_name), group in per_seed.groupby(group_columns, sort=True):
        row = {
            "experiment_group": experiment_group,
            "mode": mode,
            "threshold_name": threshold_name,
            "n_seeds": int(group["seed"].nunique()),
            "seeds": ",".join(str(seed) for seed in sorted(group["seed"].unique())),
            "threshold_mean": float(group["threshold"].mean()),
            "threshold_std": float(group["threshold"].std(ddof=1)) if len(group) > 1 else 0.0,
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
            row[f"{metric}_min"] = float(group[metric].min())
            row[f"{metric}_max"] = float(group[metric].max())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["experiment_group", "mode", "sensitivity_mean", "balanced_accuracy_mean", "auc_mean"],
        ascending=[True, True, False, False, False],
    )


if __name__ == "__main__":
    main()
