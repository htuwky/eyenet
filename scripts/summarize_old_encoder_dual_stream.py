from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


METRICS = ["auc", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1"]
EXPERIMENT_PATTERN = re.compile(r"(?P<name>.+)_seed(?P<seed>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize exploratory encoder + segment-GRU dual-stream results."
    )
    parser.add_argument("--root", default="experiments/ems_encoder_dual_stream")
    parser.add_argument("--output", default="experiments/ems_encoder_dual_stream/old_encoder_dual_stream_summary.csv")
    parser.add_argument("--phase1-summary", default="experiments/encoder_downstream/phase1_encoder_summary.csv")
    parser.add_argument("--split", default="test")
    parser.add_argument("--threshold-name", default="valid_best_balanced_accuracy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    per_seed = collect_dual_stream_rows(args)
    if per_seed.empty:
        raise FileNotFoundError(f"No matching dual-stream metrics found under {args.root}")
    summary = summarize(per_seed)
    comparison = build_comparison(summary, Path(args.phase1_summary))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(output_path.with_name(output_path.stem + "_per_seed.csv"), index=False, encoding="utf-8-sig")
    summary.to_csv(output_path, index=False, encoding="utf-8-sig")
    comparison.to_csv(output_path.with_name(output_path.stem + "_comparison.csv"), index=False, encoding="utf-8-sig")

    print("Old encoder dual-stream per-seed metrics")
    print(per_seed.to_string(index=False))
    print("\nOld encoder dual-stream summary")
    print(summary.to_string(index=False))
    print("\nComparison against phase-1 encoder references")
    print(comparison.to_string(index=False))


def collect_dual_stream_rows(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict] = []
    for metrics_path in sorted(Path(args.root).glob("*/*/metrics.csv")):
        experiment_dir = metrics_path.parent.parent.name
        fusion = metrics_path.parent.name
        match = EXPERIMENT_PATTERN.match(experiment_dir)
        if match is None:
            continue
        metrics = pd.read_csv(metrics_path)
        selected = metrics[
            (metrics["split"] == args.split)
            & (metrics["threshold_name"] == args.threshold_name)
        ]
        if selected.empty:
            continue
        row = selected.iloc[0].to_dict()
        row.update(
            {
                "experiment_group": match.group("name"),
                "experiment": experiment_dir,
                "fusion": fusion,
                "mode": f"old_dual_{fusion}",
                "seed": int(match.group("seed")),
                "metrics_path": str(metrics_path),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["experiment_group", "fusion", "seed"])


def summarize(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (experiment_group, fusion, mode), group in per_seed.groupby(
        ["experiment_group", "fusion", "mode"],
        sort=True,
    ):
        row = {
            "experiment_group": experiment_group,
            "display_name": f"Old encoder dual-stream {fusion}",
            "fusion": fusion,
            "mode": mode,
            "n_seeds": int(group["seed"].nunique()),
            "seeds": ",".join(str(seed) for seed in sorted(group["seed"].unique())),
            "complete_5seed": int(group["seed"].nunique()) == 5,
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
            row[f"{metric}_min"] = float(group[metric].min())
            row[f"{metric}_max"] = float(group[metric].max())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["balanced_accuracy_mean", "auc_mean"],
        ascending=[False, False],
    )


def build_comparison(dual_summary: pd.DataFrame, phase1_summary_path: Path) -> pd.DataFrame:
    rows = []
    for _, row in dual_summary.iterrows():
        rows.append(
            {
                "family": "old_dual_stream",
                "display_name": row["display_name"],
                "mode": row["mode"],
                "auc_mean": row["auc_mean"],
                "auc_std": row["auc_std"],
                "balanced_accuracy_mean": row["balanced_accuracy_mean"],
                "balanced_accuracy_std": row["balanced_accuracy_std"],
                "sensitivity_mean": row["sensitivity_mean"],
                "specificity_mean": row["specificity_mean"],
                "f1_mean": row["f1_mean"],
            }
        )
    if phase1_summary_path.exists():
        phase1 = pd.read_csv(phase1_summary_path)
        refs = phase1[
            (
                (phase1["experiment_group"] == "bigru64_mask045_fusion_ems_only")
                & (phase1["mode"] == "finetune")
            )
            | (
                (phase1["experiment_group"] == "bigru64_mask045_fusion_ems_gazebase_crcns_eye1_onestop")
                & (phase1["mode"] == "finetune")
            )
            | (
                (phase1["experiment_group"] == "bigru64_supervised_only")
                & (phase1["mode"] == "supervised")
            )
        ]
        for _, row in refs.iterrows():
            rows.append(
                {
                    "family": "phase1_encoder_reference",
                    "display_name": row["display_name"],
                    "mode": row["mode"],
                    "auc_mean": row["auc_mean"],
                    "auc_std": row["auc_std"],
                    "balanced_accuracy_mean": row["balanced_accuracy_mean"],
                    "balanced_accuracy_std": row["balanced_accuracy_std"],
                    "sensitivity_mean": row["sensitivity_mean"],
                    "specificity_mean": row["specificity_mean"],
                    "f1_mean": row["f1_mean"],
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["balanced_accuracy_mean", "auc_mean"],
        ascending=[False, False],
    )


if __name__ == "__main__":
    main()
