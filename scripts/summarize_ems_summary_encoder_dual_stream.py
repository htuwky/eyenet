from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

METRICS = ["auc", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1"]
EXPERIMENT_PATTERN = re.compile(r"(?P<name>.+)_seed(?P<seed>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize strict-summary + encoder dual-stream results.")
    parser.add_argument("--root", default="experiments/ems_summary_encoder_dual_stream")
    parser.add_argument("--output", default="experiments/ems_summary_encoder_dual_stream/summary.csv")
    parser.add_argument("--phase1-summary", default="experiments/encoder_downstream/phase1_encoder_summary.csv")
    parser.add_argument("--summary-only", default="experiments/ems_subject_summary_baseline_strict/summary.csv")
    parser.add_argument("--split", default="test")
    parser.add_argument("--threshold-name", default="valid_best_balanced_accuracy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    per_seed = collect_rows(args)
    if per_seed.empty:
        raise FileNotFoundError(f"No matching summary+encoder dual-stream metrics found under {args.root}")
    summary = summarize(per_seed)
    comparison = build_comparison(summary, Path(args.phase1_summary), Path(args.summary_only))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(output_path.with_name(output_path.stem + "_per_seed.csv"), index=False, encoding="utf-8-sig")
    summary.to_csv(output_path, index=False, encoding="utf-8-sig")
    comparison.to_csv(output_path.with_name(output_path.stem + "_comparison.csv"), index=False, encoding="utf-8-sig")

    print("Summary+encoder dual-stream per-seed metrics")
    print(per_seed.to_string(index=False))
    print("\nSummary+encoder dual-stream summary")
    print(summary.to_string(index=False))
    print("\nComparison against encoder and strict summary-only references")
    print(comparison.to_string(index=False))


def collect_rows(args: argparse.Namespace) -> pd.DataFrame:
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
                "mode": f"summary_encoder_dual_{fusion}",
                "seed": int(match.group("seed")),
                "metrics_path": str(metrics_path),
            }
        )
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["experiment_group", "fusion", "seed"])


def summarize(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (experiment_group, fusion, mode), group in per_seed.groupby(
        ["experiment_group", "fusion", "mode"],
        sort=True,
    ):
        row = {
            "experiment_group": experiment_group,
            "display_name": f"Strict summary + encoder dual-stream {fusion}",
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


def build_comparison(dual_summary: pd.DataFrame, phase1_summary_path: Path, summary_only_path: Path) -> pd.DataFrame:
    rows = []
    for _, row in dual_summary.iterrows():
        rows.append(reference_row("new_dual_stream", row["display_name"], row["mode"], row))

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
            rows.append(reference_row("phase1_encoder_reference", row["display_name"], row["mode"], row))

    if summary_only_path.exists():
        summary_only = pd.read_csv(summary_only_path)
        refs = summary_only[summary_only["model"].isin(["logistic_regression", "svm_rbf"])]
        for _, row in refs.iterrows():
            rows.append(reference_row("strict_summary_only", f"Strict summary-only {row['model']}", row["model"], row))

    return pd.DataFrame(rows).sort_values(
        ["balanced_accuracy_mean", "auc_mean"],
        ascending=[False, False],
    )


def reference_row(family: str, display_name: str, mode: str, row: pd.Series) -> dict:
    return {
        "family": family,
        "display_name": display_name,
        "mode": mode,
        "auc_mean": row["auc_mean"],
        "auc_std": row["auc_std"],
        "balanced_accuracy_mean": row["balanced_accuracy_mean"],
        "balanced_accuracy_std": row["balanced_accuracy_std"],
        "sensitivity_mean": row["sensitivity_mean"],
        "specificity_mean": row["specificity_mean"],
        "f1_mean": row["f1_mean"],
    }


if __name__ == "__main__":
    main()
