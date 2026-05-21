from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize EMS original/strict/clipped QC variant results.")
    parser.add_argument("--output", default="experiments/ems_filtered/qc_variant_summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = [
        ("original", "baseline", "with_position", "experiments/ems_fixed_split/baseline/metrics.csv"),
        ("original", "dual_stream_concat", "with_position", "experiments/ems_fixed_split/dual_stream_concat/metrics.csv"),
        (
            "original",
            "dual_stream_concat",
            "no_position",
            "experiments/ems_filtered/original/dual_stream_concat_no_position/metrics.csv",
        ),
        ("strict_qc", "baseline", "with_position", "experiments/ems_filtered/strict_qc/baseline/metrics.csv"),
        ("strict_qc", "dual_stream_concat", "with_position", "experiments/ems_filtered/strict_qc/dual_stream_concat/metrics.csv"),
        (
            "strict_qc",
            "dual_stream_concat",
            "no_position",
            "experiments/ems_filtered/strict_qc/dual_stream_concat_no_position/metrics.csv",
        ),
        ("clipped_qc", "baseline", "with_position", "experiments/ems_filtered/clipped_qc/baseline/metrics.csv"),
        ("clipped_qc", "dual_stream_concat", "with_position", "experiments/ems_filtered/clipped_qc/dual_stream_concat/metrics.csv"),
        (
            "clipped_qc",
            "dual_stream_concat",
            "no_position",
            "experiments/ems_filtered/clipped_qc/dual_stream_concat_no_position/metrics.csv",
        ),
    ]
    rows: list[pd.DataFrame] = []
    for variant, family, position_setting, path in specs:
        metrics_path = Path(path)
        if not metrics_path.exists():
            continue
        metrics = pd.read_csv(metrics_path)
        selected = metrics[
            (metrics["split"] == "test")
            & (metrics["threshold_name"] == "valid_best_balanced_accuracy")
        ].copy()
        selected.insert(0, "variant", variant)
        selected.insert(1, "model_family", family)
        selected.insert(2, "position_setting", position_setting)
        rows.append(selected)
    summary = pd.concat(rows, ignore_index=True)
    summary = summary.sort_values(["variant", "model_family", "position_setting", "auc"], ascending=[True, True, True, False])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print(f"\nSaved to: {output}")


if __name__ == "__main__":
    main()
