from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze concat vs gated dual-stream fusion behavior.")
    parser.add_argument("--concat-dir", default="experiments/ems_fixed_split/dual_stream_concat")
    parser.add_argument("--gated-dir", default="experiments/ems_fixed_split/dual_stream_gated")
    parser.add_argument("--output-dir", default="experiments/ems_fixed_split/summary/dual_stream_fusion")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    concat_attention = pd.read_csv(Path(args.concat_dir) / "attention_weights.csv", dtype={"subject_id": str})
    gated_attention = pd.read_csv(Path(args.gated_dir) / "attention_weights.csv", dtype={"subject_id": str})
    concat_metrics = pd.read_csv(Path(args.concat_dir) / "metrics.csv", dtype={"subject_id": str})
    gated_metrics = pd.read_csv(Path(args.gated_dir) / "metrics.csv", dtype={"subject_id": str})
    concat_predictions = pd.read_csv(Path(args.concat_dir) / "predictions.csv", dtype={"subject_id": str})
    gated_predictions = pd.read_csv(Path(args.gated_dir) / "predictions.csv", dtype={"subject_id": str})

    gate_subjects = build_gate_subject_summary(gated_attention)
    gate_by_split = summarize_gate_by_split(gate_subjects)
    attention_by_stream = summarize_attention_by_stream(concat_attention, gated_attention)
    metric_comparison = compare_primary_metrics(concat_metrics, gated_metrics)
    prediction_comparison = compare_primary_predictions(concat_predictions, gated_predictions)

    gate_subjects.to_csv(output_dir / "gated_subject_gate_weights.csv", index=False, encoding="utf-8-sig")
    gate_by_split.to_csv(output_dir / "gated_gate_summary_by_split.csv", index=False, encoding="utf-8-sig")
    attention_by_stream.to_csv(output_dir / "dual_stream_attention_summary_by_stream.csv", index=False, encoding="utf-8-sig")
    metric_comparison.to_csv(output_dir / "concat_vs_gated_primary_metric_comparison.csv", index=False, encoding="utf-8-sig")
    prediction_comparison.to_csv(output_dir / "concat_vs_gated_subject_predictions.csv", index=False, encoding="utf-8-sig")

    print("Concat vs gated primary metric comparison")
    print(metric_comparison.to_string(index=False))
    print()
    print("Gated fusion gate summary")
    print(gate_by_split.to_string(index=False))
    print()
    print("Outputs")
    print(f"gate_subjects: {output_dir / 'gated_subject_gate_weights.csv'}")
    print(f"gate_summary: {output_dir / 'gated_gate_summary_by_split.csv'}")
    print(f"attention_summary: {output_dir / 'dual_stream_attention_summary_by_stream.csv'}")
    print(f"metric_comparison: {output_dir / 'concat_vs_gated_primary_metric_comparison.csv'}")
    print(f"prediction_comparison: {output_dir / 'concat_vs_gated_subject_predictions.csv'}")


def build_gate_subject_summary(gated_attention: pd.DataFrame) -> pd.DataFrame:
    required = {"split", "subject_id", "label", "probability", "prediction", "macro_gate", "event_gate"}
    missing = required - set(gated_attention.columns)
    if missing:
        raise ValueError(f"Gated attention file is missing columns: {sorted(missing)}")
    subject_rows = (
        gated_attention[list(required)]
        .drop_duplicates()
        .sort_values(["split", "subject_id"])
        .reset_index(drop=True)
    )
    subject_rows["gate_margin_macro_minus_event"] = subject_rows["macro_gate"] - subject_rows["event_gate"]
    subject_rows["macro_dominant"] = subject_rows["macro_gate"] >= 0.5
    subject_rows["strong_macro_dominant_0_90"] = subject_rows["macro_gate"] >= 0.9
    return subject_rows


def summarize_gate_by_split(gate_subjects: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, group in gate_subjects.groupby("split"):
        rows.append(
            {
                "split": split,
                "n_subjects": int(len(group)),
                "macro_gate_mean": float(group["macro_gate"].mean()),
                "macro_gate_std": float(group["macro_gate"].std(ddof=0)),
                "macro_gate_min": float(group["macro_gate"].min()),
                "macro_gate_median": float(group["macro_gate"].median()),
                "macro_gate_max": float(group["macro_gate"].max()),
                "event_gate_mean": float(group["event_gate"].mean()),
                "macro_dominant_rate": float(group["macro_dominant"].mean()),
                "strong_macro_dominant_0_90_rate": float(group["strong_macro_dominant_0_90"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("split").reset_index(drop=True)


def summarize_attention_by_stream(concat_attention: pd.DataFrame, gated_attention: pd.DataFrame) -> pd.DataFrame:
    concat_summary = stream_attention_summary(concat_attention, "dual_concat")
    gated_summary = stream_attention_summary(gated_attention, "dual_gated")
    return pd.concat([concat_summary, gated_summary], ignore_index=True)


def stream_attention_summary(attention: pd.DataFrame, model_name: str) -> pd.DataFrame:
    rows = []
    for (split, stream), group in attention.groupby(["split", "stream"]):
        subject_top = (
            group.sort_values(["subject_id", "attention_weight"], ascending=[True, False])
            .groupby("subject_id")
            .head(10)
        )
        rows.append(
            {
                "model": model_name,
                "split": split,
                "stream": stream,
                "n_rows": int(len(group)),
                "n_subjects": int(group["subject_id"].nunique()),
                "attention_mean": float(group["attention_weight"].mean()),
                "attention_std": float(group["attention_weight"].std(ddof=0)),
                "attention_max_mean_by_subject": float(group.groupby("subject_id")["attention_weight"].max().mean()),
                "top10_attention_mean": float(subject_top["attention_weight"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "split", "stream"]).reset_index(drop=True)


def compare_primary_metrics(concat_metrics: pd.DataFrame, gated_metrics: pd.DataFrame) -> pd.DataFrame:
    concat = select_primary_test_row(concat_metrics).copy()
    gated = select_primary_test_row(gated_metrics).copy()
    concat.insert(0, "fusion_model", "dual_concat")
    gated.insert(0, "fusion_model", "dual_gated")
    combined = pd.concat([concat, gated], ignore_index=True)
    metric_cols = ["auc", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1"]
    diff = {"fusion_model": "gated_minus_concat", "model": "difference", "split": "test"}
    for metric in metric_cols:
        diff[metric] = float(gated.iloc[0][metric] - concat.iloc[0][metric])
    return pd.concat([combined, pd.DataFrame([diff])], ignore_index=True)


def select_primary_test_row(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = metrics[(metrics["split"] == "test") & (metrics["threshold_name"] == "valid_best_balanced_accuracy")]
    if rows.empty:
        raise ValueError("Could not find test/valid_best_balanced_accuracy row.")
    return rows.head(1).reset_index(drop=True)


def compare_primary_predictions(concat_predictions: pd.DataFrame, gated_predictions: pd.DataFrame) -> pd.DataFrame:
    concat = select_primary_predictions(concat_predictions, "concat")
    gated = select_primary_predictions(gated_predictions, "gated")
    merged = concat.merge(gated, on=["subject_id", "label"], how="inner")
    merged["probability_delta_gated_minus_concat"] = merged["probability_gated"] - merged["probability_concat"]
    merged["concat_correct"] = merged["prediction_concat"] == merged["label"]
    merged["gated_correct"] = merged["prediction_gated"] == merged["label"]
    merged["correctness_change"] = np.select(
        [
            merged["concat_correct"] & ~merged["gated_correct"],
            ~merged["concat_correct"] & merged["gated_correct"],
            merged["concat_correct"] & merged["gated_correct"],
        ],
        ["concat_only_correct", "gated_only_correct", "both_correct"],
        default="both_wrong",
    )
    return merged.sort_values(["correctness_change", "subject_id"]).reset_index(drop=True)


def select_primary_predictions(predictions: pd.DataFrame, suffix: str) -> pd.DataFrame:
    rows = predictions[
        (predictions["split"] == "test") & (predictions["threshold_name"] == "valid_best_balanced_accuracy")
    ].copy()
    if rows.empty:
        raise ValueError("Could not find primary test predictions.")
    return rows[["subject_id", "label", "probability", "prediction"]].rename(
        columns={"probability": f"probability_{suffix}", "prediction": f"prediction_{suffix}"}
    )


if __name__ == "__main__":
    main()
