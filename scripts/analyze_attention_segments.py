from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from eyenet.training.attention_analysis import (
    add_prediction_columns,
    build_attention_feature_contrast,
    build_attention_group_comparison,
    build_ensemble_attention,
    build_subject_attention_summary,
    build_subject_attention_group_summary,
    build_top_segments_table,
    load_seed_attention,
    mark_top_attention_segments,
    merge_attention_with_segment_features,
    save_attention_analysis_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze segment attention weights without using stimulus content.")
    parser.add_argument("--experiment-dir", default="experiments/ems_segment_sequence_pos1_5")
    parser.add_argument("--segment-features", default="data/processed/EMS/ems_segment_features_no_pupil.csv")
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--top-n-per-subject", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir)
    output_dir = Path(args.output_dir) if args.output_dir else experiment_dir / "attention_analysis"
    predictions_path = Path(args.predictions) if args.predictions else experiment_dir / "ensemble_predictions.csv"

    seed_attention = load_seed_attention(experiment_dir)
    ensemble_attention = build_ensemble_attention(seed_attention)
    segment_features = pd.read_csv(args.segment_features, dtype={"subject_id": str})
    merged = merge_attention_with_segment_features(ensemble_attention, segment_features)

    predictions = None
    if predictions_path.exists():
        predictions = pd.read_csv(predictions_path, dtype={"subject_id": str})
    merged = add_prediction_columns(merged, predictions, threshold=args.threshold)
    marked = mark_top_attention_segments(merged, top_fraction=args.top_fraction)

    subject_summary = build_subject_attention_summary(marked, top_fraction=args.top_fraction)
    subject_group_summary = build_subject_attention_group_summary(subject_summary)
    feature_contrast = build_attention_feature_contrast(marked)
    group_comparison = build_attention_group_comparison(marked)
    top_segments = build_top_segments_table(marked, top_n_per_subject=args.top_n_per_subject)

    save_attention_analysis_outputs(
        output_dir=output_dir,
        ensemble_attention=ensemble_attention,
        merged_attention_features=marked,
        subject_summary=subject_summary,
        subject_group_summary=subject_group_summary,
        feature_contrast=feature_contrast,
        group_comparison=group_comparison,
        top_segments=top_segments,
    )

    print("Attention analysis outputs")
    print(f"output_dir: {output_dir}")
    print(f"n_subjects: {subject_summary['subject_id'].nunique()}")
    print(f"n_segments: {len(marked)}")
    print("\nTop attention-vs-rest feature contrasts")
    print(feature_contrast.head(15).to_string(index=False))
    print("\nTop SZ-vs-HC contrasts inside high-attention segments")
    print(group_comparison.head(15).to_string(index=False))
    print("\nSubject attention summary by group")
    print(subject_group_summary.to_string(index=False))


if __name__ == "__main__":
    main()
