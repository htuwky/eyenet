from __future__ import annotations

import argparse
from pathlib import Path

from eyenet.training.fold_distribution_diagnostics import (
    build_event_temporal_subject_summary,
    build_fold_feature_shift,
    build_fold_label_distribution,
    build_prediction_shift_joined,
    build_set1_vs_others_feature_shift,
    load_inputs,
    save_fold_distribution_diagnostics,
    summarize_fold_shift,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose EMS official fold distribution shifts.")
    parser.add_argument("--macro-features", default="data/processed/EMS/ems_subject_features_segment_agg_no_pupil.csv")
    parser.add_argument("--event-temporal", default="data/processed/EMS/ems_event_temporal_sequences_no_pupil.csv")
    parser.add_argument("--macro-predictions", default="experiments/ems_segment_sequence_pos1_5/ensemble_predictions.csv")
    parser.add_argument("--event-temporal-predictions", default="experiments/ems_event_temporal_sequence_pos1_0/predictions.csv")
    parser.add_argument("--output-dir", default="experiments/ems_fold_distribution_diagnostics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    macro, event, macro_pred, event_pred = load_inputs(
        macro_features_path=args.macro_features,
        event_temporal_path=args.event_temporal,
        macro_predictions_path=args.macro_predictions,
        event_temporal_predictions_path=args.event_temporal_predictions,
    )
    event_subject_summary = build_event_temporal_subject_summary(event)
    fold_label_distribution = build_fold_label_distribution(macro)
    macro_fold_shift = build_fold_feature_shift(macro, table_name="macro")
    event_fold_shift = build_fold_feature_shift(event_subject_summary, table_name="event_temporal")
    fold_shift_summary = summarize_fold_shift(
        __import__("pandas").concat([macro_fold_shift, event_fold_shift], ignore_index=True)
    )
    set1_macro_shift = build_set1_vs_others_feature_shift(macro, table_name="macro")
    set1_event_shift = build_set1_vs_others_feature_shift(event_subject_summary, table_name="event_temporal")
    prediction_shift_joined, prediction_shift_summary = build_prediction_shift_joined(macro, macro_pred, event_pred)

    save_fold_distribution_diagnostics(
        output_dir=Path(args.output_dir),
        fold_label_distribution=fold_label_distribution,
        macro_fold_shift=macro_fold_shift,
        event_fold_shift=event_fold_shift,
        fold_shift_summary=fold_shift_summary,
        set1_macro_shift=set1_macro_shift,
        set1_event_shift=set1_event_shift,
        prediction_shift_joined=prediction_shift_joined,
        prediction_shift_summary=prediction_shift_summary,
        event_subject_summary=event_subject_summary,
    )

    print("EMS fold distribution diagnostics outputs")
    print(f"output_dir: {args.output_dir}")
    print("\nFold label distribution")
    print(fold_label_distribution.to_string(index=False))
    print("\nFold shift summary")
    print(fold_shift_summary.to_string(index=False))
    print("\nSet_1 vs others: macro top shifts")
    print(set1_macro_shift.head(15).to_string(index=False))
    print("\nSet_1 vs others: event-temporal top shifts")
    print(set1_event_shift.head(15).to_string(index=False))
    print("\nPrediction shift summary")
    print(prediction_shift_summary.to_string(index=False))


if __name__ == "__main__":
    main()
