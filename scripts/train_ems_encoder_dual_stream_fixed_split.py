from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from eyenet.data.encoder_ready import load_feature_schema
from eyenet.training.encoder_dual_stream import (
    EncoderDualStreamConfig,
    load_split_subjects,
    save_encoder_dual_stream_outputs,
    train_fixed_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train EMS macro + pretrained encoder dual-stream fusion on a fixed split."
    )
    parser.add_argument("--macro-segments", default="data/processed/EMS/ems_segment_features_no_pupil.csv")
    parser.add_argument("--encoder-events", default="data/processed/EMS/encoder_ready/clipped_qc_no_position/ems_encoder_events.csv")
    parser.add_argument("--encoder-schema", default="data/processed/EMS/encoder_ready/clipped_qc_no_position/feature_schema.json")
    parser.add_argument("--split", default="data/splits/EMS/ems_subject_split_60_20_20_seed42.csv")
    parser.add_argument("--output-dir", default="experiments/ems_encoder_dual_stream/seed42/gated")
    parser.add_argument("--fusion", choices=["concat", "gated"], default="gated")
    parser.add_argument("--pretrained-checkpoint", required=True)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--attention-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--feedforward-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--pos-weight", type=float, default=1.5)
    parser.add_argument("--max-seq-len", type=int, default=1500)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-segment-position", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    macro_segments = pd.read_csv(args.macro_segments, dtype={"subject_id": str})
    encoder_events = pd.read_csv(args.encoder_events, dtype={"subject_id": str}, low_memory=False)
    split_subjects = load_split_subjects(args.split)
    schema = load_feature_schema(args.encoder_schema)
    cfg = EncoderDualStreamConfig(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        fusion=args.fusion,
        projection_dim=args.projection_dim,
        hidden_dim=args.hidden_dim,
        attention_dim=args.attention_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        feedforward_dim=args.feedforward_dim,
        dropout=args.dropout,
        random_seed=args.random_seed,
        pos_weight=args.pos_weight,
        use_segment_position=not args.no_segment_position,
        max_seq_len=args.max_seq_len,
        gradient_clip_norm=args.gradient_clip_norm,
        pretrained_checkpoint=args.pretrained_checkpoint,
        freeze_encoder=args.freeze_encoder,
    )
    output_dir = Path(args.output_dir)
    (
        metrics,
        predictions,
        valid_threshold_metrics,
        selected_thresholds,
        training_log,
        attention_weights,
        run_info,
    ) = train_fixed_split(
        macro_segments=macro_segments,
        encoder_events=encoder_events,
        split_subjects=split_subjects,
        encoder_feature_columns=schema["feature_columns"],
        cfg=cfg,
        device=args.device,
        checkpoint_dir=output_dir / "checkpoints",
    )
    save_encoder_dual_stream_outputs(
        output_dir=output_dir,
        metrics=metrics,
        predictions=predictions,
        valid_threshold_metrics=valid_threshold_metrics,
        selected_thresholds=selected_thresholds,
        training_log=training_log,
        attention_weights=attention_weights,
        run_info=run_info,
    )
    test_rows = metrics[(metrics["split"] == "test") & (metrics["threshold_name"] == "valid_best_balanced_accuracy")]
    print("EMS encoder dual-stream test metrics using validation-selected best balanced-accuracy threshold")
    print(test_rows.to_string(index=False))
    print("\nValidation-selected thresholds")
    print(selected_thresholds.to_string(index=False))
    print("\nTraining log tail")
    print(training_log.tail().to_string(index=False))


if __name__ == "__main__":
    main()

