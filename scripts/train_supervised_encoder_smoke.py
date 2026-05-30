from __future__ import annotations

import argparse

import pandas as pd

from eyenet.data.encoder_ready import load_feature_schema
from eyenet.training.supervised_encoder import (
    SupervisedEncoderConfig,
    save_supervised_encoder_outputs,
    train_supervised_encoder,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a supervised encoder smoke-test model.")
    parser.add_argument("--events", default="data/processed/EMS/encoder_ready/clipped_qc_no_position/ems_encoder_events.csv")
    parser.add_argument("--schema", default="configs/features/encoder_original_13feature_core.json")
    parser.add_argument("--split", default="data/splits/EMS/ems_subject_split_60_20_20_seed42.csv")
    parser.add_argument("--output-dir", default="experiments/encoder_smoke/ems_clipped_qc_no_position")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--encoder-type", choices=["bigru_attention", "transformer"], default="bigru_attention")
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--attention-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--feedforward-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--pos-weight", type=float, default=1.5)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-balanced-train-sampler", action="store_true")
    parser.add_argument("--pretrained-checkpoint", default=None)
    parser.add_argument("--freeze-encoder", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    schema = load_feature_schema(args.schema)
    events = pd.read_csv(args.events, dtype={"subject_id": str}, low_memory=False)
    split = pd.read_csv(args.split, dtype={"subject_id": str})
    cfg = SupervisedEncoderConfig(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        encoder_type=args.encoder_type,
        projection_dim=args.projection_dim,
        hidden_dim=args.hidden_dim,
        attention_dim=args.attention_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        feedforward_dim=args.feedforward_dim,
        dropout=args.dropout,
        random_seed=args.random_seed,
        pos_weight=args.pos_weight,
        max_seq_len=args.max_seq_len,
        balanced_train_sampler=not args.no_balanced_train_sampler,
        pretrained_checkpoint=args.pretrained_checkpoint,
        freeze_encoder=args.freeze_encoder,
    )
    (
        metrics,
        predictions,
        valid_threshold_metrics,
        selected_thresholds,
        training_log,
        run_info,
    ) = train_supervised_encoder(
        events=events,
        split_subjects=split,
        feature_columns=schema["feature_columns"],
        cfg=cfg,
        device=args.device,
        checkpoint_dir=f"{args.output_dir}/checkpoints",
    )
    save_supervised_encoder_outputs(
        output_dir=args.output_dir,
        metrics=metrics,
        predictions=predictions,
        valid_threshold_metrics=valid_threshold_metrics,
        selected_thresholds=selected_thresholds,
        training_log=training_log,
        run_info=run_info,
    )
    test_rows = metrics[(metrics["split"] == "test") & (metrics["threshold_name"] == "valid_best_balanced_accuracy")]
    print("Supervised encoder smoke-test metrics using validation-selected best balanced-accuracy threshold")
    print(test_rows.to_string(index=False))
    print("\nValidation-selected thresholds")
    print(selected_thresholds.to_string(index=False))
    print("\nTraining log tail")
    print(training_log.tail().to_string(index=False))


if __name__ == "__main__":
    main()
