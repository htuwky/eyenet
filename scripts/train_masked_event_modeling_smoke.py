from __future__ import annotations

import argparse

import pandas as pd

from eyenet.data.encoder_ready import load_feature_schema
from eyenet.training.masked_event_modeling import (
    MaskedEventModelingConfig,
    save_masked_event_modeling_outputs,
    train_masked_event_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a masked event modeling encoder smoke test.")
    parser.add_argument("--events", default="data/processed/EMS/encoder_ready/clipped_qc_no_position/ems_encoder_events.csv")
    parser.add_argument("--schema", default="configs/features/encoder_original_13feature_core.json")
    parser.add_argument("--split", default="data/splits/EMS/ems_subject_split_60_20_20_seed42.csv")
    parser.add_argument("--output-dir", default="experiments/encoder_pretraining/ems_masked_event_smoke")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--attention-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--mask-probability", type=float, default=0.15)
    parser.add_argument("--mask-strategy", choices=["span", "random"], default="span")
    parser.add_argument("--min-mask-span-events", type=int, default=2)
    parser.add_argument("--max-mask-span-events", type=int, default=8)
    parser.add_argument("--require-label", action="store_true")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    schema = load_feature_schema(args.schema)
    events = pd.read_csv(args.events, dtype={"subject_id": str}, low_memory=False)
    split = pd.read_csv(args.split, dtype={"subject_id": str})
    cfg = MaskedEventModelingConfig(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        projection_dim=args.projection_dim,
        hidden_dim=args.hidden_dim,
        attention_dim=args.attention_dim,
        dropout=args.dropout,
        random_seed=args.random_seed,
        max_seq_len=args.max_seq_len,
        mask_probability=args.mask_probability,
        mask_strategy=args.mask_strategy,
        min_mask_span_events=args.min_mask_span_events,
        max_mask_span_events=args.max_mask_span_events,
        require_label=args.require_label,
    )
    training_log, run_info = train_masked_event_model(
        events=events,
        split_subjects=split,
        feature_columns=schema["feature_columns"],
        cfg=cfg,
        device=args.device,
        checkpoint_dir=f"{args.output_dir}/checkpoints",
    )
    save_masked_event_modeling_outputs(args.output_dir, training_log, run_info)
    print("Masked event modeling smoke-test summary")
    print(training_log.tail().to_string(index=False))
    print()
    print(run_info)


if __name__ == "__main__":
    main()
