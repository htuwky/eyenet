from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from eyenet.training.event_temporal_sequence import (
    EventTemporalSequenceConfig,
    save_fixed_split_event_temporal_sequence_outputs,
    train_fixed_split,
)
from eyenet.utils.config import attach_arg_defaults, cfg_arg, load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the EMS Event-Temporal Stream with a fixed train/valid/test split."
    )
    parser.add_argument("--config", default="configs/experiments/ems_event_temporal_fixed_split.yaml")
    parser.add_argument("--events", default="data/processed/EMS/ems_event_temporal_sequences_no_pupil.csv")
    parser.add_argument("--split", default="data/splits/EMS/ems_subject_split_60_20_20_seed42.csv")
    parser.add_argument("--output-dir", default="experiments/ems_fixed_split/event_temporal_sequence")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--attention-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--pos-weight", type=float, default=1.0)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--device", default=None, help="Use cuda, cpu, or leave empty for auto.")
    return attach_arg_defaults(parser, parser.parse_args())


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    args.events = cfg_arg(args, config, "events", "data.events")
    args.split = cfg_arg(args, config, "split", "data.split")
    args.output_dir = cfg_arg(args, config, "output_dir", "experiment.output_dir")
    args.random_seed = cfg_arg(args, config, "random_seed", "experiment.random_seed")
    args.batch_size = cfg_arg(args, config, "batch_size", "training.batch_size")
    args.max_epochs = cfg_arg(args, config, "max_epochs", "training.max_epochs")
    args.patience = cfg_arg(args, config, "patience", "training.patience")
    args.learning_rate = cfg_arg(args, config, "learning_rate", "training.learning_rate")
    args.weight_decay = cfg_arg(args, config, "weight_decay", "training.weight_decay")
    args.pos_weight = cfg_arg(args, config, "pos_weight", "training.pos_weight")
    args.gradient_clip_norm = cfg_arg(args, config, "gradient_clip_norm", "training.gradient_clip_norm")
    args.projection_dim = cfg_arg(args, config, "projection_dim", "model.projection_dim")
    args.hidden_dim = cfg_arg(args, config, "hidden_dim", "model.hidden_dim")
    args.attention_dim = cfg_arg(args, config, "attention_dim", "model.attention_dim")
    args.dropout = cfg_arg(args, config, "dropout", "model.dropout")
    args.max_events = cfg_arg(args, config, "max_events", "model.max_events")
    events = pd.read_csv(args.events, dtype={"subject_id": str})
    cfg = EventTemporalSequenceConfig(
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
        pos_weight=args.pos_weight,
        max_events=args.max_events,
        gradient_clip_norm=args.gradient_clip_norm,
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
        events=events,
        split_path=args.split,
        cfg=cfg,
        device=args.device,
        checkpoint_dir=output_dir / "checkpoints",
    )
    save_fixed_split_event_temporal_sequence_outputs(
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
    print("Fixed-split Event-Temporal Stream test metrics using validation-selected best balanced-accuracy threshold")
    print(test_rows.to_string(index=False))
    print("\nValidation-selected thresholds")
    print(selected_thresholds.to_string(index=False))
    print("\nTraining summary")
    print(
        pd.DataFrame(
            [
                {
                    "best_epoch": run_info["best_epoch"],
                    "stopped_epoch": run_info["stopped_epoch"],
                    "best_valid_auc": run_info["best_valid_auc"],
                    "n_features": run_info["n_features"],
                    "max_events": run_info["max_events"],
                }
            ]
        ).to_string(index=False)
    )


if __name__ == "__main__":
    main()
