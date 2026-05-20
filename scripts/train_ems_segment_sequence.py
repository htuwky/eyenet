from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from eyenet.training.segment_sequence import (
    SegmentSequenceConfig,
    save_segment_sequence_outputs,
    summarize_across_seeds,
    summarize_deep_metrics,
    train_official_folds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the EMS segment sequence deep baseline.")
    parser.add_argument("--segments", default="data/processed/EMS/ems_segment_features_no_pupil.csv")
    parser.add_argument("--output-dir", default="experiments/ems_segment_sequence")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--attention-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--pos-weight", type=float, default=1.0)
    parser.add_argument("--device", default=None, help="Use cuda, cpu, or leave empty for auto.")
    parser.add_argument("--no-segment-position", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    segments = pd.read_csv(args.segments, dtype={"subject_id": str})
    seeds = args.seeds if args.seeds else [args.random_seed]
    seed_summaries: list[pd.DataFrame] = []

    for seed in seeds:
        seed_output_dir = Path(args.output_dir)
        if len(seeds) > 1:
            seed_output_dir = seed_output_dir / f"seed_{seed}"

        run_one_seed(args, segments, seed, seed_output_dir, seed_summaries)

    if len(seeds) > 1:
        all_seed_summaries = pd.concat(seed_summaries, ignore_index=True)
        all_seed_summaries.to_csv(Path(args.output_dir) / "summary_by_seed.csv", index=False, encoding="utf-8-sig")
        across = summarize_across_seeds(all_seed_summaries)
        across.to_csv(Path(args.output_dir) / "summary_across_seeds.csv", index=False, encoding="utf-8-sig")
        print("Summary by seed")
        print(all_seed_summaries.to_string(index=False))
        print()
        print("Summary across seeds")
        print(across.to_string(index=False))


def run_one_seed(args, segments: pd.DataFrame, seed: int, output_dir: Path, seed_summaries: list[pd.DataFrame]) -> None:
    cfg = SegmentSequenceConfig(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        projection_dim=args.projection_dim,
        hidden_dim=args.hidden_dim,
        attention_dim=args.attention_dim,
        dropout=args.dropout,
        random_seed=seed,
        use_segment_position=not args.no_segment_position,
        pos_weight=args.pos_weight,
    )
    fold_metrics, predictions, training_log, attention_weights, run_info = train_official_folds(
        segments,
        cfg,
        device=args.device,
        checkpoint_dir=output_dir / "checkpoints",
    )
    summary = summarize_deep_metrics(fold_metrics)
    summary.insert(0, "seed", seed)
    summary.insert(1, "pos_weight", args.pos_weight)
    seed_summaries.append(summary)
    save_summary = summary.drop(columns=["seed", "pos_weight"])
    save_segment_sequence_outputs(
        output_dir,
        fold_metrics,
        save_summary,
        predictions,
        training_log,
        attention_weights,
        run_info,
    )
    print(f"Seed {seed}, pos_weight={args.pos_weight}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
