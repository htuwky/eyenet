from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EMS strict-summary + pretrained encoder dual-stream over seeds.")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--fusions", default="concat,gated")
    parser.add_argument("--split-dir", default="data/splits/EMS/multiseed")
    parser.add_argument("--output-root", default="experiments/ems_summary_encoder_dual_stream")
    parser.add_argument("--name-prefix", default="bigru64_onestop_strict_summary_encoder_dual")
    parser.add_argument("--subject-summary", default="data/processed/EMS/ems_subject_summary_features.csv")
    parser.add_argument("--summary-feature-set", choices=["full", "strict"], default="strict")
    parser.add_argument("--encoder-events", default="data/processed/EMS/encoder_ready/clipped_qc_no_position/ems_encoder_events.csv")
    parser.add_argument("--encoder-schema", default="configs/features/encoder_original_13feature_core.json")
    parser.add_argument(
        "--checkpoint-template",
        default=(
            "experiments/encoder_pretraining/fusion_ablation/"
            "bigru64_mask045_fusion_ems_gazebase_crcns_eye1_onestop_seed{seed}/checkpoints/best.pt"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--attention-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--summary-dim", type=int, default=16)
    parser.add_argument("--summary-residual-scale", type=float, default=0.25)
    parser.add_argument("--pos-weight", type=float, default=1.5)
    parser.add_argument("--max-seq-len", type=int, default=1500)
    parser.add_argument("--max-train-missing-rate", type=float, default=0.4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for seed in parse_ints(args.seeds):
        for fusion in parse_items(args.fusions):
            run_one(args, seed, fusion)


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_items(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(items) - {"concat", "gated", "residual_logit"})
    if unknown:
        raise ValueError(f"Unknown fusions: {unknown}. Choices: concat,gated,residual_logit")
    return items


def run_one(args: argparse.Namespace, seed: int, fusion: str) -> None:
    experiment_name = f"{args.name_prefix}_seed{seed}"
    output_dir = Path(args.output_root) / experiment_name / fusion
    expected = output_dir / "metrics.csv"
    if expected.exists() and not args.force:
        print(f"[skip] {expected}")
        return
    checkpoint = args.checkpoint_template.format(seed=seed)
    if not Path(checkpoint).exists():
        raise FileNotFoundError(f"Missing pretrained checkpoint for seed {seed}: {checkpoint}")
    command = [
        sys.executable,
        "scripts/train_ems_summary_encoder_dual_stream_fixed_split.py",
        "--subject-summary",
        args.subject_summary,
        "--summary-feature-set",
        args.summary_feature_set,
        "--encoder-events",
        args.encoder_events,
        "--encoder-schema",
        args.encoder_schema,
        "--split",
        str(Path(args.split_dir) / f"ems_subject_split_60_20_20_seed{seed}.csv"),
        "--output-dir",
        str(output_dir),
        "--fusion",
        fusion,
        "--pretrained-checkpoint",
        checkpoint,
        "--batch-size",
        str(args.batch_size),
        "--max-epochs",
        str(args.max_epochs),
        "--patience",
        str(args.patience),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--projection-dim",
        str(args.projection_dim),
        "--hidden-dim",
        str(args.hidden_dim),
        "--attention-dim",
        str(args.attention_dim),
        "--dropout",
        str(args.dropout),
        "--summary-dim",
        str(args.summary_dim),
        "--summary-residual-scale",
        str(args.summary_residual_scale),
        "--random-seed",
        str(seed),
        "--pos-weight",
        str(args.pos_weight),
        "--max-seq-len",
        str(args.max_seq_len),
        "--max-train-missing-rate",
        str(args.max_train_missing_rate),
        "--device",
        args.device,
    ]
    if args.freeze_encoder:
        command.append("--freeze-encoder")
    print(format_command(command))
    if not args.dry_run:
        subprocess.run(command, check=True)


def format_command(command: list[str]) -> str:
    return " ".join(quote_arg(part) for part in command)


def quote_arg(value: str) -> str:
    if any(char.isspace() for char in value):
        return f'"{value}"'
    return value


if __name__ == "__main__":
    main()
