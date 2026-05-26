from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run supervised-only encoder baselines across EMS split seeds.")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--events", default="data/processed/EMS/encoder_ready/clipped_qc_no_position/ems_encoder_events.csv")
    parser.add_argument("--schema", default="data/processed/EMS/encoder_ready/clipped_qc_no_position/feature_schema.json")
    parser.add_argument("--split-dir", default="data/splits/EMS/multiseed")
    parser.add_argument("--output-root", default="experiments/encoder_downstream/architecture_ablation")
    parser.add_argument("--name-prefix", default="bigru64_supervised_only")
    parser.add_argument("--encoder-type", choices=["bigru_attention", "transformer"], default="bigru_attention")
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--attention-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--feedforward-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=1500)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for seed in parse_ints(args.seeds):
        run_seed(args, seed)


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def run_seed(args: argparse.Namespace, seed: int) -> None:
    experiment_name = f"{args.name_prefix}_seed{seed}"
    output_dir = Path(args.output_root) / experiment_name / "supervised"
    expected = output_dir / "metrics.csv"
    if expected.exists() and not args.force:
        print(f"[skip] {expected}")
        return

    command = [
        sys.executable,
        "scripts/train_supervised_encoder_smoke.py",
        "--events",
        args.events,
        "--schema",
        args.schema,
        "--split",
        str(Path(args.split_dir) / f"ems_subject_split_60_20_20_seed{seed}.csv"),
        "--output-dir",
        str(output_dir),
        "--encoder-type",
        args.encoder_type,
        "--projection-dim",
        str(args.projection_dim),
        "--hidden-dim",
        str(args.hidden_dim),
        "--attention-dim",
        str(args.attention_dim),
        "--num-layers",
        str(args.num_layers),
        "--num-heads",
        str(args.num_heads),
        "--feedforward-dim",
        str(args.feedforward_dim),
        "--dropout",
        str(args.dropout),
        "--batch-size",
        str(args.batch_size),
        "--max-epochs",
        str(args.max_epochs),
        "--patience",
        str(args.patience),
        "--max-seq-len",
        str(args.max_seq_len),
        "--random-seed",
        str(seed),
        "--device",
        args.device,
    ]
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
