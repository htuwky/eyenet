from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one encoder architecture pretraining/downstream experiment.")
    parser.add_argument("--name", required=True, help="Experiment name used under output roots.")
    parser.add_argument("--pretrain-events", required=True)
    parser.add_argument("--pretrain-schema", default="configs/features/encoder_no_position_core.json")
    parser.add_argument("--pretrain-split", required=True)
    parser.add_argument("--downstream-events", default="data/processed/EMS/encoder_ready/clipped_qc_no_position/ems_encoder_events.csv")
    parser.add_argument("--downstream-schema", default="data/processed/EMS/encoder_ready/clipped_qc_no_position/feature_schema.json")
    parser.add_argument("--downstream-split", default="data/splits/EMS/ems_subject_split_60_20_20_seed42.csv")
    parser.add_argument("--pretrain-root", default="experiments/encoder_pretraining/architecture_ablation")
    parser.add_argument("--downstream-root", default="experiments/encoder_downstream/architecture_ablation")
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
    parser.add_argument("--mem-max-epochs", type=int, default=100)
    parser.add_argument("--mem-patience", type=int, default=12)
    parser.add_argument("--supervised-max-epochs", type=int, default=100)
    parser.add_argument("--supervised-patience", type=int, default=12)
    parser.add_argument("--mask-probability", type=float, default=0.30)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stages", default="mem,finetune,frozen", help="Comma-separated stages: mem,finetune,frozen.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stages = {stage.strip() for stage in args.stages.split(",") if stage.strip()}
    unknown = stages - {"mem", "finetune", "frozen"}
    if unknown:
        raise ValueError(f"Unknown stages: {sorted(unknown)}")

    pretrain_dir = Path(args.pretrain_root) / args.name
    checkpoint = pretrain_dir / "checkpoints" / "best.pt"

    if "mem" in stages:
        command = [
            sys.executable,
            "scripts/train_mem_pretrain.py",
            "--events",
            args.pretrain_events,
            "--schema",
            args.pretrain_schema,
            "--split",
            args.pretrain_split,
            "--output-dir",
            str(pretrain_dir),
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
            str(args.mem_max_epochs),
            "--patience",
            str(args.mem_patience),
            "--max-seq-len",
            str(args.max_seq_len),
            "--mask-probability",
            str(args.mask_probability),
            "--random-seed",
            str(args.random_seed),
            "--device",
            args.device,
        ]
        run_stage(command, checkpoint, args)

    if "finetune" in stages:
        run_supervised_stage("finetune", checkpoint, freeze=False, args=args)
    if "frozen" in stages:
        run_supervised_stage("frozen", checkpoint, freeze=True, args=args)


def run_supervised_stage(name: str, checkpoint: Path, freeze: bool, args: argparse.Namespace) -> None:
    output_dir = Path(args.downstream_root) / args.name / name
    command = [
        sys.executable,
        "scripts/train_supervised_encoder_smoke.py",
        "--events",
        args.downstream_events,
        "--schema",
        args.downstream_schema,
        "--split",
        args.downstream_split,
        "--output-dir",
        str(output_dir),
        "--batch-size",
        str(args.batch_size),
        "--max-epochs",
        str(args.supervised_max_epochs),
        "--patience",
        str(args.supervised_patience),
        "--max-seq-len",
        str(args.max_seq_len),
        "--random-seed",
        str(args.random_seed),
        "--device",
        args.device,
        "--pretrained-checkpoint",
        str(checkpoint),
    ]
    if freeze:
        command.append("--freeze-encoder")
    run_stage(command, output_dir / "metrics.csv", args)


def run_stage(command: list[str], expected_output: Path, args: argparse.Namespace) -> None:
    if expected_output.exists() and not args.force:
        print(f"[skip] {expected_output}")
        return
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
