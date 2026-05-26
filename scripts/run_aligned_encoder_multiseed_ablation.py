from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run aligned EMS-anchor multi-seed encoder ablations sequentially."
    )
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--dropouts", default="0.3,0.4")
    parser.add_argument("--pretrain-events", default="data/processed/mixed/ems_crcns_eye1_encoder_events.csv")
    parser.add_argument("--anchor-dataset", default="EMS")
    parser.add_argument("--anchor-split-dir", default="data/splits/EMS/multiseed")
    parser.add_argument("--aligned-split-dir", default="data/processed/mixed/multiseed_aligned")
    parser.add_argument("--pretrain-root", default="experiments/encoder_pretraining/architecture_ablation")
    parser.add_argument("--downstream-root", default="experiments/encoder_downstream/architecture_ablation")
    parser.add_argument("--name-prefix", default="bigru64_ems_crcns_mask045")
    parser.add_argument("--encoder-type", choices=["bigru_attention", "transformer"], default="bigru_attention")
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--attention-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--feedforward-dim", type=int, default=256)
    parser.add_argument("--mask-probability", type=float, default=0.45)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=1500)
    parser.add_argument("--mem-max-epochs", type=int, default=100)
    parser.add_argument("--mem-patience", type=int, default=12)
    parser.add_argument("--supervised-max-epochs", type=int, default=100)
    parser.add_argument("--supervised-patience", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stages", default="split,experiment", help="Comma-separated stages: split,experiment.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = parse_ints(args.seeds)
    dropouts = parse_floats(args.dropouts)
    stages = {stage.strip() for stage in args.stages.split(",") if stage.strip()}
    unknown = stages - {"split", "experiment"}
    if unknown:
        raise ValueError(f"Unknown stages: {sorted(unknown)}")

    for seed in seeds:
        aligned_split = aligned_split_path(args, seed)
        if "split" in stages:
            create_aligned_split(args, seed, aligned_split)
        for dropout in dropouts:
            if "experiment" in stages:
                run_experiment(args, seed, dropout, aligned_split)


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def aligned_split_path(args: argparse.Namespace, seed: int) -> Path:
    return Path(args.aligned_split_dir) / f"ems_crcns_eye1_subject_split_60_20_20_seed{seed}.csv"


def anchor_split_path(args: argparse.Namespace, seed: int) -> Path:
    return Path(args.anchor_split_dir) / f"ems_subject_split_60_20_20_seed{seed}.csv"


def create_aligned_split(args: argparse.Namespace, seed: int, output_path: Path) -> None:
    command = [
        sys.executable,
        "scripts/create_aligned_self_supervised_subject_split.py",
        "--events",
        args.pretrain_events,
        "--anchor-split",
        str(anchor_split_path(args, seed)),
        "--anchor-dataset",
        args.anchor_dataset,
        "--output",
        str(output_path),
        "--seed",
        str(seed),
        "--train-size",
        "0.6",
        "--valid-size",
        "0.2",
        "--test-size",
        "0.2",
    ]
    run_stage(command, output_path, args)


def run_experiment(args: argparse.Namespace, seed: int, dropout: float, aligned_split: Path) -> None:
    name = experiment_name(args.name_prefix, dropout, seed)
    command = [
        sys.executable,
        "scripts/run_encoder_architecture_experiment.py",
        "--name",
        name,
        "--pretrain-events",
        args.pretrain_events,
        "--pretrain-split",
        str(aligned_split),
        "--downstream-split",
        str(anchor_split_path(args, seed)),
        "--pretrain-root",
        args.pretrain_root,
        "--downstream-root",
        args.downstream_root,
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
        "--mask-probability",
        str(args.mask_probability),
        "--dropout",
        str(dropout),
        "--random-seed",
        str(seed),
        "--batch-size",
        str(args.batch_size),
        "--max-seq-len",
        str(args.max_seq_len),
        "--mem-max-epochs",
        str(args.mem_max_epochs),
        "--mem-patience",
        str(args.mem_patience),
        "--supervised-max-epochs",
        str(args.supervised_max_epochs),
        "--supervised-patience",
        str(args.supervised_patience),
        "--device",
        args.device,
    ]
    if args.force:
        command.append("--force")
    expected = Path(args.downstream_root) / name / "finetune" / "metrics.csv"
    run_stage(command, expected, args)


def experiment_name(prefix: str, dropout: float, seed: int) -> str:
    if abs(dropout - 0.3) < 1e-9:
        return f"{prefix}_aligned_seed{seed}"
    dropout_token = f"dropout{int(round(dropout * 10)):02d}"
    return f"{prefix}_{dropout_token}_aligned_seed{seed}"


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
