from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-seed EMS encoder pretraining/downstream experiments.")
    parser.add_argument("--seeds", default="0,1,2,3,4", help="Comma-separated random seeds.")
    parser.add_argument("--events", default="data/processed/EMS/encoder_ready/clipped_qc_no_position/ems_encoder_events.csv")
    parser.add_argument("--schema", default="configs/features/encoder_original_13feature_core.json")
    parser.add_argument("--subjects", default="data/processed/EMS/ems_subject_features_segment_agg_no_pupil.csv")
    parser.add_argument("--split-dir", default="data/splits/EMS/multiseed")
    parser.add_argument("--pretrain-root", default="experiments/encoder_pretraining/ems_mem_multiseed")
    parser.add_argument("--downstream-root", default="experiments/encoder_downstream/multiseed")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=1500)
    parser.add_argument("--mem-max-epochs", type=int, default=100)
    parser.add_argument("--mem-patience", type=int, default=12)
    parser.add_argument("--supervised-max-epochs", type=int, default=100)
    parser.add_argument("--supervised-patience", type=int, default=12)
    parser.add_argument("--mask-probability", type=float, default=0.30)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--stages",
        default="split,mem,from_scratch,mem_finetune,mem_frozen",
        help="Comma-separated stages: split,mem,from_scratch,mem_finetune,mem_frozen.",
    )
    parser.add_argument("--force", action="store_true", help="Rerun stages even if expected outputs already exist.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    stages = {stage.strip() for stage in args.stages.split(",") if stage.strip()}
    valid_stages = {"split", "mem", "from_scratch", "mem_finetune", "mem_frozen"}
    unknown = stages - valid_stages
    if unknown:
        raise ValueError(f"Unknown stages: {sorted(unknown)}")

    for seed in seeds:
        split_path = Path(args.split_dir) / f"ems_subject_split_60_20_20_seed{seed}.csv"
        pretrain_dir = Path(args.pretrain_root) / f"seed_{seed}"
        pretrain_checkpoint = pretrain_dir / "checkpoints" / "best.pt"

        if "split" in stages:
            command = [
                sys.executable,
                "scripts/create_ems_fixed_split.py",
                "--subjects",
                args.subjects,
                "--output",
                str(split_path),
                "--seed",
                str(seed),
                "--train-size",
                "0.6",
                "--valid-size",
                "0.2",
                "--test-size",
                "0.2",
            ]
            run_stage(command, split_path, args)

        if "mem" in stages:
            command = [
                sys.executable,
                "scripts/train_mem_pretrain.py",
                "--events",
                args.events,
                "--schema",
                args.schema,
                "--split",
                str(split_path),
                "--output-dir",
                str(pretrain_dir),
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
                str(seed),
                "--device",
                args.device,
            ]
            run_stage(command, pretrain_checkpoint, args)

        if "from_scratch" in stages:
            run_supervised_stage(
                name="from_scratch",
                seed=seed,
                split_path=split_path,
                checkpoint=None,
                freeze=False,
                args=args,
            )
        if "mem_finetune" in stages:
            run_supervised_stage(
                name="ems_mem_finetune",
                seed=seed,
                split_path=split_path,
                checkpoint=pretrain_checkpoint,
                freeze=False,
                args=args,
            )
        if "mem_frozen" in stages:
            run_supervised_stage(
                name="ems_mem_frozen",
                seed=seed,
                split_path=split_path,
                checkpoint=pretrain_checkpoint,
                freeze=True,
                args=args,
            )


def parse_seeds(value: str) -> list[int]:
    seeds = [int(seed.strip()) for seed in value.split(",") if seed.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def run_supervised_stage(
    name: str,
    seed: int,
    split_path: Path,
    checkpoint: Path | None,
    freeze: bool,
    args: argparse.Namespace,
) -> None:
    output_dir = Path(args.downstream_root) / name / f"seed_{seed}"
    command = [
        sys.executable,
        "scripts/train_supervised_encoder_smoke.py",
        "--events",
        args.events,
        "--schema",
        args.schema,
        "--split",
        str(split_path),
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
        str(seed),
        "--device",
        args.device,
    ]
    if checkpoint is not None:
        command.extend(["--pretrained-checkpoint", str(checkpoint)])
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
