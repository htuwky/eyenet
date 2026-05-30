from __future__ import annotations

import argparse
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the phase-1 EMS+GazeBase+CRCNS+OneStop BiGRU fine-tune trend-only feature ablation. "
            "This is designed to compare against the original 13-feature phase-1 OneStop fine-tune row."
        )
    )
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--schema", default="configs/features/encoder_trend_only_core.json")
    parser.add_argument("--pretrain-root", default="experiments/encoder_pretraining/fusion_ablation_trend_only")
    parser.add_argument("--downstream-root", default="experiments/encoder_downstream/fusion_ablation_trend_only")
    parser.add_argument("--name-prefix", default="bigru64_mask045_fusion_trend_only")
    parser.add_argument("--anchor-split-dir", default="data/splits/EMS/multiseed")
    parser.add_argument("--aligned-split-dir", default="data/processed/mixed/multiseed_aligned_fusion")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=1500)
    parser.add_argument("--mem-max-epochs", type=int, default=100)
    parser.add_argument("--mem-patience", type=int, default=12)
    parser.add_argument("--supervised-max-epochs", type=int, default=100)
    parser.add_argument("--supervised-patience", type=int, default=12)
    parser.add_argument("--mask-probability", type=float, default=0.45)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--architecture-stages",
        default="mem,finetune",
        help="Internal architecture stages to run. Default omits frozen for this fine-tune comparison.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = [
        sys.executable,
        "scripts/run_encoder_fusion_dataset_ablation.py",
        "--datasets",
        "ems_gazebase_crcns_eye1_onestop",
        "--seeds",
        args.seeds,
        "--anchor-split-dir",
        args.anchor_split_dir,
        "--aligned-split-dir",
        args.aligned_split_dir,
        "--pretrain-root",
        args.pretrain_root,
        "--downstream-root",
        args.downstream_root,
        "--pretrain-schema",
        args.schema,
        "--downstream-schema",
        args.schema,
        "--name-prefix",
        args.name_prefix,
        "--mask-probability",
        str(args.mask_probability),
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
        "--stages",
        "split,experiment",
        "--architecture-stages",
        args.architecture_stages,
    ]
    if args.force:
        command.append("--force")
    if args.dry_run:
        command.append("--dry-run")

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
