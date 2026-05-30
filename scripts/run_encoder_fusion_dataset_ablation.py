from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

FUSION_DATASETS = {
    "ems_only": "data/processed/EMS/encoder_ready/clipped_qc_no_position/ems_encoder_events.csv",
    "ems_crcns_eye1": "data/processed/mixed/ems_crcns_eye1_encoder_events.csv",
    "ems_gazebase_crcns_eye1": "data/processed/mixed/ems_gazebase_crcns_eye1_encoder_events.csv",
    "ems_gazebase_crcns_eye1_onestop": "data/processed/mixed/ems_gazebase_crcns_eye1_onestop_encoder_events.csv",
    "ems_gazebase_crcns_eye1_hbn": "data/processed/mixed/ems_gazebase_crcns_eye1_hbn_encoder_events.csv",
    "ems_all_public": "data/processed/mixed/ems_all_public_encoder_events.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run encoder MEM pretraining dataset-fusion ablations with aligned EMS downstream splits."
    )
    parser.add_argument(
        "--datasets",
        default="ems_only,ems_crcns_eye1,ems_gazebase_crcns_eye1,ems_gazebase_crcns_eye1_onestop",
        help=f"Comma-separated dataset keys. Choices: {','.join(FUSION_DATASETS)}",
    )
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--anchor-dataset", default="EMS")
    parser.add_argument("--anchor-split-dir", default="data/splits/EMS/multiseed")
    parser.add_argument("--aligned-split-dir", default="data/processed/mixed/multiseed_aligned_fusion")
    parser.add_argument("--pretrain-root", default="experiments/encoder_pretraining/fusion_ablation")
    parser.add_argument("--downstream-root", default="experiments/encoder_downstream/fusion_ablation")
    parser.add_argument("--pretrain-schema", default="configs/features/encoder_no_position_core.json")
    parser.add_argument("--downstream-schema", default="data/processed/EMS/encoder_ready/clipped_qc_no_position/feature_schema.json")
    parser.add_argument("--name-prefix", default="bigru64_mask045_fusion")
    parser.add_argument("--encoder-type", choices=["bigru_attention", "transformer"], default="bigru_attention")
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--attention-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--feedforward-dim", type=int, default=256)
    parser.add_argument("--mask-probability", type=float, default=0.45)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=1500)
    parser.add_argument("--mem-max-epochs", type=int, default=100)
    parser.add_argument("--mem-patience", type=int, default=12)
    parser.add_argument("--supervised-max-epochs", type=int, default=100)
    parser.add_argument("--supervised-patience", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stages", default="split,experiment", help="Comma-separated stages: split,experiment.")
    parser.add_argument(
        "--architecture-stages",
        default="mem,finetune,frozen",
        help="Comma-separated stages passed to run_encoder_architecture_experiment.py: mem,finetune,frozen.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_keys = parse_dataset_keys(args.datasets)
    seeds = parse_ints(args.seeds)
    stages = {stage.strip() for stage in args.stages.split(",") if stage.strip()}
    unknown = stages - {"split", "experiment"}
    if unknown:
        raise ValueError(f"Unknown stages: {sorted(unknown)}")

    for dataset_key in dataset_keys:
        events_path = Path(FUSION_DATASETS[dataset_key])
        if not events_path.exists():
            raise FileNotFoundError(f"Missing encoder events for {dataset_key}: {events_path}")
        for seed in seeds:
            aligned_split = aligned_split_path(args, dataset_key, seed)
            if "split" in stages:
                create_aligned_split(args, events_path, seed, aligned_split)
            if "experiment" in stages:
                run_experiment(args, dataset_key, events_path, seed, aligned_split)


def parse_dataset_keys(value: str) -> list[str]:
    keys = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(keys) - set(FUSION_DATASETS))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}. Choices: {sorted(FUSION_DATASETS)}")
    return keys


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def anchor_split_path(args: argparse.Namespace, seed: int) -> Path:
    return Path(args.anchor_split_dir) / f"ems_subject_split_60_20_20_seed{seed}.csv"


def aligned_split_path(args: argparse.Namespace, dataset_key: str, seed: int) -> Path:
    return Path(args.aligned_split_dir) / dataset_key / f"{dataset_key}_subject_split_60_20_20_seed{seed}.csv"


def create_aligned_split(args: argparse.Namespace, events_path: Path, seed: int, output_path: Path) -> None:
    command = [
        sys.executable,
        "scripts/create_aligned_self_supervised_subject_split.py",
        "--events",
        str(events_path),
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


def run_experiment(
    args: argparse.Namespace,
    dataset_key: str,
    events_path: Path,
    seed: int,
    aligned_split: Path,
) -> None:
    name = f"{args.name_prefix}_{dataset_key}_seed{seed}"
    command = [
        sys.executable,
        "scripts/run_encoder_architecture_experiment.py",
        "--name",
        name,
        "--pretrain-events",
        str(events_path),
        "--pretrain-split",
        str(aligned_split),
        "--pretrain-schema",
        args.pretrain_schema,
        "--downstream-split",
        str(anchor_split_path(args, seed)),
        "--downstream-schema",
        args.downstream_schema,
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
        str(args.dropout),
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
        "--stages",
        args.architecture_stages,
    ]
    if args.force:
        command.append("--force")
    expected = Path(args.downstream_root) / name / "finetune" / "metrics.csv"
    run_stage(command, expected, args)


def run_stage(command: list[str], expected_output: Path, args: argparse.Namespace) -> None:
    if expected_output.exists() and not args.force and output_looks_valid(expected_output):
        print(f"[skip] {expected_output}")
        return
    print(format_command(command))
    if not args.dry_run:
        subprocess.run(command, check=True)


def output_looks_valid(path: Path) -> bool:
    if path.suffix.lower() != ".csv":
        return True
    try:
        header = path.read_bytes()[:4096]
    except OSError:
        return False
    if not header or b"\x00" in header:
        return False
    first_line = header.splitlines()[0].decode("utf-8-sig", errors="replace")
    columns = {column.strip() for column in first_line.split(",")}
    if path.name.endswith("_subject_split_60_20_20_seed0.csv") or "subject_split" in path.name:
        return {"subject_id", "split"}.issubset(columns)
    return bool(columns)


def format_command(command: list[str]) -> str:
    return " ".join(quote_arg(part) for part in command)


def quote_arg(value: str) -> str:
    if any(char.isspace() for char in value):
        return f'"{value}"'
    return value


if __name__ == "__main__":
    main()
