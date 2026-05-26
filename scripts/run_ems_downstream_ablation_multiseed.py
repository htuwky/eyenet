from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


VARIANTS = {
    "macro": "scripts/train_ems_segment_sequence_fixed_split.py",
    "event": "scripts/train_ems_event_temporal_sequence_fixed_split.py",
    "concat": "scripts/train_ems_dual_stream_concat_fixed_split.py",
    "gated": "scripts/train_ems_dual_stream_gated_fixed_split.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EMS downstream single-stream and dual-stream ablations.")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--variants", default="macro,event,concat,gated")
    parser.add_argument("--split-dir", default="data/splits/EMS/multiseed")
    parser.add_argument("--output-root", default="experiments/ems_downstream_ablation")
    parser.add_argument("--name-prefix", default="ems_downstream")
    parser.add_argument("--macro-segments", default="data/processed/EMS/ems_segment_features_no_pupil.csv")
    parser.add_argument("--event-sequences", default="data/processed/EMS/ems_event_temporal_sequences_no_pupil.csv")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--attention-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--pos-weight", type=float, default=1.5)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variants = parse_variants(args.variants)
    for seed in parse_ints(args.seeds):
        for variant in variants:
            run_variant(args, seed, variant)


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_variants(value: str) -> list[str]:
    variants = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}. Choices: {sorted(VARIANTS)}")
    return variants


def run_variant(args: argparse.Namespace, seed: int, variant: str) -> None:
    experiment_name = f"{args.name_prefix}_seed{seed}"
    output_dir = Path(args.output_root) / experiment_name / variant
    expected = output_dir / "metrics.csv"
    if expected.exists() and not args.force:
        print(f"[skip] {expected}")
        return

    command = [
        sys.executable,
        VARIANTS[variant],
        "--split",
        str(Path(args.split_dir) / f"ems_subject_split_60_20_20_seed{seed}.csv"),
        "--output-dir",
        str(output_dir),
        "--batch-size",
        str(args.batch_size),
        "--max-epochs",
        str(args.max_epochs),
        "--patience",
        str(args.patience),
        "--projection-dim",
        str(args.projection_dim),
        "--hidden-dim",
        str(args.hidden_dim),
        "--attention-dim",
        str(args.attention_dim),
        "--dropout",
        str(args.dropout),
        "--random-seed",
        str(seed),
        "--pos-weight",
        str(args.pos_weight),
        "--device",
        args.device,
    ]
    if variant == "macro":
        command.extend(["--segments", args.macro_segments])
    elif variant == "event":
        command.extend(["--events", args.event_sequences])
        append_optional_max_events(command, args.max_events)
    else:
        command.extend(["--macro-segments", args.macro_segments])
        command.extend(["--event-sequences", args.event_sequences])
        append_optional_max_events(command, args.max_events)

    print(format_command(command))
    if not args.dry_run:
        subprocess.run(command, check=True)


def append_optional_max_events(command: list[str], max_events: int | None) -> None:
    if max_events is not None:
        command.extend(["--max-events", str(max_events)])


def format_command(command: list[str]) -> str:
    return " ".join(quote_arg(part) for part in command)


def quote_arg(value: str) -> str:
    if any(char.isspace() for char in value):
        return f'"{value}"'
    return value


if __name__ == "__main__":
    main()
