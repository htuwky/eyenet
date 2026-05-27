from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EMS subject-summary baselines across fixed split seeds.")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--summary", default="data/processed/EMS/ems_subject_summary_features.csv")
    parser.add_argument("--split-dir", default="data/splits/EMS/multiseed")
    parser.add_argument("--output-root", default="experiments/ems_subject_summary_baseline")
    parser.add_argument("--max-train-missing-rate", type=float, default=0.4)
    parser.add_argument("--feature-set", choices=["full", "strict"], default="full")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_seeds(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    for seed in parse_seeds(args.seeds):
        split = Path(args.split_dir) / f"ems_subject_split_60_20_20_seed{seed}.csv"
        output_dir = output_root / f"seed{seed}"
        expected = output_dir / "metrics.csv"
        if expected.exists() and not args.force:
            print(f"[skip] {expected}")
            continue
        command = [
            sys.executable,
            "scripts/train_ems_subject_summary_baseline.py",
            "--summary",
            args.summary,
            "--split",
            str(split),
            "--output-dir",
            str(output_dir),
            "--random-seed",
            str(seed),
            "--max-train-missing-rate",
            str(args.max_train_missing_rate),
            "--feature-set",
            args.feature_set,
        ]
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
