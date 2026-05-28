from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

SEED_PATTERN = re.compile(r"_seed(\d+)\.csv$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit whether EMS downstream test subjects leak into encoder pretraining train/valid splits."
    )
    parser.add_argument("--downstream-split-dir", default="data/splits/EMS/multiseed")
    parser.add_argument(
        "--pretrain-split-root",
        default="data/processed/mixed",
        help="Root containing multiseed_aligned and multiseed_aligned_fusion split files.",
    )
    parser.add_argument(
        "--output",
        default="experiments/encoder_downstream/phase1_encoder_split_leakage_audit.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    downstream_split_dir = Path(args.downstream_split_dir)
    pretrain_split_root = Path(args.pretrain_split_root)
    rows = []

    for pretrain_path in find_pretrain_splits(pretrain_split_root):
        seed = parse_seed(pretrain_path.name)
        if seed is None:
            continue
        downstream_path = downstream_split_dir / f"ems_subject_split_60_20_20_seed{seed}.csv"
        if not downstream_path.exists():
            rows.append(
                {
                    "pretrain_split": str(pretrain_path),
                    "seed": seed,
                    "status": "missing_downstream_split",
                    "passed": False,
                }
            )
            continue

        downstream = pd.read_csv(downstream_path, dtype={"subject_id": str})
        pretrain = pd.read_csv(pretrain_path, dtype={"subject_id": str, "dataset_id": str})
        downstream["subject_id"] = downstream["subject_id"].map(normalize_ems_subject_id)
        pretrain["subject_id"] = pretrain["subject_id"].map(normalize_ems_subject_id)

        test_subjects = set(downstream.loc[downstream["split"] == "test", "subject_id"])
        ems_pretrain = pretrain[pretrain["dataset_id"].astype(str).str.upper() == "EMS"].copy()

        overlaps: dict[str, set[str]] = {}
        for split_name in ["train", "valid", "test"]:
            split_subjects = set(ems_pretrain.loc[ems_pretrain["split"] == split_name, "subject_id"])
            overlaps[split_name] = test_subjects & split_subjects

        passed = len(overlaps["train"]) == 0 and len(overlaps["valid"]) == 0
        rows.append(
            {
                "pretrain_group": infer_pretrain_group(pretrain_path),
                "pretrain_split": str(pretrain_path),
                "downstream_split": str(downstream_path),
                "seed": seed,
                "n_downstream_test_subjects": len(test_subjects),
                "n_ems_subjects_in_pretrain_split": int(ems_pretrain["subject_id"].nunique()),
                "n_overlap_with_pretrain_train": len(overlaps["train"]),
                "n_overlap_with_pretrain_valid": len(overlaps["valid"]),
                "n_overlap_with_pretrain_test": len(overlaps["test"]),
                "overlap_train_subjects": ",".join(sorted(overlaps["train"])),
                "overlap_valid_subjects": ",".join(sorted(overlaps["valid"])),
                "overlap_test_subjects": ",".join(sorted(overlaps["test"])),
                "passed": passed,
                "status": "passed" if passed else "failed",
            }
        )

    audit = pd.DataFrame(rows).sort_values(["pretrain_group", "seed"])
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(audit.to_string(index=False))
    failures = audit[not audit["passed"]] if "passed" in audit.columns else audit
    if not failures.empty:
        raise SystemExit(f"Leakage audit failed for {len(failures)} split rows. See {output_path}")
    print(f"\nLeakage audit passed: {len(audit)} split rows written to {output_path}")


def find_pretrain_splits(root: Path) -> list[Path]:
    paths = []
    paths.extend((root / "multiseed_aligned").glob("*.csv"))
    paths.extend((root / "multiseed_aligned_fusion").glob("*/*.csv"))
    return sorted(paths)


def parse_seed(filename: str) -> int | None:
    match = SEED_PATTERN.search(filename)
    return int(match.group(1)) if match else None


def normalize_ems_subject_id(value: object) -> str:
    text = str(value)
    return text.zfill(3) if text.isdigit() else text


def infer_pretrain_group(path: Path) -> str:
    if path.parent.name == "multiseed_aligned":
        return path.stem.removesuffix(f"_seed{parse_seed(path.name)}")
    return path.parent.name


if __name__ == "__main__":
    main()
