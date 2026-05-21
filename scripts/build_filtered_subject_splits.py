from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter a subject split file using a subject QC report.")
    parser.add_argument("--split", default="data/splits/EMS/ems_subject_split_60_20_20_seed42.csv")
    parser.add_argument("--qc-report", default="data/processed/EMS/qc/subject_qc_report.csv")
    parser.add_argument("--output-dir", default="data/splits/EMS/filtered")
    parser.add_argument("--name", default="strict_qc")
    parser.add_argument(
        "--usable-column",
        default="usable_for_supervised_training",
        help="QC report boolean column used to keep subjects.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split = pd.read_csv(args.split, dtype={"subject_id": str})
    qc = pd.read_csv(args.qc_report, dtype={"subject_id": str})
    keep_subjects = set(qc.loc[qc[args.usable_column].astype(bool), "subject_id"].astype(str).str.zfill(3))
    filtered = split[split["subject_id"].astype(str).str.zfill(3).isin(keep_subjects)].copy()
    filtered["subject_id"] = filtered["subject_id"].astype(str).str.zfill(3)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / f"ems_subject_split_60_20_20_seed42_{args.name}.csv"
    output_summary = output_dir / f"ems_subject_split_60_20_20_seed42_{args.name}_summary.json"
    filtered.to_csv(output_csv, index=False, encoding="utf-8-sig")

    summary = {
        "name": args.name,
        "source_split": args.split,
        "qc_report": args.qc_report,
        "usable_column": args.usable_column,
        "n_subjects": int(filtered["subject_id"].nunique()),
        "split_summary": build_split_summary(filtered),
    }
    output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_split_summary(split: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for split_name, group in split.groupby("split", sort=True):
        labels = pd.to_numeric(group["label"], errors="coerce")
        rows.append(
            {
                "split": str(split_name),
                "n_subjects": int(group["subject_id"].nunique()),
                "n_hc": int((labels == 0).sum()),
                "n_sz": int((labels == 1).sum()),
                "sz_rate": float((labels == 1).mean()),
            }
        )
    return rows


if __name__ == "__main__":
    main()
