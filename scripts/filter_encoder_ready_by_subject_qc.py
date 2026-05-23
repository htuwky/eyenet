from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter an encoder-ready event table using a subject-level QC report.")
    parser.add_argument("--events", required=True)
    parser.add_argument("--qc-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--usable-column", default="usable_for_self_supervised_pretraining")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events = pd.read_csv(args.events, dtype={"subject_id": str}, low_memory=False)
    qc = pd.read_csv(args.qc_report, dtype={"subject_id": str}, low_memory=False)
    if args.usable_column not in qc.columns:
        raise ValueError(f"QC report does not contain usable column: {args.usable_column}")

    events["_subject_key"] = make_subject_key(events)
    qc["_subject_key"] = make_subject_key(qc)
    keep_keys = set(qc.loc[qc[args.usable_column].astype(bool), "_subject_key"])
    filtered = events.loc[events["_subject_key"].isin(keep_keys)].drop(columns=["_subject_key"]).copy()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output_path, index=False, encoding="utf-8-sig")

    summary = {
        "source_events": args.events,
        "qc_report": args.qc_report,
        "usable_column": args.usable_column,
        "output": str(output_path),
        "n_subjects_before": int(events["_subject_key"].nunique()),
        "n_subjects_after": int(filtered[["dataset_id", "subject_id"]].drop_duplicates().shape[0]),
        "n_events_before": int(len(events)),
        "n_events_after": int(len(filtered)),
        "dataset_counts_subjects_after": {
            str(k): int(v)
            for k, v in filtered[["dataset_id", "subject_id"]]
            .drop_duplicates()
            ["dataset_id"]
            .value_counts()
            .sort_index()
            .items()
        },
    }
    output_path.with_name(output_path.stem + "_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def make_subject_key(frame: pd.DataFrame) -> pd.Series:
    if "dataset_id" in frame.columns:
        return frame["dataset_id"].astype(str) + "::" + frame["subject_id"].astype(str).str.zfill(3)
    return frame["subject_id"].astype(str).str.zfill(3)


if __name__ == "__main__":
    main()
