from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose subject-level encoder sequence lengths and max_seq_len truncation risk."
    )
    parser.add_argument("--events", required=True, help="Encoder-ready events CSV.")
    parser.add_argument("--split", default=None, help="Optional subject split CSV used to attach train/valid/test labels.")
    parser.add_argument("--output-dir", default=None, help="Optional directory for CSV/JSON diagnostic outputs.")
    parser.add_argument("--max-seq-len", type=int, default=1500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events = pd.read_csv(args.events, dtype={"subject_id": str, "dataset_id": str}, low_memory=False)
    subject_lengths = build_subject_lengths(events)
    if args.split:
        subject_lengths = attach_split(subject_lengths, args.split)
    subject_lengths = add_truncation_columns(subject_lengths, args.max_seq_len)
    summary = summarize_lengths(subject_lengths)
    report = build_report(subject_lengths, args)

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        subject_lengths.to_csv(output_dir / "subject_sequence_lengths.csv", index=False, encoding="utf-8-sig")
        summary.to_csv(output_dir / "sequence_length_summary.csv", index=False, encoding="utf-8-sig")
        (output_dir / "truncation_summary.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print("Sequence length summary")
    print(summary.to_string(index=False))
    print()
    print(json.dumps(report, ensure_ascii=False, indent=2))


def build_subject_lengths(events: pd.DataFrame) -> pd.DataFrame:
    data = events.copy()
    if "dataset_id" not in data.columns:
        data["dataset_id"] = "dataset"
    if "label" not in data.columns:
        data["label"] = pd.NA
    group_cols = ["dataset_id", "subject_id"]
    rows = (
        data.groupby(group_cols, dropna=False)
        .agg(
            label=("label", first_non_null),
            event_split=("split", first_non_null) if "split" in data.columns else ("subject_id", lambda _: pd.NA),
            n_events=("subject_id", "size"),
        )
        .reset_index()
    )
    rows["subject_id"] = rows["subject_id"].astype(str)
    rows["dataset_id"] = rows["dataset_id"].astype(str)
    return rows


def attach_split(subject_lengths: pd.DataFrame, split_path: str | Path) -> pd.DataFrame:
    split = pd.read_csv(split_path, dtype={"subject_id": str, "dataset_id": str}, low_memory=False)
    if "dataset_id" not in split.columns:
        dataset_ids = subject_lengths["dataset_id"].dropna().unique()
        if len(dataset_ids) != 1:
            raise ValueError("Split file lacks dataset_id but events contain multiple dataset_id values.")
        split["dataset_id"] = dataset_ids[0]
    required = {"dataset_id", "subject_id", "split"}
    missing = required - set(split.columns)
    if missing:
        raise ValueError(f"Split file is missing required columns: {sorted(missing)}")

    left = subject_lengths.copy()
    left["_subject_key"] = left.apply(subject_key, axis=1)
    split = split[["dataset_id", "subject_id", "split"]].copy()
    split["_subject_key"] = split.apply(subject_key, axis=1)
    split = split.drop_duplicates("_subject_key").rename(columns={"split": "split"})
    merged = left.merge(split[["_subject_key", "split"]], on="_subject_key", how="left", validate="one_to_one")
    merged["split"] = merged["split"].fillna(merged["event_split"])
    return merged.drop(columns=["_subject_key"])


def add_truncation_columns(subject_lengths: pd.DataFrame, max_seq_len: int) -> pd.DataFrame:
    out = subject_lengths.copy()
    if "split" not in out.columns:
        out["split"] = out["event_split"]
    out["max_seq_len"] = int(max_seq_len)
    out["truncated"] = out["n_events"] > max_seq_len
    out["retained_events"] = out["n_events"].clip(upper=max_seq_len)
    out["dropped_events"] = (out["n_events"] - max_seq_len).clip(lower=0)
    out["retained_fraction"] = out["retained_events"] / out["n_events"]
    return out


def summarize_lengths(subject_lengths: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["dataset_id", "split"]
    for keys, group in subject_lengths.groupby(group_cols, dropna=False, sort=True):
        row = dict(zip(group_cols, keys, strict=True))
        row.update(summary_row(group))
        rows.append(row)
    overall = {"dataset_id": "ALL", "split": "ALL", **summary_row(subject_lengths)}
    rows.append(overall)
    return pd.DataFrame(rows)


def summary_row(group: pd.DataFrame) -> dict:
    lengths = group["n_events"]
    return {
        "n_subjects": int(len(group)),
        "n_events_mean": float(lengths.mean()),
        "n_events_median": float(lengths.median()),
        "n_events_min": int(lengths.min()),
        "n_events_p90": float(lengths.quantile(0.90)),
        "n_events_p95": float(lengths.quantile(0.95)),
        "n_events_max": int(lengths.max()),
        "n_truncated": int(group["truncated"].sum()),
        "pct_truncated": float(group["truncated"].mean()),
        "dropped_events_total": int(group["dropped_events"].sum()),
        "retained_fraction_mean": float(group["retained_fraction"].mean()),
    }


def build_report(subject_lengths: pd.DataFrame, args: argparse.Namespace) -> dict:
    truncated = subject_lengths[subject_lengths["truncated"]].copy()
    top_truncated = truncated.sort_values("dropped_events", ascending=False).head(10)
    return {
        "events": args.events,
        "split": args.split,
        "max_seq_len": int(args.max_seq_len),
        "n_subjects": int(len(subject_lengths)),
        "n_truncated_subjects": int(subject_lengths["truncated"].sum()),
        "pct_truncated_subjects": float(subject_lengths["truncated"].mean()),
        "dropped_events_total": int(subject_lengths["dropped_events"].sum()),
        "retained_fraction_mean": float(subject_lengths["retained_fraction"].mean()),
        "top_truncated_subjects": top_truncated[
            ["dataset_id", "subject_id", "split", "label", "n_events", "dropped_events", "retained_fraction"]
        ].to_dict(orient="records"),
    }


def first_non_null(values: pd.Series):
    non_null = values.dropna()
    return non_null.iloc[0] if not non_null.empty else pd.NA


def subject_key(row: pd.Series) -> str:
    subject_id = str(row["subject_id"]).strip()
    if subject_id.isdigit():
        subject_id = subject_id.zfill(3)
    return f"{row['dataset_id']}::{subject_id}"


if __name__ == "__main__":
    main()
