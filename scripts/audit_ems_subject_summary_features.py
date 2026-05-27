from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ID_COLUMNS = {"subject_id", "label"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit EMS subject-level summary features.")
    parser.add_argument("--summary", default="data/processed/EMS/ems_subject_summary_features.csv")
    parser.add_argument("--output", default="data/processed/EMS/ems_subject_summary_features_audit.csv")
    parser.add_argument("--high-missing-threshold", type=float, default=0.4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = pd.read_csv(args.summary, dtype={"subject_id": str})
    if "subject_id" not in summary.columns:
        raise ValueError("Summary feature table must contain subject_id.")

    summary["subject_id"] = summary["subject_id"].astype(str).str.zfill(3)
    feature_cols = [col for col in summary.columns if col not in ID_COLUMNS]
    numeric = summary[feature_cols].apply(pd.to_numeric, errors="coerce")

    audit_rows = []
    for col in feature_cols:
        series = numeric[col]
        finite = np.isfinite(series.to_numpy(dtype=float, na_value=np.nan))
        non_missing = int(series.notna().sum())
        audit_rows.append(
            {
                "feature": col,
                "missing_count": int(series.isna().sum()),
                "missing_rate": float(series.isna().mean()),
                "non_missing_count": non_missing,
                "finite_count": int(finite.sum()),
                "n_unique_non_missing": int(series.nunique(dropna=True)),
                "mean": float(series.mean()) if non_missing else np.nan,
                "std": float(series.std()) if non_missing else np.nan,
                "min": float(series.min()) if non_missing else np.nan,
                "max": float(series.max()) if non_missing else np.nan,
                "all_missing": bool(non_missing == 0),
                "constant_non_missing": bool(series.nunique(dropna=True) <= 1),
                "high_missing": bool(series.isna().mean() > args.high_missing_threshold),
            }
        )

    audit = pd.DataFrame(audit_rows).sort_values(
        ["all_missing", "high_missing", "constant_non_missing", "missing_rate"],
        ascending=[False, False, False, False],
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output, index=False, encoding="utf-8-sig")

    labeled = summary["label"].notna() if "label" in summary.columns else pd.Series(False, index=summary.index)
    label_counts = summary.loc[labeled, "label"].value_counts(dropna=False).sort_index()

    print(f"summary: {args.summary}")
    print(f"subjects: {summary['subject_id'].nunique()}")
    print(f"rows: {len(summary)}")
    print(f"labeled_subjects: {int(labeled.sum())}")
    if not label_counts.empty:
        print("label_counts:")
        for label, count in label_counts.items():
            print(f"  {label}: {count}")
    print(f"feature_columns: {len(feature_cols)}")
    print(f"all_missing_columns: {int(audit['all_missing'].sum())}")
    print(f"constant_columns: {int(audit['constant_non_missing'].sum())}")
    print(f"high_missing_columns: {int(audit['high_missing'].sum())}")
    print(f"wrote: {output}")

    problem_cols = audit[audit["all_missing"] | audit["constant_non_missing"] | audit["high_missing"]]
    if not problem_cols.empty:
        print("\nColumns to inspect/drop first:")
        print(problem_cols[["feature", "missing_rate", "n_unique_non_missing"]].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
