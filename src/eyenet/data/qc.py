from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SubjectQCConfig:
    min_events: int = 100
    min_trials: int = 5
    min_valid_coordinate_rate: float = 0.95
    max_out_of_range_coordinate_rate: float = 0.05
    max_missing_transition_rate: float = 0.25
    max_nonpositive_duration_rate: float = 0.01
    min_median_duration_ms: float = 50.0
    max_median_duration_ms: float = 2000.0
    require_label_for_supervised: bool = True
    require_label_for_self_supervised: bool = False


def build_subject_qc_report(events: pd.DataFrame, cfg: SubjectQCConfig | None = None) -> pd.DataFrame:
    cfg = cfg or SubjectQCConfig()
    data = events.copy()
    data["subject_id"] = data["subject_id"].astype(str)

    for column in [
        "label",
        "x_norm",
        "y_norm",
        "duration_ms",
        "saccade_amplitude_norm",
        "saccade_amplitude_dva",
        "transition_velocity_dva_s_approx",
    ]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    data["coordinate_valid"] = data["x_norm"].notna() & data["y_norm"].notna()
    data["coordinate_in_range"] = (
        data["coordinate_valid"]
        & data["x_norm"].between(0.0, 1.0, inclusive="both")
        & data["y_norm"].between(0.0, 1.0, inclusive="both")
    )
    data["coordinate_out_of_range"] = data["coordinate_valid"] & ~data["coordinate_in_range"]
    data["duration_nonpositive"] = data["duration_ms"].notna() & (data["duration_ms"] <= 0)
    if "saccade_amplitude_norm" in data.columns:
        data["transition_missing"] = data["saccade_amplitude_norm"].isna()
    else:
        data["transition_missing"] = True

    group_cols = ["dataset_id", "subject_id"] if "dataset_id" in data.columns else ["subject_id"]
    rows: list[dict[str, Any]] = []
    for keys, group in data.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        dataset_id = str(keys[0]) if len(group_cols) == 2 else ""
        subject_id = str(keys[-1])
        label_values = group["label"].dropna().unique() if "label" in group.columns else np.array([])
        split_values = sorted(str(value) for value in group["split"].dropna().unique()) if "split" in group.columns else []
        fold_values = sorted(str(value) for value in group["fold"].dropna().unique()) if "fold" in group.columns else []
        n_events = int(len(group))
        n_trials = int(group["trial_id"].nunique()) if "trial_id" in group.columns else 0
        valid_coordinate_rate = float(group["coordinate_in_range"].mean()) if n_events else 0.0
        out_of_range_coordinate_rate = float(group["coordinate_out_of_range"].mean()) if n_events else 0.0
        missing_transition_rate = float(group["transition_missing"].mean()) if n_events else 1.0
        nonpositive_duration_rate = float(group["duration_nonpositive"].mean()) if n_events else 1.0
        median_duration_ms = safe_float(group["duration_ms"].median()) if "duration_ms" in group.columns else np.nan
        mean_duration_ms = safe_float(group["duration_ms"].mean()) if "duration_ms" in group.columns else np.nan
        median_saccade_amplitude_norm = (
            safe_float(group["saccade_amplitude_norm"].median()) if "saccade_amplitude_norm" in group.columns else np.nan
        )
        median_transition_velocity_dva_s = (
            safe_float(group["transition_velocity_dva_s_approx"].median())
            if "transition_velocity_dva_s_approx" in group.columns
            else np.nan
        )

        hard_flags = {
            "qc_low_event_count": n_events < cfg.min_events,
            "qc_low_trial_count": n_trials < cfg.min_trials,
            "qc_low_valid_coordinate_rate": valid_coordinate_rate < cfg.min_valid_coordinate_rate,
            "qc_high_out_of_range_coordinate_rate": out_of_range_coordinate_rate > cfg.max_out_of_range_coordinate_rate,
            "qc_high_missing_transition_rate": missing_transition_rate > cfg.max_missing_transition_rate,
            "qc_high_nonpositive_duration_rate": nonpositive_duration_rate > cfg.max_nonpositive_duration_rate,
            "qc_extreme_median_duration": bool(
                pd.notna(median_duration_ms)
                and (median_duration_ms < cfg.min_median_duration_ms or median_duration_ms > cfg.max_median_duration_ms)
            ),
        }
        hard_qc_pass = not any(hard_flags.values())
        has_label = len(label_values) > 0
        supervised_label_ok = has_label or not cfg.require_label_for_supervised
        self_supervised_label_ok = has_label or not cfg.require_label_for_self_supervised
        usable_for_supervised_training = bool(hard_qc_pass and supervised_label_ok)
        usable_for_self_supervised_pretraining = bool(hard_qc_pass and self_supervised_label_ok)
        reasons = [name for name, flagged in hard_flags.items() if flagged]
        if cfg.require_label_for_supervised and not has_label:
            reasons.append("missing_label_for_supervised")
        if cfg.require_label_for_self_supervised and not has_label:
            reasons.append("missing_label_for_self_supervised")

        rows.append(
            {
                "dataset_id": dataset_id,
                "subject_id": subject_id,
                "label": safe_float(label_values[0]) if has_label else np.nan,
                "has_label": bool(has_label),
                "splits": "|".join(split_values),
                "folds": "|".join(fold_values),
                "n_events": n_events,
                "n_trials": n_trials,
                "valid_coordinate_rate": valid_coordinate_rate,
                "out_of_range_coordinate_rate": out_of_range_coordinate_rate,
                "missing_transition_rate": missing_transition_rate,
                "nonpositive_duration_rate": nonpositive_duration_rate,
                "median_duration_ms": median_duration_ms,
                "mean_duration_ms": mean_duration_ms,
                "median_saccade_amplitude_norm": median_saccade_amplitude_norm,
                "median_transition_velocity_dva_s": median_transition_velocity_dva_s,
                **hard_flags,
                "qc_flag_count": int(sum(hard_flags.values())),
                "hard_qc_pass": bool(hard_qc_pass),
                "usable_for_supervised_training": usable_for_supervised_training,
                "usable_for_self_supervised_pretraining": usable_for_self_supervised_pretraining,
                "qc_reasons": ";".join(reasons),
            }
        )

    return pd.DataFrame(rows).sort_values(["dataset_id", "subject_id"]).reset_index(drop=True)


def summarize_subject_qc(report: pd.DataFrame, cfg: SubjectQCConfig) -> dict[str, Any]:
    flag_cols = [column for column in report.columns if column.startswith("qc_") and column != "qc_reasons"]
    summary: dict[str, Any] = {
        "config": asdict(cfg),
        "n_subjects": int(len(report)),
        "n_labeled_subjects": int(report["has_label"].sum()) if "has_label" in report.columns else 0,
        "n_hard_qc_pass": int(report["hard_qc_pass"].sum()),
        "n_usable_for_supervised_training": int(report["usable_for_supervised_training"].sum()),
        "n_usable_for_self_supervised_pretraining": int(report["usable_for_self_supervised_pretraining"].sum()),
        "flag_counts": {column: int(report[column].sum()) for column in flag_cols if report[column].dtype != object},
    }
    if "label" in report.columns:
        summary["label_counts_supervised_usable"] = {
            str(key): int(value)
            for key, value in report.loc[report["usable_for_supervised_training"], "label"]
            .value_counts(dropna=False)
            .sort_index()
            .items()
        }
    if "dataset_id" in report.columns:
        summary["dataset_counts"] = {
            str(key): int(value) for key, value in report["dataset_id"].value_counts(dropna=False).sort_index().items()
        }
    return summary


def save_subject_qc_outputs(
    output_dir: str | Path,
    report: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path / "subject_qc_report.csv", index=False, encoding="utf-8-sig")
    report.loc[report["usable_for_supervised_training"]].to_csv(
        output_path / "usable_supervised_subjects.csv",
        index=False,
        encoding="utf-8-sig",
    )
    report.loc[report["usable_for_self_supervised_pretraining"]].to_csv(
        output_path / "usable_self_supervised_subjects.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (output_path / "subject_qc_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def safe_float(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    return float(value)
