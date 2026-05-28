from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_PROFILE_FEATURES = [
    "n_trials",
    "n_fixations",
    "fixations_per_trial_mean",
    "fixations_per_trial_std",
    "fix_duration_ms_median",
    "fix_duration_ms_std",
    "fix_duration_ms_max",
    "long_fixation_ratio_gt500ms",
    "short_fixation_ratio_lt150ms",
    "scanpath_length_norm_per_trial_mean",
    "transition_velocity_dva_s_median",
    "transition_velocity_dva_s_mean",
    "saccade_amp_dva_mean",
    "spatial_coverage_8x8",
    "spatial_coverage_16x16",
    "transition_angle_entropy_8bin",
]


def build_subject_prediction_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    data = predictions.copy()
    data["is_error"] = data["label"] != data["prediction"]
    data["false_positive"] = ((data["label"] == 0) & (data["prediction"] == 1)).astype(int)
    data["false_negative"] = ((data["label"] == 1) & (data["prediction"] == 0)).astype(int)
    summary = (
        data.groupby(["subject_id", "label", "fold"], as_index=False)
        .agg(
            n_models=("model", "nunique"),
            n_errors=("is_error", "sum"),
            n_false_positive=("false_positive", "sum"),
            n_false_negative=("false_negative", "sum"),
            mean_probability=("probability", "mean"),
            min_probability=("probability", "min"),
            max_probability=("probability", "max"),
        )
        .sort_values(["n_errors", "subject_id"], ascending=[False, True])
        .reset_index(drop=True)
    )
    summary["error_rate"] = summary["n_errors"] / summary["n_models"]
    summary["error_group"] = np.select(
        [
            summary["n_false_positive"] > 0,
            summary["n_false_negative"] > 0,
        ],
        ["false_positive", "false_negative"],
        default="correct",
    )
    summary["stable_error"] = summary["n_errors"] == summary["n_models"]
    return summary


def build_misclassified_profiles(
    features: pd.DataFrame,
    prediction_summary: pd.DataFrame,
    profile_features: list[str] | None = None,
) -> pd.DataFrame:
    profile_features = profile_features or DEFAULT_PROFILE_FEATURES
    cols = ["subject_id", "fold", "label"] + [col for col in profile_features if col in features.columns]
    merged = prediction_summary.merge(features[cols], on=["subject_id", "fold", "label"], how="left")
    return merged.sort_values(["n_errors", "subject_id"], ascending=[False, True]).reset_index(drop=True)


def add_qc_flags(profiles: pd.DataFrame) -> pd.DataFrame:
    out = profiles.copy()
    out["qc_low_trial_count"] = out["n_trials"] < 95
    out["qc_low_fixation_count"] = robust_low_flag(out["n_fixations"])
    out["qc_high_fixation_count"] = robust_high_flag(out["n_fixations"])
    out["qc_extreme_fix_duration_median"] = robust_extreme_flag(out["fix_duration_ms_median"])
    out["qc_extreme_scanpath"] = robust_extreme_flag(out["scanpath_length_norm_per_trial_mean"])
    out["qc_extreme_transition_velocity"] = robust_extreme_flag(out["transition_velocity_dva_s_median"])
    flag_cols = [col for col in out.columns if col.startswith("qc_")]
    out["qc_flag_count"] = out[flag_cols].sum(axis=1)
    return out


def robust_low_flag(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return series < (q1 - multiplier * iqr)


def robust_high_flag(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return series > (q3 + multiplier * iqr)


def robust_extreme_flag(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    return robust_low_flag(series, multiplier=multiplier) | robust_high_flag(series, multiplier=multiplier)


def build_qc_flag_summary(profiles: pd.DataFrame) -> pd.DataFrame:
    flag_cols = [col for col in profiles.columns if col.startswith("qc_") and col != "qc_flag_count"]
    rows: list[dict] = []
    for group, group_df in profiles.groupby("error_group"):
        row = {"error_group": group, "n_subjects": int(len(group_df))}
        for col in flag_cols:
            row[col] = int(group_df[col].sum())
            row[f"{col}_rate"] = float(group_df[col].mean())
        row["mean_qc_flag_count"] = float(group_df["qc_flag_count"].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("error_group").reset_index(drop=True)


def build_error_group_feature_differences(
    profiles: pd.DataFrame,
    features: list[str] | None = None,
) -> pd.DataFrame:
    features = features or DEFAULT_PROFILE_FEATURES
    available_features = [feature for feature in features if feature in profiles.columns]
    rows: list[dict] = []
    correct = profiles[profiles["error_group"] == "correct"]
    for group_name in ["false_positive", "false_negative"]:
        group = profiles[profiles["error_group"] == group_name]
        if group.empty:
            continue
        for feature in available_features:
            rows.append(
                {
                    "error_group": group_name,
                    "feature": feature,
                    "group_mean": float(group[feature].mean()),
                    "correct_mean": float(correct[feature].mean()),
                    "mean_difference": float(group[feature].mean() - correct[feature].mean()),
                    "group_median": float(group[feature].median()),
                    "correct_median": float(correct[feature].median()),
                    "median_difference": float(group[feature].median() - correct[feature].median()),
                    "group_n": int(len(group)),
                    "correct_n": int(len(correct)),
                }
            )
    return pd.DataFrame(rows).sort_values(["error_group", "feature"]).reset_index(drop=True)


def build_fp_fn_summary(profiles: pd.DataFrame) -> pd.DataFrame:
    return (
        profiles.groupby(["fold", "error_group"], as_index=False)
        .agg(
            n_subjects=("subject_id", "nunique"),
            mean_probability=("mean_probability", "mean"),
            mean_qc_flag_count=("qc_flag_count", "mean"),
        )
        .sort_values(["fold", "error_group"])
        .reset_index(drop=True)
    )


def save_qc_error_outputs(
    output_dir: str | Path,
    profiles: pd.DataFrame,
    feature_differences: pd.DataFrame,
    qc_flag_summary: pd.DataFrame,
    fp_fn_summary: pd.DataFrame,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    profiles.to_csv(output_path / "misclassified_subject_profiles.csv", index=False, encoding="utf-8-sig")
    feature_differences.to_csv(output_path / "error_group_feature_differences.csv", index=False, encoding="utf-8-sig")
    qc_flag_summary.to_csv(output_path / "qc_flag_summary.csv", index=False, encoding="utf-8-sig")
    fp_fn_summary.to_csv(output_path / "fp_fn_summary.csv", index=False, encoding="utf-8-sig")
