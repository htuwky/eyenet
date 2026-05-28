from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind

DEFAULT_FEATURES = [
    "scanpath_length_dva",
    "scanpath_length_norm",
    "n_fixations",
    "spatial_coverage_8x8",
    "spatial_bbox_area",
    "fix_duration_ms_mean",
    "fix_duration_ms_median",
    "long_fixation_ratio_gt500ms",
    "saccade_amp_dva_mean",
    "saccade_amp_norm_mean",
    "transition_velocity_dva_s_mean",
    "transition_velocity_dva_s_median",
    "bcea_norm",
]


def build_subject_level_feature_table(
    attention_segment_features: pd.DataFrame,
    features: list[str] | None = None,
    attention_group: str = "top",
) -> pd.DataFrame:
    features = features or DEFAULT_FEATURES
    available_features = [col for col in features if col in attention_segment_features.columns]
    if not available_features:
        raise ValueError("None of the requested features exist in the attention segment feature table.")

    data = attention_segment_features.copy()
    data["subject_id"] = data["subject_id"].astype(str).str.zfill(3)
    if attention_group != "all":
        data = data[data["attention_group"] == attention_group].copy()
    if data.empty:
        raise ValueError(f"No rows found for attention_group='{attention_group}'.")

    subject_table = (
        data.groupby(["subject_id", "fold", "label", "prediction", "is_correct", "error_type"], as_index=False)[
            available_features
        ]
        .mean()
        .sort_values("subject_id")
        .reset_index(drop=True)
    )
    subject_table["label"] = subject_table["label"].astype(int)
    return subject_table


def compare_groups(
    subject_feature_table: pd.DataFrame,
    features: list[str] | None = None,
    positive_label: int = 1,
    negative_label: int = 0,
    n_bootstrap: int = 5000,
    random_seed: int = 42,
) -> pd.DataFrame:
    features = features or DEFAULT_FEATURES
    available_features = [col for col in features if col in subject_feature_table.columns]
    rng = np.random.default_rng(random_seed)
    rows: list[dict] = []

    for feature in available_features:
        positive = subject_feature_table.loc[
            subject_feature_table["label"] == positive_label, feature
        ].dropna().to_numpy(dtype=float)
        negative = subject_feature_table.loc[
            subject_feature_table["label"] == negative_label, feature
        ].dropna().to_numpy(dtype=float)
        if len(positive) < 2 or len(negative) < 2:
            continue

        diff = float(np.mean(positive) - np.mean(negative))
        ci_low, ci_high = bootstrap_mean_difference_ci(
            positive,
            negative,
            n_bootstrap=n_bootstrap,
            rng=rng,
        )
        _, t_p = ttest_ind(positive, negative, equal_var=False, nan_policy="omit")
        u_stat, u_p = mannwhitneyu(positive, negative, alternative="two-sided")
        d_value = cohen_d(positive, negative)

        rows.append(
            {
                "feature": feature,
                "n_positive": int(len(positive)),
                "n_negative": int(len(negative)),
                "positive_mean": float(np.mean(positive)),
                "negative_mean": float(np.mean(negative)),
                "positive_median": float(np.median(positive)),
                "negative_median": float(np.median(negative)),
                "mean_difference_positive_minus_negative": diff,
                "bootstrap_ci95_low": ci_low,
                "bootstrap_ci95_high": ci_high,
                "cohen_d": d_value,
                "abs_cohen_d": abs(d_value),
                "welch_t_p": float(t_p),
                "mannwhitney_u": float(u_stat),
                "mannwhitney_p": float(u_p),
            }
        )

    results = pd.DataFrame(rows)
    if results.empty:
        return results
    results["mannwhitney_fdr_bh"] = benjamini_hochberg(results["mannwhitney_p"].to_numpy(dtype=float))
    results["welch_t_fdr_bh"] = benjamini_hochberg(results["welch_t_p"].to_numpy(dtype=float))
    return results.sort_values("abs_cohen_d", ascending=False).reset_index(drop=True)


def summarize_error_groups(subject_feature_table: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    features = features or DEFAULT_FEATURES
    available_features = [col for col in features if col in subject_feature_table.columns]
    rows: list[dict] = []
    for error_type, group in subject_feature_table.groupby("error_type", sort=True):
        row = {
            "error_type": error_type,
            "n_subjects": int(len(group)),
            "prediction_rate": float(group["prediction"].mean()) if "prediction" in group else np.nan,
        }
        for feature in available_features:
            row[f"{feature}_mean"] = float(group[feature].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def save_statistical_validation_outputs(
    output_dir: str | Path,
    subject_feature_table: pd.DataFrame,
    group_comparison: pd.DataFrame,
    error_group_summary: pd.DataFrame,
) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    subject_feature_table.to_csv(root / "subject_top_attention_features.csv", index=False, encoding="utf-8-sig")
    group_comparison.to_csv(root / "hc_sz_statistical_tests.csv", index=False, encoding="utf-8-sig")
    error_group_summary.to_csv(root / "error_group_feature_summary.csv", index=False, encoding="utf-8-sig")


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    pooled = pooled_std(a, b)
    if pooled == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def pooled_std(a: np.ndarray, b: np.ndarray) -> float:
    numerator = (len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)
    denominator = len(a) + len(b) - 2
    if denominator <= 0:
        return 0.0
    return float(np.sqrt(numerator / denominator))


def bootstrap_mean_difference_ci(
    positive: np.ndarray,
    negative: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> tuple[float, float]:
    diffs = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        pos_sample = rng.choice(positive, size=len(positive), replace=True)
        neg_sample = rng.choice(negative, size=len(negative), replace=True)
        diffs[i] = np.mean(pos_sample) - np.mean(neg_sample)
    return (
        float(np.quantile(diffs, alpha / 2)),
        float(np.quantile(diffs, 1 - alpha / 2)),
    )


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(n, dtype=float)
    running_min = 1.0
    for rank, index in enumerate(order[::-1], start=1):
        original_rank = n - rank + 1
        value = p_values[index] * n / original_rank
        running_min = min(running_min, value)
        adjusted[index] = min(running_min, 1.0)
    return adjusted
