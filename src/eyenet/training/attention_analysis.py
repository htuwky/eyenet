from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

IDENTIFIER_COLUMNS = {
    "subject_id",
    "split",
    "fold",
    "label",
    "segment_id",
    "segment_index",
}


def load_seed_attention(experiment_dir: str | Path) -> pd.DataFrame:
    root = Path(experiment_dir)
    rows: list[pd.DataFrame] = []
    for seed_dir in sorted(root.glob("seed_*")):
        attention_path = seed_dir / "attention_weights.csv"
        if not attention_path.exists():
            continue
        seed = int(seed_dir.name.replace("seed_", ""))
        attention = pd.read_csv(attention_path, dtype={"subject_id": str})
        attention["seed"] = seed
        rows.append(attention)
    if not rows:
        raise FileNotFoundError(f"No seed attention_weights.csv files found under {root}")
    return pd.concat(rows, ignore_index=True)


def build_ensemble_attention(seed_attention: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        seed_attention.groupby(["subject_id", "label", "segment_index"], as_index=False)
        .agg(
            attention_weight=("attention_weight", "mean"),
            attention_weight_std=("attention_weight", "std"),
            probability=("probability", "mean"),
            probability_std=("probability", "std"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values(["subject_id", "segment_index"])
        .reset_index(drop=True)
    )
    grouped["attention_weight"] = grouped.groupby("subject_id")["attention_weight"].transform(
        lambda x: x / x.sum() if x.sum() > 0 else x
    )
    return grouped


def merge_attention_with_segment_features(
    ensemble_attention: pd.DataFrame,
    segment_features: pd.DataFrame,
) -> pd.DataFrame:
    features = segment_features.copy()
    features["subject_id"] = features["subject_id"].astype(str).str.zfill(3)
    features = features[features["split"] == "train_valid"].copy()
    merged = features.merge(
        ensemble_attention,
        on=["subject_id", "label", "segment_index"],
        how="inner",
        suffixes=("", "_attention"),
    )
    if merged.empty:
        raise ValueError("Attention and segment feature tables did not merge. Check subject_id and segment_index.")
    return merged.sort_values(["subject_id", "segment_index"]).reset_index(drop=True)


def add_prediction_columns(
    merged: pd.DataFrame,
    predictions: pd.DataFrame | None,
    threshold: float,
) -> pd.DataFrame:
    data = merged.copy()
    if predictions is None:
        data["prediction"] = (data["probability"] >= threshold).astype(int)
        data["is_correct"] = data["prediction"] == data["label"].astype(int)
        data["error_type"] = _error_type(data["label"].astype(int), data["prediction"])
        return data

    pred = predictions.copy()
    pred["subject_id"] = pred["subject_id"].astype(str).str.zfill(3)
    pred["prediction"] = (pred["probability"] >= threshold).astype(int)
    pred = pred[["subject_id", "probability", "probability_std", "prediction"]].rename(
        columns={
            "probability": "ensemble_probability",
            "probability_std": "ensemble_probability_std",
        }
    )
    data = data.merge(pred, on="subject_id", how="left")
    data["prediction"] = data["prediction"].fillna((data["probability"] >= threshold).astype(int)).astype(int)
    data["ensemble_probability"] = data["ensemble_probability"].fillna(data["probability"])
    data["is_correct"] = data["prediction"] == data["label"].astype(int)
    data["error_type"] = _error_type(data["label"].astype(int), data["prediction"])
    return data


def build_subject_attention_summary(merged: pd.DataFrame, top_fraction: float = 0.10) -> pd.DataFrame:
    rows: list[dict] = []
    eps = 1e-12
    for subject_id, group in merged.groupby("subject_id", sort=True):
        weights = group["attention_weight"].to_numpy(dtype=float)
        weights = weights / weights.sum() if weights.sum() > 0 else weights
        n_segments = len(group)
        top_k = max(1, int(np.ceil(n_segments * top_fraction)))
        sorted_weights = np.sort(weights)[::-1]
        entropy = float(-(weights * np.log(weights + eps)).sum())
        rows.append(
            {
                "subject_id": subject_id,
                "fold": group["fold"].iloc[0],
                "label": int(group["label"].iloc[0]),
                "prediction": int(group["prediction"].iloc[0]),
                "is_correct": bool(group["is_correct"].iloc[0]),
                "error_type": group["error_type"].iloc[0],
                "probability": float(group.get("ensemble_probability", group["probability"]).iloc[0]),
                "n_segments": int(n_segments),
                "attention_entropy": entropy,
                "attention_entropy_norm": float(entropy / np.log(n_segments)) if n_segments > 1 else 0.0,
                "attention_effective_segments": float(1.0 / np.square(weights).sum()) if weights.sum() > 0 else 0.0,
                "attention_max": float(weights.max()) if n_segments else 0.0,
                "attention_top10_mass": float(sorted_weights[:top_k].sum()),
            }
        )
    return pd.DataFrame(rows)


def mark_top_attention_segments(merged: pd.DataFrame, top_fraction: float = 0.10) -> pd.DataFrame:
    data = merged.copy()
    data["attention_rank_pct"] = data.groupby("subject_id")["attention_weight"].rank(pct=True, method="first")
    data["attention_group"] = np.where(data["attention_rank_pct"] >= 1.0 - top_fraction, "top", "rest")
    return data


def feature_columns(data: pd.DataFrame) -> list[str]:
    excluded = IDENTIFIER_COLUMNS | {
        "attention_weight",
        "attention_weight_std",
        "probability",
        "probability_std",
        "n_seeds",
        "prediction",
        "is_correct",
        "error_type",
        "ensemble_probability",
        "ensemble_probability_std",
        "attention_rank_pct",
        "attention_group",
    }
    return [
        col
        for col in data.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(data[col])
    ]


def build_attention_feature_contrast(marked: pd.DataFrame) -> pd.DataFrame:
    cols = feature_columns(marked)
    rows: list[dict] = []
    top = marked[marked["attention_group"] == "top"]
    rest = marked[marked["attention_group"] == "rest"]
    for col in cols:
        top_values = top[col].dropna().to_numpy(dtype=float)
        rest_values = rest[col].dropna().to_numpy(dtype=float)
        if len(top_values) == 0 or len(rest_values) == 0:
            continue
        pooled_std = _pooled_std(top_values, rest_values)
        difference = float(np.mean(top_values) - np.mean(rest_values))
        rows.append(
            {
                "feature": col,
                "rest_attention_mean": float(np.mean(rest_values)),
                "top_attention_mean": float(np.mean(top_values)),
                "difference_top_minus_rest": difference,
                "standardized_difference": float(difference / pooled_std) if pooled_std > 0 else 0.0,
                "abs_standardized_difference": float(abs(difference / pooled_std)) if pooled_std > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("abs_standardized_difference", ascending=False).reset_index(drop=True)


def build_attention_group_comparison(marked: pd.DataFrame) -> pd.DataFrame:
    cols = feature_columns(marked)
    top = marked[marked["attention_group"] == "top"]
    rows: list[dict] = []
    for col in cols:
        hc = top[top["label"].astype(int) == 0][col].dropna().to_numpy(dtype=float)
        sz = top[top["label"].astype(int) == 1][col].dropna().to_numpy(dtype=float)
        if len(hc) == 0 or len(sz) == 0:
            continue
        pooled_std = _pooled_std(sz, hc)
        difference = float(np.mean(sz) - np.mean(hc))
        rows.append(
            {
                "feature": col,
                "hc_top_attention_mean": float(np.mean(hc)),
                "sz_top_attention_mean": float(np.mean(sz)),
                "difference_sz_minus_hc": difference,
                "standardized_difference": float(difference / pooled_std) if pooled_std > 0 else 0.0,
                "abs_standardized_difference": float(abs(difference / pooled_std)) if pooled_std > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("abs_standardized_difference", ascending=False).reset_index(drop=True)


def build_top_segments_table(marked: pd.DataFrame, top_n_per_subject: int = 5) -> pd.DataFrame:
    cols = [
        "subject_id",
        "fold",
        "label",
        "prediction",
        "is_correct",
        "error_type",
        "segment_index",
        "segment_id",
        "attention_weight",
        "n_fixations",
        "fix_duration_ms_mean",
        "saccade_amp_dva_mean",
        "transition_velocity_dva_s_mean",
        "scanpath_length_dva",
        "spatial_coverage_8x8",
        "transition_angle_entropy_8bin",
        "bcea_norm",
    ]
    available_cols = [col for col in cols if col in marked.columns]
    return (
        marked.sort_values(["subject_id", "attention_weight"], ascending=[True, False])
        .groupby("subject_id", as_index=False)
        .head(top_n_per_subject)[available_cols]
        .reset_index(drop=True)
    )


def build_subject_attention_group_summary(subject_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for group_col in ["label", "error_type", "is_correct"]:
        for group_value, group in subject_summary.groupby(group_col, dropna=False, sort=True):
            rows.append(
                {
                    "group_variable": group_col,
                    "group_value": group_value,
                    "n_subjects": int(len(group)),
                    "probability_mean": float(group["probability"].mean()),
                    "attention_entropy_norm_mean": float(group["attention_entropy_norm"].mean()),
                    "attention_effective_segments_mean": float(group["attention_effective_segments"].mean()),
                    "attention_max_mean": float(group["attention_max"].mean()),
                    "attention_top10_mass_mean": float(group["attention_top10_mass"].mean()),
                }
            )
    return pd.DataFrame(rows)


def save_attention_analysis_outputs(
    output_dir: str | Path,
    ensemble_attention: pd.DataFrame,
    merged_attention_features: pd.DataFrame,
    subject_summary: pd.DataFrame,
    subject_group_summary: pd.DataFrame,
    feature_contrast: pd.DataFrame,
    group_comparison: pd.DataFrame,
    top_segments: pd.DataFrame,
) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    ensemble_attention.to_csv(root / "attention_ensemble.csv", index=False, encoding="utf-8-sig")
    merged_attention_features.to_csv(root / "attention_segment_features.csv", index=False, encoding="utf-8-sig")
    subject_summary.to_csv(root / "attention_subject_summary.csv", index=False, encoding="utf-8-sig")
    subject_group_summary.to_csv(root / "attention_subject_group_summary.csv", index=False, encoding="utf-8-sig")
    feature_contrast.to_csv(root / "attention_feature_contrast.csv", index=False, encoding="utf-8-sig")
    group_comparison.to_csv(root / "attention_group_comparison.csv", index=False, encoding="utf-8-sig")
    top_segments.to_csv(root / "attention_top_segments.csv", index=False, encoding="utf-8-sig")


def _error_type(label: pd.Series, prediction: pd.Series) -> np.ndarray:
    return np.select(
        [
            (label == 0) & (prediction == 1),
            (label == 1) & (prediction == 0),
        ],
        ["false_positive", "false_negative"],
        default="correct",
    )


def _pooled_std(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    numerator = (len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)
    denominator = len(a) + len(b) - 2
    return float(np.sqrt(numerator / denominator)) if denominator > 0 else 0.0
