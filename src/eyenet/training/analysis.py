from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from eyenet.training.baseline import default_model_specs, make_pipeline, summarize_metrics

METRIC_COLUMNS = ["auc", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1"]


def make_paper_table(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    summary = summarize_metrics(fold_metrics)
    rows: list[dict] = []
    for row in summary.to_dict(orient="records"):
        formatted = {"model": row["model"]}
        for metric in METRIC_COLUMNS:
            formatted[metric] = f"{row[f'{metric}_mean']:.3f} ± {row[f'{metric}_std']:.3f}"
        rows.append(formatted)
    return pd.DataFrame(rows)


def build_error_analysis(predictions: pd.DataFrame) -> pd.DataFrame:
    data = predictions.copy()
    data["is_correct"] = data["label"] == data["prediction"]
    data["error_type"] = np.select(
        [
            (data["label"] == 0) & (data["prediction"] == 1),
            (data["label"] == 1) & (data["prediction"] == 0),
        ],
        ["false_positive", "false_negative"],
        default="correct",
    )
    subject_summary = (
        data.groupby(["subject_id", "label"], as_index=False)
        .agg(
            n_models=("model", "nunique"),
            n_predictions=("model", "size"),
            n_errors=("is_correct", lambda x: int((~x).sum())),
            mean_probability=("probability", "mean"),
            max_probability=("probability", "max"),
            min_probability=("probability", "min"),
        )
        .sort_values(["n_errors", "subject_id"], ascending=[False, True])
        .reset_index(drop=True)
    )
    subject_summary["error_rate"] = subject_summary["n_errors"] / subject_summary["n_predictions"]
    return subject_summary


def build_fold_error_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    data = predictions.copy()
    data["is_correct"] = data["label"] == data["prediction"]
    data["false_positive"] = ((data["label"] == 0) & (data["prediction"] == 1)).astype(int)
    data["false_negative"] = ((data["label"] == 1) & (data["prediction"] == 0)).astype(int)
    return (
        data.groupby(["model", "fold"], as_index=False)
        .agg(
            n_subjects=("subject_id", "nunique"),
            n_errors=("is_correct", lambda x: int((~x).sum())),
            false_positives=("false_positive", "sum"),
            false_negatives=("false_negative", "sum"),
            mean_probability=("probability", "mean"),
        )
        .assign(error_rate=lambda df: df["n_errors"] / df["n_subjects"])
        .sort_values(["model", "fold"])
        .reset_index(drop=True)
    )


def compute_permutation_importance_by_fold(
    features: pd.DataFrame,
    model_name: str = "random_forest",
    random_seed: int = 42,
    n_repeats: int = 30,
) -> pd.DataFrame:
    feature_cols = [col for col in features.columns if col not in {"subject_id", "fold", "label"}]
    spec_map = {spec.name: spec for spec in default_model_specs(random_seed=random_seed)}
    if model_name not in spec_map:
        raise ValueError(f"Unknown model '{model_name}'. Available: {sorted(spec_map)}")

    rows: list[dict] = []
    for fold in sorted(features["fold"].unique()):
        train_df = features[features["fold"] != fold].copy()
        valid_df = features[features["fold"] == fold].copy()
        model = make_pipeline(spec_map[model_name])
        model.fit(train_df[feature_cols], train_df["label"])

        result = permutation_importance(
            model,
            valid_df[feature_cols],
            valid_df["label"],
            scoring="roc_auc",
            n_repeats=n_repeats,
            random_state=random_seed,
            n_jobs=-1,
        )
        for feature, mean, std in zip(feature_cols, result.importances_mean, result.importances_std, strict=False):
            rows.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "feature": feature,
                    "importance_mean": float(mean),
                    "importance_std": float(std),
                }
            )
    return pd.DataFrame(rows)


def summarize_importance(importance: pd.DataFrame) -> pd.DataFrame:
    return (
        importance.groupby(["model", "feature"], as_index=False)
        .agg(
            importance_mean=("importance_mean", "mean"),
            importance_std_across_folds=("importance_mean", "std"),
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def save_analysis_outputs(
    output_dir: str | Path,
    baseline_table: pd.DataFrame,
    error_analysis: pd.DataFrame,
    fold_error_summary: pd.DataFrame,
    permutation_importance_df: pd.DataFrame,
    top_features: pd.DataFrame,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    baseline_table.to_csv(output_path / "baseline_table.csv", index=False, encoding="utf-8-sig")
    error_analysis.to_csv(output_path / "error_analysis.csv", index=False, encoding="utf-8-sig")
    fold_error_summary.to_csv(output_path / "fold_error_summary.csv", index=False, encoding="utf-8-sig")
    permutation_importance_df.to_csv(output_path / "permutation_importance.csv", index=False, encoding="utf-8-sig")
    top_features.to_csv(output_path / "top_features.csv", index=False, encoding="utf-8-sig")
