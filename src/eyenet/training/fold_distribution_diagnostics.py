from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ID_COLUMNS = {"subject_id", "fold", "label"}


def load_inputs(
    macro_features_path: str | Path,
    event_temporal_path: str | Path,
    macro_predictions_path: str | Path | None = None,
    event_temporal_predictions_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    macro = pd.read_csv(macro_features_path, dtype={"subject_id": str})
    event = pd.read_csv(event_temporal_path, dtype={"subject_id": str})
    macro_pred = (
        pd.read_csv(macro_predictions_path, dtype={"subject_id": str}) if macro_predictions_path else None
    )
    event_pred = (
        pd.read_csv(event_temporal_predictions_path, dtype={"subject_id": str})
        if event_temporal_predictions_path
        else None
    )
    for df in [macro, event, macro_pred, event_pred]:
        if df is not None:
            df["subject_id"] = df["subject_id"].astype(str).str.zfill(3)
    return macro, event, macro_pred, event_pred


def build_fold_label_distribution(subject_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for fold, group in subject_table.groupby("fold", sort=True):
        n_hc = int((group["label"] == 0).sum())
        n_sz = int((group["label"] == 1).sum())
        rows.append(
            {
                "fold": fold,
                "n_subjects": int(len(group)),
                "n_hc": n_hc,
                "n_sz": n_sz,
                "sz_rate": n_sz / len(group) if len(group) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_event_temporal_subject_summary(events: pd.DataFrame) -> pd.DataFrame:
    features = [
        "duration_ms",
        "log_duration_ms",
        "saccade_amplitude_norm",
        "saccade_amplitude_dva",
        "transition_velocity_dva_s_approx",
        "log_transition_velocity_dva_s",
        "transition_missing",
        "x_norm",
        "y_norm",
        "x_dva",
        "y_dva",
    ]
    data = events.copy()
    grouped = data.groupby(["subject_id", "fold", "label"], as_index=False)
    summary = grouped.agg(
        n_events=("subject_event_index", "size"),
        n_segments=("segment_index", "nunique"),
        events_per_segment_mean=("event_index_in_segment", "mean"),
        **{f"event_{feature}_mean": (feature, "mean") for feature in features},
        **{f"event_{feature}_std": (feature, "std") for feature in features},
        **{f"event_{feature}_median": (feature, "median") for feature in features},
    )
    summary["label"] = summary["label"].astype(int)
    return summary


def build_fold_feature_shift(features: pd.DataFrame, table_name: str, top_n: int = 30) -> pd.DataFrame:
    numeric_features = [
        col
        for col in features.columns
        if col not in ID_COLUMNS and pd.api.types.is_numeric_dtype(features[col])
    ]
    rows: list[dict] = []
    for fold in sorted(features["fold"].unique()):
        fold_df = features[features["fold"] == fold]
        other_df = features[features["fold"] != fold]
        for feature in numeric_features:
            a = fold_df[feature].dropna().to_numpy(dtype=float)
            b = other_df[feature].dropna().to_numpy(dtype=float)
            if len(a) < 2 or len(b) < 2:
                continue
            pooled = pooled_std(a, b)
            diff = float(np.mean(a) - np.mean(b))
            rows.append(
                {
                    "table": table_name,
                    "fold": fold,
                    "feature": feature,
                    "fold_mean": float(np.mean(a)),
                    "others_mean": float(np.mean(b)),
                    "difference_fold_minus_others": diff,
                    "standardized_difference": float(diff / pooled) if pooled > 0 else 0.0,
                    "abs_standardized_difference": float(abs(diff / pooled)) if pooled > 0 else 0.0,
                }
            )
    result = pd.DataFrame(rows).sort_values(["fold", "abs_standardized_difference"], ascending=[True, False])
    return result.groupby("fold", as_index=False).head(top_n).reset_index(drop=True)


def summarize_fold_shift(fold_shift: pd.DataFrame) -> pd.DataFrame:
    return (
        fold_shift.groupby(["table", "fold"], as_index=False)
        .agg(
            mean_abs_standardized_difference=("abs_standardized_difference", "mean"),
            max_abs_standardized_difference=("abs_standardized_difference", "max"),
            n_features_reported=("feature", "nunique"),
        )
        .sort_values(["table", "mean_abs_standardized_difference"], ascending=[True, False])
        .reset_index(drop=True)
    )


def build_set1_vs_others_feature_shift(features: pd.DataFrame, table_name: str, top_n: int = 50) -> pd.DataFrame:
    if "Set_1" not in set(features["fold"]):
        raise ValueError("Set_1 is not present in the feature table.")
    numeric_features = [
        col
        for col in features.columns
        if col not in ID_COLUMNS and pd.api.types.is_numeric_dtype(features[col])
    ]
    set1 = features[features["fold"] == "Set_1"]
    others = features[features["fold"] != "Set_1"]
    rows: list[dict] = []
    for feature in numeric_features:
        a = set1[feature].dropna().to_numpy(dtype=float)
        b = others[feature].dropna().to_numpy(dtype=float)
        if len(a) < 2 or len(b) < 2:
            continue
        pooled = pooled_std(a, b)
        diff = float(np.mean(a) - np.mean(b))
        rows.append(
            {
                "table": table_name,
                "feature": feature,
                "set1_mean": float(np.mean(a)),
                "others_mean": float(np.mean(b)),
                "difference_set1_minus_others": diff,
                "standardized_difference": float(diff / pooled) if pooled > 0 else 0.0,
                "abs_standardized_difference": float(abs(diff / pooled)) if pooled > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("abs_standardized_difference", ascending=False).head(top_n).reset_index(drop=True)


def build_prediction_shift_joined(
    subject_table: pd.DataFrame,
    macro_predictions: pd.DataFrame | None,
    event_predictions: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = subject_table[["subject_id", "fold", "label"]].copy()
    if macro_predictions is not None:
        macro = macro_predictions.rename(
            columns={"probability": "macro_probability", "prediction": "macro_prediction"}
        )
        data = data.merge(
            macro[["subject_id", "macro_probability", "macro_prediction"]],
            on="subject_id",
            how="left",
        )
    if event_predictions is not None:
        event = event_predictions.rename(
            columns={"probability": "event_temporal_probability", "prediction": "event_temporal_prediction"}
        )
        data = data.merge(
            event[["subject_id", "event_temporal_probability", "event_temporal_prediction"]],
            on="subject_id",
            how="left",
        )
    data["probability_gap_event_minus_macro"] = (
        data["event_temporal_probability"] - data["macro_probability"]
        if {"event_temporal_probability", "macro_probability"}.issubset(data.columns)
        else np.nan
    )

    agg_spec = {
        "n_subjects": ("subject_id", "nunique"),
    }
    for col in ["macro_probability", "event_temporal_probability", "probability_gap_event_minus_macro"]:
        if col in data.columns:
            agg_spec[f"{col}_mean"] = (col, "mean")
            agg_spec[f"{col}_std"] = (col, "std")
    summary = data.groupby("fold", as_index=False).agg(**agg_spec)
    return data, summary


def save_fold_distribution_diagnostics(
    output_dir: str | Path,
    fold_label_distribution: pd.DataFrame,
    macro_fold_shift: pd.DataFrame,
    event_fold_shift: pd.DataFrame,
    fold_shift_summary: pd.DataFrame,
    set1_macro_shift: pd.DataFrame,
    set1_event_shift: pd.DataFrame,
    prediction_shift_joined: pd.DataFrame,
    prediction_shift_summary: pd.DataFrame,
    event_subject_summary: pd.DataFrame,
) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    fold_label_distribution.to_csv(root / "fold_label_distribution.csv", index=False, encoding="utf-8-sig")
    macro_fold_shift.to_csv(root / "fold_feature_shift_macro.csv", index=False, encoding="utf-8-sig")
    event_fold_shift.to_csv(root / "fold_feature_shift_event_temporal.csv", index=False, encoding="utf-8-sig")
    fold_shift_summary.to_csv(root / "fold_shift_summary.csv", index=False, encoding="utf-8-sig")
    set1_macro_shift.to_csv(root / "set1_vs_others_feature_shift_macro.csv", index=False, encoding="utf-8-sig")
    set1_event_shift.to_csv(root / "set1_vs_others_feature_shift_event_temporal.csv", index=False, encoding="utf-8-sig")
    prediction_shift_joined.to_csv(root / "fold_prediction_shift_joined.csv", index=False, encoding="utf-8-sig")
    prediction_shift_summary.to_csv(root / "fold_prediction_shift_summary.csv", index=False, encoding="utf-8-sig")
    event_subject_summary.to_csv(root / "event_temporal_subject_summary.csv", index=False, encoding="utf-8-sig")


def pooled_std(a: np.ndarray, b: np.ndarray) -> float:
    numerator = (len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)
    denominator = len(a) + len(b) - 2
    if denominator <= 0:
        return 0.0
    return float(np.sqrt(numerator / denominator))
