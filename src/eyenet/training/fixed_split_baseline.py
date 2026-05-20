from __future__ import annotations

from pathlib import Path

import pandas as pd

from eyenet.training.baseline import compute_metrics, default_model_specs, make_pipeline, predict_positive_probability
from eyenet.training.thresholds import analyze_thresholds, choose_thresholds


def attach_fixed_split(features: pd.DataFrame, split_path: str | Path) -> pd.DataFrame:
    split = pd.read_csv(split_path, dtype={"subject_id": str})
    split["subject_id"] = split["subject_id"].astype(str).str.zfill(3)
    data = features.copy()
    data["subject_id"] = data["subject_id"].astype(str).str.zfill(3)
    data = data.drop(columns=["split"], errors="ignore")
    data = data.merge(split[["subject_id", "split", "official_fold"]], on="subject_id", how="inner")
    if data["subject_id"].nunique() != split["subject_id"].nunique():
        raise ValueError("Feature table and split file have different subject coverage.")
    return data


def run_fixed_split_baseline(
    features: pd.DataFrame,
    split_path: str | Path,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = attach_fixed_split(features, split_path)
    feature_cols = [
        col
        for col in data.columns
        if col not in {"subject_id", "fold", "official_fold", "split", "label"}
    ]
    train_df = data[data["split"] == "train"].copy()
    valid_df = data[data["split"] == "valid"].copy()
    test_df = data[data["split"] == "test"].copy()
    if train_df.empty or valid_df.empty or test_df.empty:
        raise ValueError("Fixed split must contain train, valid, and test subjects.")

    prediction_rows: list[dict] = []
    threshold_rows: list[pd.DataFrame] = []
    selected_rows: list[pd.DataFrame] = []
    metric_rows: list[dict] = []

    for spec in default_model_specs(random_seed=random_seed):
        model = make_pipeline(spec)
        model.fit(train_df[feature_cols], train_df["label"])

        valid_prob = predict_positive_probability(model, valid_df[feature_cols])
        test_prob = predict_positive_probability(model, test_df[feature_cols])

        valid_predictions = make_prediction_frame(spec.name, "valid", valid_df, valid_prob, threshold=0.5)
        threshold_metrics = analyze_thresholds(valid_predictions)
        threshold_metrics.insert(0, "model", spec.name)
        selected = choose_thresholds(threshold_metrics.drop(columns=["model"]))
        selected.insert(0, "model", spec.name)
        selected_rows.append(selected)
        threshold_rows.append(threshold_metrics)

        best_balanced = selected[selected["criterion"] == "best_balanced_accuracy"].iloc[0]
        best_f1 = selected[selected["criterion"] == "best_f1"].iloc[0]
        screening_candidates = selected[selected["criterion"].str.contains("sensitivity_at_least_0.80", regex=False)]
        screening_threshold = (
            float(screening_candidates.iloc[0]["threshold"])
            if not screening_candidates.empty
            else float(best_balanced["threshold"])
        )
        threshold_map = {
            "default_0.50": 0.5,
            "valid_best_balanced_accuracy": float(best_balanced["threshold"]),
            "valid_best_f1": float(best_f1["threshold"]),
            "valid_screening_sensitivity_at_least_0.80": screening_threshold,
        }

        for split_name, split_df, prob in [("valid", valid_df, valid_prob), ("test", test_df, test_prob)]:
            for threshold_name, threshold in threshold_map.items():
                pred_frame = make_prediction_frame(spec.name, split_name, split_df, prob, threshold=threshold)
                pred_frame["threshold_name"] = threshold_name
                pred_frame["threshold"] = threshold
                prediction_rows.extend(pred_frame.to_dict(orient="records"))
                metrics = compute_metrics(
                    pred_frame["label"].to_numpy(dtype=int),
                    pred_frame["prediction"].to_numpy(dtype=int),
                    pred_frame["probability"].to_numpy(dtype=float),
                )
                metric_rows.append(
                    {
                        "model": spec.name,
                        "split": split_name,
                        "threshold_name": threshold_name,
                        "threshold": threshold,
                        **metrics,
                    }
                )

    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(prediction_rows),
        pd.concat(threshold_rows, ignore_index=True),
        pd.concat(selected_rows, ignore_index=True),
        data[["subject_id", "label", "official_fold", "split"]],
    )


def make_prediction_frame(model_name: str, split_name: str, split_df: pd.DataFrame, prob, threshold: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": model_name,
            "split": split_name,
            "subject_id": split_df["subject_id"].to_numpy(),
            "label": split_df["label"].astype(int).to_numpy(),
            "probability": prob,
            "prediction": (prob >= threshold).astype(int),
        }
    )


def save_fixed_split_baseline_outputs(
    output_dir: str | Path,
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    valid_threshold_metrics: pd.DataFrame,
    selected_thresholds: pd.DataFrame,
    split_subjects: pd.DataFrame,
) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(root / "metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(root / "predictions.csv", index=False, encoding="utf-8-sig")
    valid_threshold_metrics.to_csv(root / "valid_threshold_metrics.csv", index=False, encoding="utf-8-sig")
    selected_thresholds.to_csv(root / "selected_thresholds.csv", index=False, encoding="utf-8-sig")
    split_subjects.to_csv(root / "split_subjects.csv", index=False, encoding="utf-8-sig")
