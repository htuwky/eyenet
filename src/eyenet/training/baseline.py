from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: object
    scale: bool = True


def default_model_specs(random_seed: int = 42) -> list[ModelSpec]:
    return [
        ModelSpec(
            "logistic_regression",
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_seed),
            scale=True,
        ),
        ModelSpec(
            "svm_rbf",
            SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=random_seed),
            scale=True,
        ),
        ModelSpec(
            "random_forest",
            RandomForestClassifier(
                n_estimators=500,
                class_weight="balanced",
                random_state=random_seed,
                n_jobs=-1,
            ),
            scale=False,
        ),
        ModelSpec(
            "hist_gradient_boosting",
            HistGradientBoostingClassifier(random_state=random_seed),
            scale=False,
        ),
        ModelSpec(
            "mlp",
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                alpha=0.001,
                max_iter=1000,
                early_stopping=True,
                random_state=random_seed,
            ),
            scale=True,
        ),
    ]


def make_pipeline(spec: ModelSpec) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if spec.scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", spec.estimator))
    return Pipeline(steps)


def run_official_fold_baseline(features: pd.DataFrame, random_seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = [col for col in features.columns if col not in {"subject_id", "fold", "label"}]
    rows: list[dict] = []
    prediction_rows: list[dict] = []

    for spec in default_model_specs(random_seed=random_seed):
        for fold in sorted(features["fold"].unique()):
            train_df = features[features["fold"] != fold].copy()
            valid_df = features[features["fold"] == fold].copy()
            model = make_pipeline(spec)
            model.fit(train_df[feature_cols], train_df["label"])
            prob = predict_positive_probability(model, valid_df[feature_cols])
            pred = (prob >= 0.5).astype(int)
            metrics = compute_metrics(valid_df["label"].to_numpy(), pred, prob)
            rows.append({"model": spec.name, "fold": fold, **metrics})

            for subject_id, y_true, y_prob, y_pred in zip(
                valid_df["subject_id"], valid_df["label"], prob, pred, strict=False
            ):
                prediction_rows.append(
                    {
                        "model": spec.name,
                        "fold": fold,
                        "subject_id": subject_id,
                        "label": int(y_true),
                        "probability": float(y_prob),
                        "prediction": int(y_pred),
                    }
                )

    fold_metrics = pd.DataFrame(rows)
    predictions = pd.DataFrame(prediction_rows)
    return fold_metrics, predictions


def predict_positive_probability(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model[-1], "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model[-1], "decision_function"):
        scores = model.decision_function(x)
        return 1.0 / (1.0 + np.exp(-scores))
    return model.predict(x).astype(float)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def summarize_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["auc", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1"]
    rows: list[dict] = []
    for model, group in fold_metrics.groupby("model"):
        row = {"model": model}
        for metric in metric_cols:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("auc_mean", ascending=False).reset_index(drop=True)


def save_baseline_outputs(
    output_dir: str | Path,
    fold_metrics: pd.DataFrame,
    summary_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    fold_metrics.to_csv(output_path / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    summary_metrics.to_csv(output_path / "summary_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(output_path / "predictions.csv", index=False, encoding="utf-8-sig")
