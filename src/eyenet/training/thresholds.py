from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from eyenet.training.baseline import compute_metrics


def analyze_thresholds(
    predictions: pd.DataFrame,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    thresholds = thresholds if thresholds is not None else np.linspace(0.05, 0.95, 91)
    y_true = predictions["label"].to_numpy(dtype=int)
    y_prob = predictions["probability"].to_numpy(dtype=float)
    rows: list[dict] = []
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        metrics = compute_metrics(y_true, y_pred, y_prob)
        rows.append({"threshold": float(threshold), **metrics})
    return pd.DataFrame(rows)


def choose_thresholds(threshold_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    rows.append(best_by_metric(threshold_metrics, "balanced_accuracy", "best_balanced_accuracy"))
    rows.append(best_by_metric(threshold_metrics, "f1", "best_f1"))
    for target_sensitivity in [0.80, 0.85, 0.90]:
        candidates = threshold_metrics[threshold_metrics["sensitivity"] >= target_sensitivity]
        if candidates.empty:
            continue
        best = candidates.sort_values(["specificity", "balanced_accuracy"], ascending=False).iloc[0].to_dict()
        best["criterion"] = f"sensitivity_at_least_{target_sensitivity:.2f}"
        rows.append(best)
    return pd.DataFrame(rows)


def best_by_metric(threshold_metrics: pd.DataFrame, metric: str, criterion: str) -> dict:
    best = threshold_metrics.sort_values(metric, ascending=False).iloc[0].to_dict()
    best["criterion"] = criterion
    return best


def save_threshold_outputs(
    output_dir: str | Path,
    threshold_metrics: pd.DataFrame,
    selected_thresholds: pd.DataFrame,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    threshold_metrics.to_csv(output_path / "threshold_analysis.csv", index=False, encoding="utf-8-sig")
    selected_thresholds.to_csv(output_path / "selected_thresholds.csv", index=False, encoding="utf-8-sig")
