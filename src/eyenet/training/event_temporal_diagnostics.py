from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from eyenet.training.baseline import compute_metrics
from eyenet.training.thresholds import analyze_thresholds, choose_thresholds


def load_event_temporal_experiment(experiment_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = Path(experiment_dir)
    predictions = pd.read_csv(root / "predictions.csv", dtype={"subject_id": str})
    fold_metrics = pd.read_csv(root / "fold_metrics.csv")
    training_log = pd.read_csv(root / "training_log.csv")
    return predictions, fold_metrics, training_log


def build_fold_probability_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for fold, fold_df in predictions.groupby("fold", sort=True):
        label0 = fold_df[fold_df["label"] == 0]["probability"].to_numpy(dtype=float)
        label1 = fold_df[fold_df["label"] == 1]["probability"].to_numpy(dtype=float)
        rows.append(
            {
                "fold": fold,
                "n_subjects": int(len(fold_df)),
                "n_hc": int((fold_df["label"] == 0).sum()),
                "n_sz": int((fold_df["label"] == 1).sum()),
                "prob_mean": float(fold_df["probability"].mean()),
                "prob_std": float(fold_df["probability"].std()),
                "prob_min": float(fold_df["probability"].min()),
                "prob_p25": float(fold_df["probability"].quantile(0.25)),
                "prob_median": float(fold_df["probability"].median()),
                "prob_p75": float(fold_df["probability"].quantile(0.75)),
                "prob_max": float(fold_df["probability"].max()),
                "hc_prob_mean": float(np.mean(label0)),
                "hc_prob_median": float(np.median(label0)),
                "sz_prob_mean": float(np.mean(label1)),
                "sz_prob_median": float(np.median(label1)),
                "sz_minus_hc_prob_mean": float(np.mean(label1) - np.mean(label0)),
                "fold_auc": float(roc_auc_score(fold_df["label"], fold_df["probability"])),
                "mean_direction_ok": bool(np.mean(label1) > np.mean(label0)),
                "default_positive_rate": float((fold_df["probability"] >= 0.5).mean()),
                "hc_default_positive_rate": float((label0 >= 0.5).mean()),
                "sz_default_positive_rate": float((label1 >= 0.5).mean()),
            }
        )
    return pd.DataFrame(rows)


def build_fold_threshold_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for fold, fold_df in predictions.groupby("fold", sort=True):
        selected = choose_thresholds(analyze_thresholds(fold_df))
        selected.insert(0, "fold", fold)
        rows.append(selected)
    return pd.concat(rows, ignore_index=True)


def build_training_curve_summary(training_log: pd.DataFrame, fold_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    best_by_fold = fold_metrics.set_index("fold")
    for fold, group in training_log.groupby("fold", sort=True):
        group = group.sort_values("epoch")
        best_idx = group["valid_auc"].idxmax()
        best_row = group.loc[best_idx]
        last_row = group.iloc[-1]
        min_loss_row = group.loc[group["valid_loss"].idxmin()]
        best_epoch = int(best_by_fold.loc[fold, "best_epoch"]) if fold in best_by_fold.index else int(best_row["epoch"])
        stopped_epoch = int(best_by_fold.loc[fold, "stopped_epoch"]) if fold in best_by_fold.index else int(last_row["epoch"])
        rows.append(
            {
                "fold": fold,
                "n_epochs_run": int(group["epoch"].max()),
                "best_epoch": best_epoch,
                "stopped_epoch": stopped_epoch,
                "best_epoch_fraction": float(best_epoch / stopped_epoch) if stopped_epoch else np.nan,
                "best_valid_auc": float(best_row["valid_auc"]),
                "last_valid_auc": float(last_row["valid_auc"]),
                "valid_auc_drop_best_to_last": float(best_row["valid_auc"] - last_row["valid_auc"]),
                "best_valid_loss": float(min_loss_row["valid_loss"]),
                "last_valid_loss": float(last_row["valid_loss"]),
                "valid_loss_change_min_to_last": float(last_row["valid_loss"] - min_loss_row["valid_loss"]),
                "train_loss_at_best_auc": float(best_row["train_loss"]),
                "valid_loss_at_best_auc": float(best_row["valid_loss"]),
                "generalization_gap_at_best_auc": float(best_row["valid_loss"] - best_row["train_loss"]),
                "early_best_epoch_flag": bool(best_epoch <= 3),
                "large_auc_drop_flag": bool((best_row["valid_auc"] - last_row["valid_auc"]) >= 0.05),
            }
        )
    return pd.DataFrame(rows)


def build_pooled_auc_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    data = predictions.copy()
    y_true = data["label"].to_numpy(dtype=int)
    raw_prob = data["probability"].to_numpy(dtype=float)

    data["fold_centered_probability"] = data.groupby("fold")["probability"].transform(lambda x: x - x.mean())
    data["fold_z_probability"] = data.groupby("fold")["probability"].transform(
        lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) > 0 else x * 0
    )
    data["fold_rank_probability"] = data.groupby("fold")["probability"].rank(pct=True)

    rows = [
        {
            "score_name": "raw_probability",
            "pooled_auc": float(roc_auc_score(y_true, raw_prob)),
            "interpretation": "Actual pooled AUC from saved probabilities; affected by between-fold probability scale.",
        },
        {
            "score_name": "fold_centered_probability",
            "pooled_auc": float(roc_auc_score(y_true, data["fold_centered_probability"])),
            "interpretation": "AUC after subtracting each fold mean; tests fold offset/calibration effects.",
        },
        {
            "score_name": "fold_z_probability",
            "pooled_auc": float(roc_auc_score(y_true, data["fold_z_probability"])),
            "interpretation": "AUC after fold-wise z-score; tests fold scale effects.",
        },
        {
            "score_name": "fold_rank_probability",
            "pooled_auc": float(roc_auc_score(y_true, data["fold_rank_probability"])),
            "interpretation": "AUC after fold-wise rank normalization; approximates within-fold ordering only.",
        },
    ]
    fold_auc = data.groupby("fold").apply(lambda x: roc_auc_score(x["label"], x["probability"]), include_groups=False)
    rows.append(
        {
            "score_name": "mean_fold_auc",
            "pooled_auc": float(fold_auc.mean()),
            "interpretation": "Mean of official fold AUCs; not a single global threshold metric.",
        }
    )
    return pd.DataFrame(rows)


def build_misclassified_subjects(predictions: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    data = predictions.copy()
    data["prediction_at_threshold"] = (data["probability"] >= threshold).astype(int)
    data["is_correct"] = data["prediction_at_threshold"] == data["label"]
    data["error_type"] = np.select(
        [
            (data["label"] == 0) & (data["prediction_at_threshold"] == 1),
            (data["label"] == 1) & (data["prediction_at_threshold"] == 0),
        ],
        ["false_positive", "false_negative"],
        default="correct",
    )
    return (
        data[~data["is_correct"]]
        .sort_values(["fold", "error_type", "probability"], ascending=[True, True, False])
        .reset_index(drop=True)
    )


def build_default_metric_recheck(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for fold, fold_df in predictions.groupby("fold", sort=True):
        y_true = fold_df["label"].to_numpy(dtype=int)
        y_prob = fold_df["probability"].to_numpy(dtype=float)
        y_pred = (y_prob >= 0.5).astype(int)
        rows.append({"fold": fold, **compute_metrics(y_true, y_pred, y_prob)})
    y_true = predictions["label"].to_numpy(dtype=int)
    y_prob = predictions["probability"].to_numpy(dtype=float)
    y_pred = (y_prob >= 0.5).astype(int)
    rows.append({"fold": "pooled", **compute_metrics(y_true, y_pred, y_prob)})
    return pd.DataFrame(rows)


def save_event_temporal_diagnostics(
    output_dir: str | Path,
    fold_probability_summary: pd.DataFrame,
    fold_threshold_summary: pd.DataFrame,
    training_curve_summary: pd.DataFrame,
    pooled_auc_diagnostics: pd.DataFrame,
    misclassified_subjects: pd.DataFrame,
    default_metric_recheck: pd.DataFrame,
) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    fold_probability_summary.to_csv(root / "fold_probability_summary.csv", index=False, encoding="utf-8-sig")
    fold_threshold_summary.to_csv(root / "fold_threshold_summary.csv", index=False, encoding="utf-8-sig")
    training_curve_summary.to_csv(root / "training_curve_summary.csv", index=False, encoding="utf-8-sig")
    pooled_auc_diagnostics.to_csv(root / "pooled_auc_diagnostics.csv", index=False, encoding="utf-8-sig")
    misclassified_subjects.to_csv(root / "misclassified_subjects.csv", index=False, encoding="utf-8-sig")
    default_metric_recheck.to_csv(root / "default_metric_recheck.csv", index=False, encoding="utf-8-sig")
