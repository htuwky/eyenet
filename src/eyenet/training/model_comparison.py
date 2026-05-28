from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from eyenet.training.baseline import compute_metrics

METRIC_COLUMNS = ["auc", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1"]


def load_named_predictions(
    path: str | Path,
    model_name: str,
    probability_col: str = "probability",
    threshold: float = 0.5,
    model_filter: str | None = None,
) -> pd.DataFrame:
    predictions = pd.read_csv(path, dtype={"subject_id": str})
    if model_filter is not None:
        if "model" not in predictions.columns:
            raise ValueError(f"Cannot filter model='{model_filter}' because {path} has no model column.")
        predictions = predictions[predictions["model"] == model_filter].copy()
    if predictions.empty:
        raise ValueError(f"No predictions found for {model_name} from {path}")
    predictions["subject_id"] = predictions["subject_id"].astype(str).str.zfill(3)
    predictions = predictions.rename(columns={probability_col: "probability"})
    predictions["prediction"] = (predictions["probability"] >= threshold).astype(int)
    return predictions[["subject_id", "fold", "label", "probability", "prediction"]].assign(model=model_name)


def align_prediction_tables(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    reference_name: str,
    candidate_name: str,
) -> pd.DataFrame:
    ref = reference.rename(
        columns={
            "probability": f"{reference_name}_probability",
            "prediction": f"{reference_name}_prediction",
        }
    )[["subject_id", "fold", "label", f"{reference_name}_probability", f"{reference_name}_prediction"]]
    cand = candidate.rename(
        columns={
            "probability": f"{candidate_name}_probability",
            "prediction": f"{candidate_name}_prediction",
        }
    )[["subject_id", f"{candidate_name}_probability", f"{candidate_name}_prediction"]]
    merged = ref.merge(cand, on="subject_id", how="inner")
    if merged.empty:
        raise ValueError("Prediction tables did not merge on subject_id.")
    return merged.sort_values(["fold", "subject_id"]).reset_index(drop=True)


def compute_model_metrics_from_aligned(
    aligned: pd.DataFrame,
    model_name: str,
) -> dict:
    y_true = aligned["label"].to_numpy(dtype=int)
    y_prob = aligned[f"{model_name}_probability"].to_numpy(dtype=float)
    y_pred = aligned[f"{model_name}_prediction"].to_numpy(dtype=int)
    return compute_metrics(y_true, y_pred, y_prob)


def build_model_metric_table(aligned: pd.DataFrame, model_names: list[str]) -> pd.DataFrame:
    rows = []
    for model_name in model_names:
        rows.append({"model": model_name, **compute_model_metrics_from_aligned(aligned, model_name)})
    return pd.DataFrame(rows)


def bootstrap_model_comparison(
    aligned: pd.DataFrame,
    reference_name: str,
    candidate_name: str,
    n_bootstrap: int = 5000,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_seed)
    n = len(aligned)
    model_metric_samples: list[dict] = []
    difference_samples: list[dict] = []

    for sample_index in range(n_bootstrap):
        sampled_indices = rng.integers(0, n, size=n)
        sample = aligned.iloc[sampled_indices].reset_index(drop=True)
        if sample["label"].nunique() < 2:
            continue
        ref_metrics = compute_model_metrics_from_aligned(sample, reference_name)
        cand_metrics = compute_model_metrics_from_aligned(sample, candidate_name)
        for metric in METRIC_COLUMNS:
            model_metric_samples.append(
                {
                    "bootstrap_index": sample_index,
                    "model": reference_name,
                    "metric": metric,
                    "value": ref_metrics[metric],
                }
            )
            model_metric_samples.append(
                {
                    "bootstrap_index": sample_index,
                    "model": candidate_name,
                    "metric": metric,
                    "value": cand_metrics[metric],
                }
            )
            difference_samples.append(
                {
                    "bootstrap_index": sample_index,
                    "metric": metric,
                    "difference_candidate_minus_reference": cand_metrics[metric] - ref_metrics[metric],
                }
            )

    model_ci = summarize_bootstrap_model_metrics(pd.DataFrame(model_metric_samples))
    difference_ci = summarize_bootstrap_differences(pd.DataFrame(difference_samples))
    return model_ci, difference_ci


def summarize_bootstrap_model_metrics(samples: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (model, metric), group in samples.groupby(["model", "metric"], sort=True):
        values = group["value"].to_numpy(dtype=float)
        rows.append(
            {
                "model": model,
                "metric": metric,
                "bootstrap_mean": float(np.mean(values)),
                "ci95_low": float(np.quantile(values, 0.025)),
                "ci95_high": float(np.quantile(values, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def summarize_bootstrap_differences(samples: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for metric, group in samples.groupby("metric", sort=True):
        values = group["difference_candidate_minus_reference"].to_numpy(dtype=float)
        p_two_sided = 2 * min(float((values <= 0).mean()), float((values >= 0).mean()))
        rows.append(
            {
                "metric": metric,
                "difference_mean": float(np.mean(values)),
                "ci95_low": float(np.quantile(values, 0.025)),
                "ci95_high": float(np.quantile(values, 0.975)),
                "bootstrap_p_two_sided": min(p_two_sided, 1.0),
                "candidate_better_rate": float((values > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def make_publication_table(
    point_metrics: pd.DataFrame,
    model_ci: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []
    point_long = point_metrics.melt(
        id_vars=["model"],
        value_vars=METRIC_COLUMNS,
        var_name="metric",
        value_name="point",
    )
    merged = point_long.merge(model_ci, on=["model", "metric"], how="left")
    for row in merged.to_dict(orient="records"):
        rows.append(
            {
                "model": row["model"],
                "metric": row["metric"],
                "value_ci95": f"{row['point']:.3f} ({row['ci95_low']:.3f}-{row['ci95_high']:.3f})",
            }
        )
    return pd.DataFrame(rows).pivot(index="model", columns="metric", values="value_ci95").reset_index()


def save_model_comparison_outputs(
    output_dir: str | Path,
    aligned_predictions: pd.DataFrame,
    point_metrics: pd.DataFrame,
    bootstrap_metric_ci: pd.DataFrame,
    bootstrap_difference_ci: pd.DataFrame,
    publication_table: pd.DataFrame,
) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    aligned_predictions.to_csv(root / "aligned_predictions.csv", index=False, encoding="utf-8-sig")
    point_metrics.to_csv(root / "point_metrics.csv", index=False, encoding="utf-8-sig")
    bootstrap_metric_ci.to_csv(root / "bootstrap_metric_ci.csv", index=False, encoding="utf-8-sig")
    bootstrap_difference_ci.to_csv(root / "bootstrap_difference_ci.csv", index=False, encoding="utf-8-sig")
    publication_table.to_csv(root / "publication_metric_table.csv", index=False, encoding="utf-8-sig")
