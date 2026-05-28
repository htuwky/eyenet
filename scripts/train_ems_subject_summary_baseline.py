from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from eyenet.data.subject_summary import (
    EXCLUDE_COLUMNS,
    apply_summary_feature_set,
    select_summary_feature_columns,
)
from eyenet.training.baseline import compute_metrics, default_model_specs, make_pipeline, predict_positive_probability
from eyenet.training.fixed_split_baseline import attach_fixed_split, make_prediction_frame
from eyenet.training.thresholds import analyze_thresholds, choose_thresholds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EMS subject-summary baseline on a fixed subject split.")
    parser.add_argument("--summary", default="data/processed/EMS/ems_subject_summary_features.csv")
    parser.add_argument("--split", default="data/splits/EMS/multiseed/ems_subject_split_60_20_20_seed0.csv")
    parser.add_argument("--output-dir", default="experiments/ems_subject_summary_baseline/seed0")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--max-train-missing-rate", type=float, default=0.4)
    parser.add_argument(
        "--feature-set",
        choices=["full", "strict"],
        default="full",
        help=(
            "full uses all generated summary features after train-only filtering; strict keeps only "
            "duration, saccade amplitude, fixation-ratio, and angular-entropy summaries."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = pd.read_csv(args.summary, dtype={"subject_id": str})
    data = attach_fixed_split(summary, args.split)
    if data["label"].isna().any():
        missing = data.loc[data["label"].isna(), "subject_id"].tolist()
        raise ValueError(f"Split subjects missing labels in summary table: {missing[:10]}")

    candidate_cols = [col for col in data.columns if col not in EXCLUDE_COLUMNS]
    candidate_cols, feature_set_audit = apply_summary_feature_set(candidate_cols, args.feature_set)
    for col in candidate_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    train_df = data[data["split"] == "train"].copy()
    valid_df = data[data["split"] == "valid"].copy()
    test_df = data[data["split"] == "test"].copy()
    if train_df.empty or valid_df.empty or test_df.empty:
        raise ValueError("Fixed split must contain train, valid, and test subjects.")

    feature_cols, feature_audit = select_summary_feature_columns(
        train_df=train_df,
        candidate_cols=candidate_cols,
        max_train_missing_rate=args.max_train_missing_rate,
    )

    prediction_rows: list[dict] = []
    threshold_rows: list[pd.DataFrame] = []
    selected_rows: list[pd.DataFrame] = []
    metric_rows: list[dict] = []

    for spec in default_model_specs(random_seed=args.random_seed):
        model = make_pipeline(spec)
        model.fit(train_df[feature_cols], train_df["label"].astype(int))

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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(output_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(prediction_rows).to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    pd.concat(threshold_rows, ignore_index=True).to_csv(
        output_dir / "valid_threshold_metrics.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(selected_rows, ignore_index=True).to_csv(
        output_dir / "selected_thresholds.csv", index=False, encoding="utf-8-sig"
    )
    data[["subject_id", "label", "official_fold", "split"]].to_csv(
        output_dir / "split_subjects.csv", index=False, encoding="utf-8-sig"
    )
    feature_audit = feature_audit.merge(feature_set_audit, on="feature", how="right")
    feature_audit.to_csv(output_dir / "feature_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"feature": feature_cols}).to_csv(output_dir / "feature_columns.csv", index=False, encoding="utf-8-sig")
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args) | {"n_selected_features": len(feature_cols)}, handle, indent=2)

    metrics = pd.DataFrame(metric_rows)
    test_rows = metrics[(metrics["split"] == "test") & (metrics["threshold_name"] == "valid_best_balanced_accuracy")]
    print("EMS subject-summary baseline test metrics using validation-selected best balanced-accuracy threshold")
    print(test_rows.sort_values(["balanced_accuracy", "auc"], ascending=False).to_string(index=False))
    print(f"\nfeature_set: {args.feature_set}")
    print(f"selected_features: {len(feature_cols)} / {len(candidate_cols)}")
    print(f"wrote: {output_dir}")


if __name__ == "__main__":
    main()
