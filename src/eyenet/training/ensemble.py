from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_seed_predictions(experiment_dir: str | Path) -> pd.DataFrame:
    root = Path(experiment_dir)
    rows: list[pd.DataFrame] = []
    for seed_dir in sorted(root.glob("seed_*")):
        prediction_path = seed_dir / "predictions.csv"
        if not prediction_path.exists():
            continue
        seed = seed_dir.name.replace("seed_", "")
        predictions = pd.read_csv(prediction_path, dtype={"subject_id": str})
        predictions["seed"] = int(seed)
        rows.append(predictions)
    if not rows:
        raise FileNotFoundError(f"No seed predictions found under {root}")
    return pd.concat(rows, ignore_index=True)


def build_ensemble_predictions(seed_predictions: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    grouped = (
        seed_predictions.groupby(["fold", "subject_id", "label"], as_index=False)
        .agg(
            probability=("probability", "mean"),
            probability_std=("probability", "std"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values(["fold", "subject_id"])
        .reset_index(drop=True)
    )
    grouped["prediction"] = (grouped["probability"] >= threshold).astype(int)
    return grouped


def save_ensemble_predictions(output_dir: str | Path, seed_predictions: pd.DataFrame, ensemble: pd.DataFrame) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    seed_predictions.to_csv(root / "all_seed_predictions.csv", index=False, encoding="utf-8-sig")
    ensemble.to_csv(root / "ensemble_predictions.csv", index=False, encoding="utf-8-sig")
