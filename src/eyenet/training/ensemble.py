from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def _seed_from_name(name: str) -> int | None:
    match = re.search(r"seed[_-]?(\d+)", name)
    if match is None:
        return None
    return int(match.group(1))


def load_seed_predictions(
    experiment_dir: str | Path,
    contains: str | None = None,
    mode: str | None = None,
) -> pd.DataFrame:
    root = Path(experiment_dir)
    rows: list[pd.DataFrame] = []
    if contains:
        seed_dirs = [path for path in sorted(root.iterdir()) if path.is_dir() and contains in path.name]
    else:
        seed_dirs = sorted(root.glob("seed_*"))

    for seed_dir in seed_dirs:
        seed = _seed_from_name(seed_dir.name)
        if seed is None:
            continue
        prediction_path = seed_dir / "predictions.csv"
        if mode is not None:
            prediction_path = seed_dir / mode / "predictions.csv"
        if not prediction_path.exists():
            continue
        predictions = pd.read_csv(prediction_path, dtype={"subject_id": str})
        predictions["seed"] = seed
        rows.append(predictions)
    if not rows:
        raise FileNotFoundError(f"No seed predictions found under {root}")
    return pd.concat(rows, ignore_index=True)


def build_ensemble_predictions(seed_predictions: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    # Current fixed-split predictions do not require fold; keep it only when present for legacy outputs.
    group_columns = [column for column in ["fold", "split", "subject_id", "label"] if column in seed_predictions.columns]
    if "subject_id" not in group_columns or "label" not in group_columns:
        raise ValueError("Predictions must include subject_id and label columns.")

    grouped = (
        seed_predictions.groupby(group_columns, as_index=False)
        .agg(
            probability=("probability", "mean"),
            probability_std=("probability", "std"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values(group_columns)
        .reset_index(drop=True)
    )
    grouped["prediction"] = (grouped["probability"] >= threshold).astype(int)
    return grouped


def save_ensemble_predictions(output_dir: str | Path, seed_predictions: pd.DataFrame, ensemble: pd.DataFrame) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    seed_predictions.to_csv(root / "all_seed_predictions.csv", index=False, encoding="utf-8-sig")
    ensemble.to_csv(root / "ensemble_predictions.csv", index=False, encoding="utf-8-sig")
