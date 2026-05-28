from __future__ import annotations

import pandas as pd

from eyenet.training.ensemble import build_ensemble_predictions


def test_build_ensemble_predictions_without_fold_groups_by_split_subject_label() -> None:
    predictions = pd.DataFrame(
        {
            "split": ["test", "test", "test", "test"],
            "subject_id": ["001", "001", "002", "002"],
            "label": [1, 1, 0, 0],
            "probability": [0.8, 0.6, 0.1, 0.3],
            "seed": [0, 1, 0, 1],
        }
    )

    ensemble = build_ensemble_predictions(predictions, threshold=0.5)

    assert ensemble[["split", "subject_id", "label", "probability", "n_seeds", "prediction"]].to_dict(
        orient="records"
    ) == [
        {"split": "test", "subject_id": "001", "label": 1, "probability": 0.7, "n_seeds": 2, "prediction": 1},
        {"split": "test", "subject_id": "002", "label": 0, "probability": 0.2, "n_seeds": 2, "prediction": 0},
    ]


def test_build_ensemble_predictions_preserves_fold_when_present() -> None:
    predictions = pd.DataFrame(
        {
            "fold": ["a", "a", "b", "b"],
            "split": ["test", "test", "test", "test"],
            "subject_id": ["001", "001", "001", "001"],
            "label": [1, 1, 1, 1],
            "probability": [0.8, 0.6, 0.2, 0.4],
            "seed": [0, 1, 0, 1],
        }
    )

    ensemble = build_ensemble_predictions(predictions, threshold=0.5)

    assert ensemble[["fold", "split", "subject_id", "probability", "prediction"]].to_dict(orient="records") == [
        {"fold": "a", "split": "test", "subject_id": "001", "probability": 0.7, "prediction": 1},
        {"fold": "b", "split": "test", "subject_id": "001", "probability": 0.30000000000000004, "prediction": 0},
    ]
