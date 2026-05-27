from __future__ import annotations

import pandas as pd

from eyenet.data.subject_summary import apply_summary_feature_set, select_summary_feature_columns


def test_strict_summary_feature_set_keeps_behavioral_dynamics() -> None:
    candidates = [
        "seg__fix_duration_ms__mean",
        "event__saccade_amplitude_norm__std",
        "event__x_norm__mean",
        "summary_n_events",
        "event__transition_missing__median",
    ]

    selected, audit = apply_summary_feature_set(candidates, "strict")

    assert selected == ["seg__fix_duration_ms__mean", "event__saccade_amplitude_norm__std"]
    assert set(audit["feature"]) == set(candidates)


def test_summary_feature_selection_uses_train_only_missing_and_constant_filters() -> None:
    train_df = pd.DataFrame(
        {
            "good_feature": [1.0, 2.0, 3.0, 4.0],
            "constant_feature": [1.0, 1.0, 1.0, 1.0],
            "high_missing_feature": [None, None, None, 1.0],
        }
    )

    selected, audit = select_summary_feature_columns(
        train_df=train_df,
        candidate_cols=["good_feature", "constant_feature", "high_missing_feature"],
        max_train_missing_rate=0.4,
    )

    assert selected == ["good_feature"]
    reasons = dict(zip(audit["feature"], audit["drop_reason"], strict=True))
    assert reasons["constant_feature"] == "constant_or_all_missing_train"
    assert reasons["high_missing_feature"] == "high_train_missing"
