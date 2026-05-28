# EyeNet Project Review

Date: 2026-05-28

## Current Outcome

The current strongest research-profile result is:

```text
fixed-split EMS+GazeBase+CRCNS eye-1+OneStop BiGRU MEM fine-tune late ensemble
```

Primary output root:

```text
experiments/research_profile/fixed_split_onestop_ensemble/
```

Important files:

```text
experiments/research_profile/fixed_split_onestop_ensemble/fixed_onestop_fiveseed_threshold_strategy_summary.csv
experiments/research_profile/fixed_split_onestop_ensemble/fixed_onestop_fiveseed_late_ensemble/selected_thresholds.csv
experiments/research_profile/fixed_split_onestop_ensemble/fixed_onestop_fiveseed_late_ensemble/seed_coverage.csv
```

The fixed-split late ensemble is valid because the coverage check passed:

```text
n_seeds=5
n_rows=32
```

## Main Results

Five-seed fixed-split single-model summary:

| Policy | AUC Mean | BA Mean | Sens Mean | Spec Mean | F1 Mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| valid_best_balanced_accuracy | 0.872 | 0.738 | 0.875 | 0.600 | 0.770 |
| valid_screening_sensitivity_at_least_0.80 | 0.872 | 0.713 | 0.913 | 0.513 | 0.764 |
| valid_best_f1 | 0.872 | 0.706 | 0.888 | 0.525 | 0.754 |
| default_0.50 | 0.872 | 0.706 | 0.800 | 0.613 | 0.737 |

Fixed-split late ensemble:

| Criterion | Threshold | AUC | BA | Sens | Spec | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| best_f1 | 0.38 | 0.898 | 0.781 | 0.938 | 0.625 | 0.811 |
| sensitivity_at_least_0.90 | 0.38 | 0.898 | 0.781 | 0.938 | 0.625 | 0.811 |
| sensitivity_at_least_0.80 | 0.41 | 0.898 | 0.781 | 0.875 | 0.688 | 0.800 |
| best_balanced_accuracy | 0.80 | 0.898 | 0.813 | 0.625 | 1.000 | 0.769 |
| fixed_0.50 | 0.50 | 0.898 | 0.781 | 0.750 | 0.813 | 0.774 |

## Closed Lines

- `research_ems_only_bigger96` is closed. It did not beat the existing EMS-only and OneStop candidates after five seeds.
- Old dual-stream and strict summary+encoder dual-stream runs remain exploratory closure evidence.
- Frozen encoder probing is not the main downstream method.

## Engineering State

Implemented and active:

- `src/` package layout with editable install support.
- `pyproject.toml` with pytest and ruff configuration.
- Train-fitted preprocessing saved with checkpoints.
- Strict content-agnostic encoder feature policy.
- Aligned MEM pretraining split generator.
- Multi-seed threshold summarization.
- Multi-seed prediction analysis with no hard `fold` dependency.
- Complete-seed coverage check for late ensemble validity.

Current code changes:

- `scripts/analyze_multiseed_predictions.py`
- `src/eyenet/training/ensemble.py`
- `tests/test_ensemble.py`

## Engineering Rules

1. Use randomized five-seed summaries for model selection across subject splits.
2. Use fixed-split late ensemble only when `seed_coverage.csv` confirms complete seed coverage.
3. Select thresholds from validation predictions only.
4. Do not use image pixels, stimulus semantics, AOI labels, task names, or device IDs as direct model inputs.
5. Do not merge public-dataset disease labels into EMS SZ/HC supervision.
6. Keep experiment outputs and checkpoints out of Git.

## Recommended Next Work

1. Freeze the research-profile result table in the report.
2. Add deployment-profile calibration on validation predictions.
3. Implement `scripts/predict_subject_risk.py` for checkpoint + preprocessor inference.
4. Add a small model-card or run-card format for promoted checkpoints.
