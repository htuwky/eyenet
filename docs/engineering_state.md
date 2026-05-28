# EyeNet Engineering State

## Current Scope

The current codebase supports EMS fixed-split experiments, randomized multi-seed model selection, and multi-dataset self-supervised encoder pretraining for a content-agnostic eye-movement screening model.

The active downstream protocols are:

- randomized five-seed EMS subject splits for model-selection stability;
- fixed EMS seed42 split for final research-profile ensemble evaluation.

The active EMS evaluation field is `split`, with values `train`, `valid`, and `test`.
The old EMS official fold assignment is retained only as `official_fold` metadata in fixed-split files.
Legacy scripts that train on `fold` are not the current mainline.

## Active Inputs

- Macro-behavior segment table: `data/processed/EMS/ems_segment_features_no_pupil.csv`
- Event-temporal sequence table: `data/processed/EMS/ems_event_temporal_sequences_no_pupil.csv`
- Fixed EMS subject split: `data/splits/EMS/ems_subject_split_60_20_20_seed42.csv`
- Encoder-ready EMS table: `data/processed/EMS/encoder_ready/clipped_qc_no_position/ems_encoder_events.csv`
- Encoder-ready HBN table: `data/processed/HBN/encoder_ready/no_position/hbn_encoder_events_qc.csv`
- Encoder-ready GazeBase table: `data/processed/GazeBase/encoder_ready/no_position/gazebase_encoder_events.csv`
- Fixed-split OneStop ensemble root: `experiments/research_profile/fixed_split_onestop_ensemble/`

All active model features are derived from eye-tracking traces and acquisition metadata, not from image or video content.

## Active Experiment Configs

- ML baseline: `configs/experiments/ems_baseline.yaml`
- Macro-behavior stream: `configs/experiments/ems_macro_behavior_fixed_split.yaml`
- Event-temporal stream: `configs/experiments/ems_event_temporal_fixed_split.yaml`
- Dual-stream concat: `configs/experiments/ems_dual_stream.yaml`
- Dual-stream gated: `configs/experiments/ems_dual_stream_gated_fixed_split.yaml`

Each config is intentionally explicit about data paths, output directory, model hyperparameters, and training hyperparameters. Command-line arguments can override config values.

## Active Training Entrypoints

- `python scripts/train_ems_baseline_fixed_split.py`
- `python scripts/train_ems_segment_sequence_fixed_split.py`
- `python scripts/train_ems_event_temporal_sequence_fixed_split.py`
- `python scripts/train_ems_dual_stream_concat_fixed_split.py`
- `python scripts/train_ems_dual_stream_gated_fixed_split.py`
- `python scripts/train_mem_pretrain.py`
- `python scripts/train_supervised_encoder_smoke.py`
- `python scripts/run_encoder_architecture_experiment.py`
- `python scripts/analyze_multiseed_predictions.py`

Use `--config` to select an experiment config and `--device cuda` for GPU training.

## Deployment-Relevant State

Deep-learning checkpoints now save the preprocessing state in `checkpoints/preprocessor.joblib`. This includes the train-fitted imputers and scalers needed to reproduce valid/test/inference normalization.

Determinism settings are enabled for PyTorch/CUDNN seed control, but exact GPU reproducibility can still vary by CUDA, PyTorch, and hardware version.

## Current Research-Profile Result

The strongest current research-profile result is the fixed-split five-seed late ensemble:

```text
experiments/research_profile/fixed_split_onestop_ensemble/fixed_onestop_fiveseed_late_ensemble/selected_thresholds.csv
```

Coverage check:

```text
experiments/research_profile/fixed_split_onestop_ensemble/fixed_onestop_fiveseed_late_ensemble/seed_coverage.csv
n_seeds=5
n_rows=32
```

Key operating points:

- `best_f1`: AUC 0.898, balanced accuracy 0.781, sensitivity 0.938, specificity 0.625, F1 0.811.
- `best_balanced_accuracy`: AUC 0.898, balanced accuracy 0.813, sensitivity 0.625, specificity 1.000, F1 0.769.

Late ensemble is only valid when `seed_coverage.csv` confirms complete seed coverage for all test subjects.

## Known Open Issues

1. There is no deployment inference script yet. The next deployment-oriented step should load `best.pt` plus `preprocessor.joblib` and run prediction on a standardized subject table.
2. Deployment-profile calibration is not implemented yet.
3. Saliency4ASD remains deferred because its pseudo-subject/session structure and ASD labels do not align with the EMS SZ/HC downstream task.
4. Broad hyperparameter search is intentionally deferred; the `bigger96` research architecture line is closed.
5. Public-dataset labels must not be merged into EMS SZ/HC labels.
6. Legacy official-fold scripts remain in the repository for diagnostics and historical reproducibility, but they should not be used for new mainline results.
7. Smoke-test outputs under `experiments/smoke_tests/` are for engineering verification only and should not be used as paper results.

## Current Experiment Summary

See `docs/current_experiment_summary.md`.
