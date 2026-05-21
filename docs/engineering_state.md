# EyeNet Engineering State

## Current Scope

The current codebase supports EMS fixed-split experiments for a content-agnostic eye-movement screening model. The active protocol is a subject-level 60/20/20 train/validation/test split.

## Active Inputs

- Macro-behavior segment table: `data/processed/EMS/ems_segment_features_no_pupil.csv`
- Event-temporal sequence table: `data/processed/EMS/ems_event_temporal_sequences_no_pupil.csv`
- Subject split: `data/splits/EMS/ems_subject_split_60_20_20_seed42.csv`

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

Use `--config` to select an experiment config and `--device cuda` for GPU training.

## Deployment-Relevant State

Deep-learning checkpoints now save the preprocessing state in `checkpoints/preprocessor.joblib`. This includes the train-fitted imputers and scalers needed to reproduce valid/test/inference normalization.

Determinism settings are enabled for PyTorch/CUDNN seed control, but exact GPU reproducibility can still vary by CUDA, PyTorch, and hardware version.

## Known Open Issues

1. Position-feature ablation is still needed before cross-dataset training. Use `--no-segment-position` to disable segment position features.
2. There is no deployment inference script yet. The next engineering step should load `best.pt` plus `preprocessor.joblib` and run prediction on a standardized subject table.
3. Cross-dataset training now has initial dataset configs, a dataset registry, and a shared event-table schema validator.
4. Cross-dataset training still needs dataset adapters, a QC policy, and an encoder-ready feature schema.
5. Smoke-test outputs under `experiments/smoke_tests/` are for engineering verification only and should not be used as paper results.
