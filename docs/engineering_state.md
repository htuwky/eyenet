# EyeNet Engineering State

## Current Scope

The current codebase supports EMS fixed-split experiments and multi-dataset self-supervised encoder pretraining for a content-agnostic eye-movement screening model. The active downstream protocol is a subject-level 60/20/20 train/validation/test split on EMS.

## Active Inputs

- Macro-behavior segment table: `data/processed/EMS/ems_segment_features_no_pupil.csv`
- Event-temporal sequence table: `data/processed/EMS/ems_event_temporal_sequences_no_pupil.csv`
- Subject split: `data/splits/EMS/ems_subject_split_60_20_20_seed42.csv`
- Encoder-ready EMS table: `data/processed/EMS/encoder_ready/clipped_qc_no_position/ems_encoder_events.csv`
- Encoder-ready HBN table: `data/processed/HBN/encoder_ready/no_position/hbn_encoder_events_qc.csv`
- Encoder-ready GazeBase table: `data/processed/GazeBase/encoder_ready/no_position/gazebase_encoder_events.csv`

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

Use `--config` to select an experiment config and `--device cuda` for GPU training.

## Deployment-Relevant State

Deep-learning checkpoints now save the preprocessing state in `checkpoints/preprocessor.joblib`. This includes the train-fitted imputers and scalers needed to reproduce valid/test/inference normalization.

Determinism settings are enabled for PyTorch/CUDNN seed control, but exact GPU reproducibility can still vary by CUDA, PyTorch, and hardware version.

## Known Open Issues

1. There is no deployment inference script yet. The next deployment-oriented step should load `best.pt` plus `preprocessor.joblib` and run prediction on a standardized subject table.
2. Saliency4ASD and CRCNS eye-1 still need adapter screening.
3. Broad hyperparameter search is intentionally deferred until remaining data sources are screened.
4. Public-dataset labels must not be merged into EMS SZ/HC labels.
5. Smoke-test outputs under `experiments/smoke_tests/` are for engineering verification only and should not be used as paper results.

## Current Experiment Summary

See `docs/current_experiment_summary.md`.
