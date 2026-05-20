# EyeNet

EyeNet is a content-agnostic eye-movement modeling project for mental-health risk screening. The current implementation focuses on the EMS schizophrenia eye-movement dataset and establishes a reproducible baseline before moving to multi-dataset encoder pretraining.

The project goal is to build a screening pipeline that depends on eye-tracking signals and acquisition metadata, not on a specific image, video, or viewing paradigm.

## Current Status

Implemented:

- EMS raw-to-event table construction.
- Subject-level macro-behavior feature extraction.
- Event-temporal sequence construction.
- Fixed subject-level 60/20/20 train/validation/test split.
- Traditional machine-learning baselines.
- Macro-behavior stream.
- Event-temporal stream.
- Dual-stream concat fusion.
- Dual-stream gated fusion.
- Experiment configs and deployment-relevant preprocessing checkpoint saving.

Current main EMS result is the fixed-split dual-stream concat model. Gated fusion is kept as a supplementary/negative experiment because the current gate did not improve the held-out test result.

## Repository Layout

```text
configs/          Dataset and experiment YAML configs
data/splits/      Small reproducibility split files only
docs/             Project, data, experiment, and engineering documentation
scripts/          Preprocessing, feature extraction, training, and analysis entrypoints
src/eyenet/       Reusable package code
```

Large raw data, processed tables, hospital data, outputs, experiments, checkpoints, and model artifacts are intentionally excluded from Git.

## Environment

Create the conda environment:

```powershell
conda env create -f environment.yml
conda activate eyenet
```

For local package imports:

```powershell
$env:PYTHONPATH="D:\CodeProjects\Python\eyenet\src"
```

## Active Data Inputs

The active EMS scripts expect these local files:

```text
data/processed/EMS/ems_segment_features_no_pupil.csv
data/processed/EMS/ems_event_temporal_sequences_no_pupil.csv
data/splits/EMS/ems_subject_split_60_20_20_seed42.csv
```

Only the split file is tracked by Git. The processed feature tables must be regenerated locally or restored from local storage.

## Main Commands

Traditional ML baseline:

```powershell
python scripts/train_ems_baseline_fixed_split.py --config configs/experiments/ems_baseline.yaml
```

Macro-behavior stream:

```powershell
python scripts/train_ems_segment_sequence_fixed_split.py --config configs/experiments/ems_macro_behavior_fixed_split.yaml --device cuda
```

Event-temporal stream:

```powershell
python scripts/train_ems_event_temporal_sequence_fixed_split.py --config configs/experiments/ems_event_temporal_fixed_split.yaml --device cuda
```

Dual-stream concat:

```powershell
python scripts/train_ems_dual_stream_concat_fixed_split.py --config configs/experiments/ems_dual_stream.yaml --device cuda
```

Dual-stream gated:

```powershell
python scripts/train_ems_dual_stream_gated_fixed_split.py --config configs/experiments/ems_dual_stream_gated_fixed_split.yaml --device cuda
```

## Engineering Notes

Deep-learning checkpoints save:

```text
checkpoints/best.pt
checkpoints/preprocessor.joblib
```

`preprocessor.joblib` is required for valid inference because it stores train-fitted imputers and scalers. Do not refit preprocessing on validation, test, or deployment subjects.

For cross-dataset validation, run position-feature ablations with:

```powershell
--no-segment-position
```

## Documentation

- `docs/data_dictionary.md`
- `docs/model_design.md`
- `docs/experiment_protocol.md`
- `docs/engineering_state.md`
- `docs/preprocessing_protocol.md`

## Next Engineering Steps

1. Add a dataset registry and schema validator for cross-dataset ingestion.
2. Add an inference script that loads `best.pt` plus `preprocessor.joblib`.
3. Run no-position ablations before multi-dataset encoder training.
4. Add model cards and deployment quality-control rules.
