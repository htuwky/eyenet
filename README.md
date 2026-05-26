# EyeNet

EyeNet is a content-agnostic eye-movement modeling project for mental-health risk screening. The current codebase establishes a reproducible EMS schizophrenia benchmark and prepares the project for multi-dataset self-supervised encoder pretraining.

The long-term target is a deployable preliminary screening system that can work across eye trackers, screen settings, and viewing paradigms by using eye-movement behavior rather than image or video content.

## Core Idea

The project separates representation learning from disease-specific supervision:

```text
public eye-movement datasets
  -> shared event schema
  -> QC and encoder-ready features
  -> self-supervised universal eye-movement encoder
  -> supervised fine-tuning on EMS and later hospital adolescent data
```

Clinical labels from unrelated disorders must not be merged into a single binary disease label. External datasets such as GazeBase, HBN, CRCNS eye-1, and Saliency4ASD are used for self-supervised pretraining or auxiliary analysis unless their labels match the downstream task.

## Current Status

Implemented:

- EMS raw-to-event table construction.
- Subject-level macro-behavior feature extraction.
- Event-temporal sequence construction.
- Fixed subject-level 60/20/20 train/validation/test split.
- Traditional machine-learning baselines.
- Macro-behavior sequence stream.
- Event-temporal sequence stream.
- Dual-stream concat and gated fusion experiments.
- Subject-level QC reports and filtered EMS variants.
- Encoder-ready no-position event table.
- Masked-event self-supervised pretraining smoke test.
- Supervised encoder downstream transfer smoke test.
- Dataset configs and acquisition notes for EMS, GazeBase, HBN, CRCNS eye-1, and Saliency4ASD.
- HBN and GazeBase raw-gaze adapters into the shared fixation-event schema.
- EMS-only, HBN+EMS, and GazeBase+EMS masked-event pretraining comparisons.

Current conclusion:

- EMS fixed-split supervised modeling is functional.
- Multi-seed masked-event pretraining plus supervised fine-tuning is the current encoder-selection path.
- The current BiGRU fusion pass favors `EMS + GazeBase + CRCNS_eye1 + OneStop` over EMS-only, HBN fusion, and full public fusion by mean downstream AUC.
- HBN and full public fusion did not improve the BiGRU encoder in the current pass, so additional datasets should be treated as an empirical question rather than an automatic improvement.
- The next engineering step is Transformer fusion screening, then encoder hyperparameter narrowing, then downstream dual-stream ablation and final encoder-to-downstream fusion.

See `docs/current_experiment_summary.md` for the current experiment table and next experimental order.
See `docs/server_training_workflow.md` for the remote training and lightweight result-sync workflow.

## Environment Setup

Use the project as an editable package:

```powershell
conda activate eyenet
cd D:\CodeProjects\Python\eyenet
python -m pip install -e .
python -c "import eyenet; print(eyenet.__file__)"
```

Scripts should then run directly with `python scripts/<name>.py` without setting `PYTHONPATH`.

## Repository Layout

```text
configs/
  datasets/       Dataset metadata and local path declarations.
  experiments/    Experiment defaults for reproducible training commands.
  features/       Shared feature schemas.

data/
  raw/            Local raw datasets. Ignored by Git.
  processed/      Generated event tables, QC reports, and encoder-ready data. Ignored by Git.
  splits/         Small reproducibility split files. Tracked by Git.

docs/
  Project design, data dictionaries, engineering protocols, and handoff notes.

experiments/
  Training logs, predictions, checkpoints, and metric tables. Ignored by Git.

scripts/
  CLI entrypoints for preprocessing, training, validation, and summaries.

src/eyenet/
  Reusable package code.
```

## Data Policy

Tracked:

- source code
- configs
- documentation
- small split files under `data/splits/`

Not tracked:

- raw datasets
- processed datasets
- hospital data
- experiment outputs
- checkpoints
- model artifacts
- generated tables

The content-agnostic encoder must not use image pixels, video frames, stimulus categories, or dataset-specific task names as direct predictive features.

## Environment

Create the conda environment:

```powershell
conda env create -f environment.yml
conda activate eyenet
```

Update an existing environment:

```powershell
conda env update -n eyenet -f environment.yml --prune
```

Install the package in editable mode before running scripts:

```powershell
cd D:\CodeProjects\Python\eyenet
python -m pip install -e .
python -c "import eyenet; print(eyenet.__file__)"
```

For one-off execution without activating the shell:

```powershell
conda run -n eyenet python <script>
```

## Active Raw Data Status

Local raw data is expected under:

```text
data/raw/EMS/
data/raw/GazeBase/
data/raw/HBN/
data/raw/CRCNS_eye1/
data/raw/Saliency4ASD/
```

Current local state at the last handoff:

```text
EMS           present
GazeBase      GazeBase_v2_0.zip present
HBN           data.zip present
CRCNS_eye1    empty or pending local verification
Saliency4ASD  downloaded/extracted locally
```

Raw files are intentionally ignored by Git.

## Main EMS Commands

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

Encoder pretraining smoke test:

```powershell
python scripts/train_masked_event_modeling_smoke.py
```

## External Dataset Utilities

Inspect a dataset bundled with `pymovements`:

```powershell
python scripts/inspect_pymovements_dataset.py --dataset HBN --root data/raw/HBN
python scripts/inspect_pymovements_dataset.py --dataset GazeBase --root data/raw/GazeBase
```

Download through `pymovements` when network access works:

```powershell
python scripts/download_pymovements_dataset.py --dataset HBN --root data/raw/HBN
python scripts/download_pymovements_dataset.py --dataset GazeBase --root data/raw/GazeBase
```

## Documentation

Recommended reading order:

- `docs/next_session_handoff.md`
- `docs/engineering_protocol.md`
- `docs/data_dictionary.md`
- `docs/dataset_registry.md`
- `docs/encoder_pretraining_preparation.md`
- `docs/external_dataset_acquisition.md`
- `docs/pymovements_datasets.md`
- `scripts/README.md`

## Next Engineering Step

Screen the remaining public datasets before broad model and hyperparameter ablation:

```text
data/raw/Saliency4ASD/
  -> inspect fixation/raw table structure and metadata
  -> convert usable eye-movement records to shared EyeNet event schema
  -> validate schema
  -> subject-level QC
  -> encoder-ready no-position table
  -> masked-event modeling smoke test
```

CRCNS eye-1 should follow after local-file availability is verified.
