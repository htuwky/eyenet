# EyeNet

EyeNet is a content-agnostic eye-movement modeling project for mental-health risk screening. The current codebase establishes a reproducible EMS schizophrenia benchmark and a phase-1 self-supervised encoder result.

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
- Encoder-ready original 13-feature event table plus explicit trend-only ablation schemas.
- Masked-event self-supervised pretraining smoke test.
- Supervised encoder downstream transfer smoke test.
- Dataset configs and acquisition notes for EMS, GazeBase, HBN, CRCNS eye-1, and Saliency4ASD.
- HBN and GazeBase raw-gaze adapters into the shared fixation-event schema.
- EMS-only, HBN+EMS, and GazeBase+EMS masked-event pretraining comparisons.

Current conclusion:

- Phase-1 encoder selection is complete enough for report and draft writing.
- The main result is strict aligned five-seed BiGRU MEM pretraining followed by EMS supervised fine-tuning.
- `EMS-only MEM BiGRU fine-tune` has the strongest mean test balanced accuracy.
- `EMS + GazeBase + CRCNS_eye1 + OneStop` has the strongest mean test AUC among public-data fusion candidates.
- The fixed-split `EMS + GazeBase + CRCNS_eye1 + OneStop` five-seed late ensemble is the current strongest research-profile result.
- Late ensemble results require complete test-subject seed coverage; otherwise they are diagnostics only.
- Public datasets help empirically, but more public sources do not automatically improve downstream EMS balanced accuracy.
- Old segment-GRU dual-stream and new strict summary+encoder dual-stream runs are closed as exploratory evidence because they did not beat the encoder-only balanced-accuracy reference.
- Transformer experiments are future exploratory work, not the current mainline.

See `docs/current_experiment_summary.md` for the current experiment table and next experimental order.
See `docs/encoder_model_selection_summary.md` for the phase-1 encoder source-of-truth table.
See `docs/old_encoder_dual_stream_closure.md` and `docs/new_summary_encoder_dual_stream_closure.md` for dual-stream closure.
See `docs/model_profiles_and_next_steps.md` for the research/deployment profile strategy.
See `docs/project_review_2026-05-28.md` for the latest fixed-split ensemble result and engineering review.
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

The next phase separates two model profiles without duplicating every training run:

```text
research_profile:
  publication-facing AUC/F1/balanced-accuracy optimization

deployment_profile:
  hospital screening prototype focused on stability, QC, calibration, and maintainability
```

Most experiments should train once, save predictions, and then be evaluated under both profiles. Only final candidates should receive profile-specific optimization.

Immediate next step:

```text
1. Recompute existing encoder predictions under F1-oriented threshold policies.
2. Start a small BiGRU hyperparameter ablation for the research profile.
3. Keep the current encoder-only MEM BiGRU as the deployment baseline until a better stable candidate is proven.
```
