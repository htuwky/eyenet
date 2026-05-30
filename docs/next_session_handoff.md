# Next Session Handoff

For current state, use:

```text
docs/current_experiment_summary.md
docs/engineering_state.md
```

This handoff file is historical and contains earlier HBN-first planning notes. HBN and GazeBase have since been integrated; use the current summary files above for active planning.

Date: 2026-05-21

## Current State

The EMS supervised modeling pipeline is established. Traditional ML, macro-behavior sequence, event-temporal sequence, dual-stream concat, gated fusion, encoder-ready data, supervised encoder smoke test, and masked-event pretraining smoke test have all been implemented.

The current research direction is:

```text
multi-dataset self-supervised eye-movement encoder
  -> supervised fine-tuning on EMS
  -> later fine-tuning/evaluation on hospital adolescent screening data
```

## Important Decisions

- The model must be content-agnostic.
- External disease labels must not be merged into one binary "mental disorder" label.
- GazeBase, CRCNS eye-1, HBN, and Saliency4ASD are mainly for encoder pretraining or auxiliary analysis, not direct SZ supervised training.
- EMS remains the current supervised downstream benchmark.
- The first encoder feature schema is now named `encoder_original_13feature_core`, with 13 always-available features. Historical notes may still mention `encoder_no_position_core`, which is a backward-compatible alias and includes `x_norm`/`y_norm`.
- Do not use image/video content as model input.

## Local Raw Data Status

```text
data/raw/EMS/           present
data/raw/GazeBase/      GazeBase_v2_0.zip present at dataset root
data/raw/HBN/           data.zip present at dataset root
data/raw/CRCNS_eye1/    empty
data/raw/Saliency4ASD/  downloaded/extracted locally
```

GazeBase and HBN zip files are valid archives. They are currently at:

```text
data/raw/GazeBase/GazeBase_v2_0.zip
data/raw/HBN/data.zip
```

`pymovements` usually expects:

```text
data/raw/GazeBase/downloads/GazeBase_v2_0.zip
data/raw/HBN/downloads/data.zip
```

Before using the `pymovements` download/extract path, either move/copy these archives into `downloads/` or update the loader code to accept root-level archives.

## Next Recommended Step

Implement the HBN adapter first.

Reason:

- HBN is smaller than GazeBase.
- HBN is already a video-viewing dataset.
- HBN is closer to the user's adolescent screening application.
- HBN files are direct CSV files inside one zip.

Expected HBN pipeline:

```text
data/raw/HBN/data.zip
  -> inspect columns and metadata
  -> convert raw samples to shared EyeNet event schema
  -> validate schema
  -> subject-level QC
  -> encoder-ready original 13-feature table, or an explicit ablation schema when testing no-position/trend-only variants
  -> HBN masked-event smoke test
```

## Caution

HBN is raw gaze sample data, not fixation events. Do not treat each raw 120 Hz sample as a fixation. The adapter must choose one of:

1. fixed temporal windows for self-supervised sequence learning, or
2. fixation/saccade detection before event table creation.

For the first adapter smoke test, fixed temporal windows are acceptable. For publication-grade experiments, fixation/saccade detection should be validated separately.

## Useful Commands

Environment:

```powershell
conda activate eyenet
cd D:\CodeProjects\Python\eyenet
python -m pip install -e .
```

Inspect bundled `pymovements` metadata:

```powershell
python scripts/inspect_pymovements_dataset.py --dataset HBN --root data/raw/HBN
python scripts/inspect_pymovements_dataset.py --dataset GazeBase --root data/raw/GazeBase
```

## Files to Review First Next Time

```text
docs/engineering_protocol.md
docs/external_dataset_acquisition.md
docs/pymovements_datasets.md
docs/encoder_pretraining_preparation.md
scripts/README.md
```
