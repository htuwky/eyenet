# External Dataset Acquisition Plan

This document tracks what the user needs to download or request before multi-dataset encoder pretraining.

The target is not to train one final model per dataset. Each dataset is first converted into the shared EyeNet event schema, validated independently, then added to a pooled self-supervised pretraining corpus for one universal eye-movement encoder.

## Priority Order

| Priority | Dataset | Current Role | Why It Matters | User Action |
| --- | --- | --- | --- | --- |
| 1 | GazeBase | Large normal-control pretraining source | Learns general eye-movement dynamics across tasks and sessions. | Download if access is open, then place raw files under `data/raw/GazeBase/`. |
| 2 | CRCNS eye-1 | Natural movie/video viewing pretraining source | Closer to the user's video-viewing paradigm than EMS images. | Request/download from CRCNS, then place raw files under `data/raw/CRCNS_eye1/`. |
| 3 | HBN | Adolescent/child auxiliary pretraining source | Useful because the final application is adolescent screening. | Confirm access route and license, then place eye-tracking files under `data/raw/HBN/`. |
| 4 | Saliency4ASD | Auxiliary abnormal-attention dataset | Useful for auxiliary representation learning, but ASD labels must not be merged with SZ labels. | Download if license permits, then place raw files under `data/raw/Saliency4ASD/`. |

## Required Metadata Per Dataset

Each dataset adapter should try to recover:

- `subject_id`
- `dataset_id`
- `timestamp` or event order
- `x`, `y`
- validity or missingness flag if available
- sampling rate
- screen resolution
- screen physical size or visual angle metadata if available
- viewing distance if available
- task or trial boundary if available
- label only when clinically valid for a specific downstream task

If physical screen metadata is missing, normalized-coordinate features can still be used. DVA features should be treated as optional.

## Adapter Acceptance Criteria

Before a dataset can enter encoder pretraining, it must have:

1. A raw-to-event conversion script.
2. A schema validation report.
3. A subject-level QC report.
4. An encoder-ready table using `configs/features/encoder_original_13feature_core.json` unless a named ablation schema is being tested.
5. A single-dataset masked-event smoke test.

## Training Protocol

Recommended sequence:

1. Convert one dataset into the shared schema.
2. Run QC and generate encoder-ready events.
3. Run masked-event smoke pretraining on that dataset alone.
4. Add the dataset to the pooled pretraining corpus only if the smoke test behaves normally.
5. Train one universal encoder on all accepted datasets.
6. Fine-tune/evaluate on EMS and later on the user's hospital cohort.

## User Checklist

The user should prepare:

- Raw dataset files under the expected `data/raw/<dataset_id>/` folder.
- Any license or access notes in a local text file, for example `data/raw/GazeBase/LICENSE_NOTE.txt`.
- Documentation of screen/sampling/task metadata if it is not obvious from the dataset files.
- A note identifying which files contain gaze samples, fixation events, or subject labels.

Do not place large raw data or restricted files under Git.
