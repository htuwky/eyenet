# Project Status

For the current authoritative experiment summary, see:

```text
docs/current_experiment_summary.md
```

This file is retained as a historical project-status note and may lag behind the active encoder-pretraining work.

## Goal

Build a content-agnostic eye-movement screening framework that can generalize across hardware, sampling rates, and viewing paradigms after appropriate preprocessing.

The model should use eye movement behavior, not visual stimulus content.

## Dataset State

EMS remains the supervised downstream benchmark. HBN and GazeBase have been converted into the shared fixation-event schema for self-supervised pretraining comparisons. Saliency4ASD and CRCNS eye-1 are the remaining public dataset-screening targets.

EMS local data status:

- Raw and processed EMS data are stored locally and excluded from Git.
- Fixed split metadata is tracked under `data/splits/EMS/`.
- Current model features exclude pupil diameter because it is not consistently available in the target collection setting.

## Current Model Families

### Traditional ML Baseline

Uses subject-level aggregate eye-movement features.

Purpose:

- Establish a strong non-deep-learning baseline.
- Provide interpretable comparison for the paper.
- Check whether deep learning actually adds value.

### Macro-Behavior Stream

Uses segment-level aggregate behavior features over the viewing sequence.

Purpose:

- Model subject-level behavior patterns such as fixation duration, scanpath length, spatial coverage, saccade amplitude, and transition velocity.
- Preserve coarse temporal progression across viewing windows.

### Event-Temporal Stream

Uses event-level fixation/saccade sequence features.

Purpose:

- Model fine-grained temporal dynamics.
- Complement macro behavior with event order, duration, transition amplitude, direction, and velocity information.

### Dual-Stream Concat Fusion

Concatenates macro-behavior and event-temporal representations before classification.

Current role:

- Main dual-stream model candidate.
- Best current deep-learning fusion choice.

### Dual-Stream Gated Fusion

Learns adaptive macro/event weighting.

Current role:

- Supplementary experiment.
- Did not improve the current held-out test result, so it is not the main model.

## Current Main Evaluation Protocol

- Subject-level 60/20/20 train/validation/test split.
- Validation split is used for threshold selection.
- Test split is used only for final reporting.
- Report AUC, accuracy, balanced accuracy, sensitivity, specificity, F1, TP, TN, FP, and FN.

## Engineering Work Completed

- Config-driven experiment entrypoints.
- Train-fitted preprocessing saved with deep-learning checkpoints.
- PyTorch/CUDNN seed controls.
- No-position feature switch for cross-dataset ablation.
- Fixed-split summary and fusion-analysis scripts.
- Paper draft generation script.

## Historical Immediate Next Steps

1. Run no-position ablations for Macro and Dual Concat.
2. Build a standardized inference script.
3. Add dataset registry and schema validation before adding new datasets.
4. Start cross-dataset encoder pretraining only after the schema is stable.
