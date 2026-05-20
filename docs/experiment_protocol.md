# Experiment Protocol

## Evaluation Rules

- Use subject-level splits only.
- For EMS, use the fixed subject-level 60/20/20 train/validation/test split for the current main experiments.
- Select operating thresholds on the validation split only, then report final metrics on the held-out test split.
- Report AUC, accuracy, balanced accuracy, sensitivity, specificity, and F1.
- For deployment-oriented experiments, also report calibration and decision-curve metrics.

## Baseline First

Before deep models, run interpretable machine-learning baselines:

- Logistic Regression
- SVM
- Random Forest
- HistGradientBoosting or LightGBM
- MLP

## Required Ablations

- Macro-behavior stream only
- Event-temporal stream only
- Macro-behavior + event-temporal dual stream
- Concat fusion vs gated fusion
- With vs without segment-position features
- Pupil excluded vs optional pupil included
- Single-source vs multi-source training when additional datasets are available
- Without vs with domain adaptation when multi-source training begins

## Current EMS Main Result Policy

The current main model candidate is dual-stream concat fusion. Gated fusion is retained as a supplementary experiment because it did not improve the held-out test result in the current fixed split.

Smoke-test outputs under `experiments/smoke_tests/` are engineering checks only and must not be reported as paper results.
