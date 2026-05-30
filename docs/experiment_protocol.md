# Experiment Protocol

## Evaluation Rules

- Use subject-level splits only.
- For EMS model selection, use subject-level 60/20/20 train/validation/test splits with seeds `0,1,2,3,4`.
- Fixed-split runs are allowed for controlled diagnostics and late-ensemble analysis, but they must be reported separately from randomized five-seed model-selection results.
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

## Result Tiers

The project distinguishes four result tiers:

- `smoke`: engineering checks only. These verify that code runs and must not be reported as scientific results.
- `single_split`: fast screening on one split. These can guide the next run but do not support final claims.
- `exploratory`: useful model-search evidence, including non-aligned mixed pretraining, Transformer trials, and dual-stream prototypes.
- `final_aligned_5seed`: strict EMS-anchor aligned five-seed results. These are the default phase-1 results for main paper/report tables.
- `fixed_split_profile`: controlled fixed-EMS-split reruns used for profile diagnostics, calibration, or late-ensemble analysis. These must not be mixed with randomized five-seed rows in the same claim.

## Current EMS Main Result Policy

The current phase-1 main line is encoder pretraining and EMS fine-tuning, not dual-stream fusion or broad architecture search.

Use the strict aligned five-seed encoder results as the current source of truth:

```text
experiments/encoder_downstream/phase1_encoder_summary.csv
experiments/encoder_downstream/phase1_encoder_split_leakage_audit.csv
docs/encoder_model_selection_summary.md
```

Dual-stream models are retained as phase-2 exploratory work. They should not be used to select or describe the phase-1 primary encoder.

Smoke-test outputs under `experiments/smoke_tests/` and temporary smoke directories are engineering checks only and must not be reported as paper results.

Current feature-schema policy:

- Original 13-feature public-fusion BiGRU is the deployment baseline.
- Trend-only public-fusion BiGRU is the research/generalization candidate.
- Trend plus subject-centered position is a closed negative ablation.

## Model Profiles

Future experiments should distinguish two model profiles:

- `research_profile`: publication-oriented evaluation optimized for AUC, F1, balanced accuracy, sensitivity, specificity, and statistical comparison.
- `deployment_profile`: hospital screening prototype evaluation optimized for stability, QC behavior, calibration, fixed operating points, and maintainability.

The default workflow is not to train every experiment twice. Train once, save checkpoint and predictions, then evaluate the same predictions under both profile-specific threshold and reporting policies.

Profile-specific extra training is reserved for final candidates only.

See:

```text
docs/model_profiles_and_next_steps.md
```
