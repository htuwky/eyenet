# EyeNet Model Profiles and Next Steps

Last updated: 2026-05-28

## Purpose

EyeNet now needs two clearly separated model profiles:

```text
research_profile
  publication-oriented model evaluation
  optimized for AUC, F1, balanced accuracy, and reviewer-facing comparisons

deployment_profile
  hospital screening prototype
  optimized for stability, interpretability, QC, calibration, and maintainability
```

This does not mean every experiment must be trained twice. The default rule is:

```text
train once
save checkpoint and predictions
evaluate under both profiles
only profile-specific final candidates receive extra optimization
```

## Shared Foundation

Both profiles must share the same content-agnostic project foundation:

- shared fixation/event schema
- shared subject-level split policy
- shared leakage audit rules
- shared train-only preprocessing
- shared QC policy
- shared MEM encoder pretraining pipeline
- no image pixels, stimulus semantics, AOI labels, task names, device model IDs, or sampling-rate one-hot features as direct predictive inputs

The profiles differ only after model prediction:

```text
classifier or ensemble policy
threshold policy
calibration policy
reporting format
deployment output contract
```

## Research Profile

Goal:

```text
maximize publication-facing metrics while keeping validation/test separation strict
```

Primary metrics:

- AUC
- F1
- balanced accuracy
- sensitivity
- specificity
- seed-to-seed standard deviation
- paired comparisons or bootstrap confidence intervals when available

Allowed methods:

- BiGRU encoder hyperparameter ablation
- validation-selected best F1 threshold
- validation-selected best balanced accuracy threshold
- sensitivity-constrained threshold reports
- selective public-data pretraining source choices
- late probability ensemble
- calibrated probability ensemble

Not allowed:

- selecting thresholds on the test split
- choosing a final model from test-set inspection without validation justification
- using content, task, stimulus, or device identifiers that violate the content-agnostic claim
- mixing unrelated disease labels into EMS SZ/HC supervision

Current research baseline:

```text
encoder-only MEM BiGRU fine-tune
EMS-only: strongest balanced accuracy reference
EMS+GazeBase+CRCNS+OneStop: strongest public-data AUC candidate
```

Near-term research work:

1. Recompute existing predictions under F1-oriented and sensitivity-constrained threshold policies.
2. Run a small BiGRU hyperparameter ablation on encoder-only models.
3. If single-model gains are limited, test late ensembles using validation-selected weights or simple averaging.
4. Update paper tables only after strict five-seed results are available.

## Deployment Profile

Goal:

```text
produce a stable hospital screening prototype, not a clinical diagnosis system
```

Primary concerns:

- robust QC gate
- calibrated risk score
- high-sensitivity operating point
- predictable false-positive burden
- simple model/version tracking
- stable behavior across device and paradigm shifts
- readable output for clinicians or researchers

Preferred methods:

- single encoder-only MEM BiGRU fine-tune checkpoint family
- fixed validation-derived deployment threshold
- optional calibration layer
- QC warning and out-of-distribution warning
- subject-level report output

Avoid in the first deployment prototype:

- large ensembles that are hard to maintain
- profile-specific features that are unavailable in hospital data
- threshold policies that maximize paper metrics but create unstable clinical behavior
- complex dual-stream models unless they show stable benefit and clear operational value

Current deployment baseline:

```text
encoder-only MEM BiGRU fine-tune
threshold policy: sensitivity_at_least_0.90 or fixed-specificity policy, selected on validation only
output: risk_score + predicted_label + threshold + QC warnings + model_version
```

Future deployment entry point:

```text
scripts/predict_subject_risk.py
```

Expected output contract:

```json
{
  "subject_id": "...",
  "risk_score": 0.73,
  "threshold": 0.58,
  "predicted_label": "risk",
  "qc_status": "pass",
  "warnings": [],
  "model_version": "..."
}
```

## Experiment Flow

For most future experiments:

```text
1. Train one model.
2. Save checkpoint, preprocessing state, validation predictions, and test predictions.
3. Generate research_profile metrics.
4. Generate deployment_profile metrics.
5. Promote only strong candidates to five-seed evaluation.
```

Profile-specific extra training should happen only when a model is already a serious candidate:

```text
research_profile final candidate
  additional hyperparameter tuning or ensemble selection

deployment_profile final candidate
  calibration, threshold hardening, inference packaging, and QC reporting
```

## Immediate Next Steps

The next work should start with no-training analysis:

1. Summarize existing encoder predictions under multiple threshold policies.
2. Compare `valid_best_balanced_accuracy` with `valid_best_f1`.
3. Identify whether F1 can improve without retraining.
4. Decide whether BiGRU hyperparameter ablation should optimize EMS-only, EMS+GazeBase+CRCNS+OneStop, or both.

Then run the first small research-profile tuning batch:

| Experiment | projection | hidden | dropout | max_seq_len | mask_probability | Purpose |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 64 | 64 | 0.3 | 1500 | 0.45 | Current reference |
| bigger96 | 128 | 96 | 0.3 | 1500 | 0.45 | Test moderate capacity increase |
| bigger128_regularized | 128 | 128 | 0.4 | 1500 | 0.45 | Test larger model with stronger regularization |
| longer_seq | 64 | 64 | 0.3 | 3000 | 0.45 | Test whether sequence truncation limits performance |
| lower_mask | 64 | 64 | 0.3 | 1500 | 0.35 | Test easier MEM task |
| higher_mask | 64 | 64 | 0.3 | 1500 | 0.55 | Test stronger MEM objective |

Use seed-0 screening first. Promote only the best two or three candidates to five-seed.

## Reporting Rule

Use this terminology consistently:

```text
phase-1 main model:
  current encoder-only MEM BiGRU fine-tune result

research_profile candidate:
  metric-optimized publication candidate

deployment_profile candidate:
  stable hospital screening prototype candidate

exploratory evidence:
  old dual-stream, new summary dual-stream, Transformer, and non-aligned screenings unless rerun under strict profile rules
```

This keeps the paper story and deployment story aligned without forcing every experiment to be trained twice.
