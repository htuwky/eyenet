# Encoder Pretraining Preparation

This document tracks the required preparation before multi-dataset encoder pretraining.

## Current Principle

The encoder should learn content-agnostic eye-movement representations. The universal encoder operates on fixation events. Datasets with raw gaze samples must first run fixation detection, then enter the shared fixation-event schema.

Clinical labels from different disorders must not be merged into one binary label.

## Current Authority

This document records the historical preparation process. For the current experiment state and next experimental order, use:

```text
docs/current_experiment_summary.md
```

## Preparation Checklist

| Item | Status | Notes |
| --- | --- | --- |
| Dataset configs | Done for EMS, GazeBase, CRCNS eye-1, HBN, Saliency4ASD | HBN/GazeBase/Saliency4ASD are available locally; CRCNS eye-1 is still absent. |
| Dataset registry | Done | See `docs/dataset_registry.md`. |
| Shared event schema | Done | See `docs/data_dictionary.md`. |
| Schema validator | Done | `scripts/validate_dataset_schema.py`. |
| EMS schema validation | Done | Structural validation passes; QC warnings remain. |
| QC policy | Initial version done | `scripts/build_subject_qc_report.py` outputs subject-level flags and usable-subject lists. |
| HBN adapter | Done | `scripts/build_hbn_fixation_events.py` converts HBN raw gaze to fixation events with I-DT. |
| GazeBase adapter | Done | `scripts/build_gazebase_fixation_events.py` converts GazeBase raw DVA gaze to fixation events with I-DT. |
| Dataset adapters | In progress | Saliency4ASD and CRCNS eye-1 remain to be screened. |
| Encoder-ready feature schema | Done | Current original 13-feature schema is `encoder_original_13feature_core`. Legacy generated paths may still contain `no_position` in their directory names. |
| Encoder DataLoader | Done | Subject-level sequences with mask, IDs, train-only normalization, and optional labels for self-supervised data. |
| Masked event pretraining script | Done | Uses fixation span masking by default and checkpoint exports `encoder_state_dict`. |
| Pretrained encoder downstream transfer | Initial version done | Compared from-scratch, fine-tuned pretrained, and frozen pretrained encoders on EMS. |
| Inference script | Pending | Needed before deployment-oriented evaluation. |

## EMS Validation Result

EMS event table:

```text
data/processed/EMS/ems_events.csv
```

Validation report:

```text
data/processed/EMS/ems_events_schema_report.json
```

Current structural status:

```text
passed: true
```

Known warnings:

- Missing labels exist because the full EMS table includes the official unlabeled test portion.
- A small number of `x_norm` and `y_norm` values fall outside `[0, 1]`, which should be handled by QC or clipping policy before cross-dataset pretraining.

## Subject-Level QC

QC script:

```powershell
python scripts/build_subject_qc_report.py `
  --events data/processed/EMS/ems_events.csv `
  --output-dir data/processed/EMS/qc `
  --require-label-for-self-supervised
```

Outputs:

```text
subject_qc_report.csv
usable_supervised_subjects.csv
usable_self_supervised_subjects.csv
subject_qc_summary.json
```

The `--require-label-for-self-supervised` flag is recommended for EMS so the unlabeled official test portion is excluded from EMS encoder preparation.

## Recommended Next Step

HBN and GazeBase have both passed the full adapter -> schema -> QC -> encoder-ready -> MEM downstream path. The next dataset task is Saliency4ASD adapter development, followed by CRCNS eye-1 local-file verification.

## HBN Adapter Smoke Test

HBN raw gaze files are converted into fixation events with a conservative I-DT detector:

```powershell
python scripts/build_hbn_fixation_events.py `
  --max-files 20 `
  --output data/processed/HBN/smoke/hbn_fixation_events.csv
```

Smoke-test result:

```text
n_files: 20
n_files_with_fixations: 20
n_subjects: 8
n_fixation_events: 11670
median_duration_ms: 150.0
```

The HBN smoke table passes schema validation with missing labels allowed. The encoder dataloader also works with `label=-1` placeholders for self-supervised pretraining.

Masked fixation modeling now uses span masking by default:

```text
mask_strategy: span
min_mask_span_events: 2
max_mask_span_events: 8
```

## EMS Filtered Variants

Two EMS variants are generated for QC sensitivity analysis:

### Strict QC

Definition:

- Keep only labeled subjects that pass subject-level hard QC.
- Remove events whose normalized coordinates are outside `[0, 1]`.
- Recompute transition features after event removal.

Output:

```text
data/processed/EMS/filtered/strict_qc/
```

Current summary:

```text
n_subjects: 150
label_counts: HC 77, SZ 73
n_events: 208436
out_of_range_coordinate_events: 0
```

### Clipped QC

Definition:

- Keep all 160 labeled EMS subjects.
- Clip `x_norm` and `y_norm` to `[0, 1]`.
- Recompute DVA coordinates and transition features after clipping.

Output:

```text
data/processed/EMS/filtered/clipped_qc/
```

Current summary:

```text
n_subjects: 160
label_counts: HC 80, SZ 80
n_events: 225159
clipped_coordinate_events: 4058
out_of_range_coordinate_events: 0
```

Both variants passed strict schema validation with labels required and normalized coordinates constrained to `[0, 1]`.

## Position-Feature Ablation

Dual-stream concat was rerun with `--no-segment-position` for original, strict-QC, and clipped-QC inputs.

Current test results using validation-selected best balanced-accuracy thresholds:

| Variant | Position Features | AUC | Balanced Accuracy | Sensitivity | Specificity |
| --- | --- | ---: | ---: | ---: | ---: |
| original | yes | 0.898 | 0.813 | 0.750 | 0.875 |
| original | no | 0.871 | 0.750 | 0.750 | 0.750 |
| strict_qc | yes | 0.938 | 0.833 | 0.733 | 0.933 |
| strict_qc | no | 0.844 | 0.733 | 1.000 | 0.467 |
| clipped_qc | yes | 0.836 | 0.781 | 0.688 | 0.875 |
| clipped_qc | no | 0.816 | 0.594 | 0.688 | 0.500 |

Interpretation:

- The model currently benefits from segment-position features.
- This is acceptable as an EMS-specific supervised model but risky for cross-dataset encoder pretraining.
- Encoder pretraining should initially avoid dataset-specific position features or treat them as optional ablation features.

## Encoder-Ready Table

The first encoder-ready EMS table uses the recommended default:

```text
dataset variant: clipped_qc
feature schema: encoder_original_13feature_core
```

Feature schema:

```text
configs/features/encoder_original_13feature_core.json
```

Output:

```text
data/processed/EMS/encoder_ready/clipped_qc_no_position/
  ems_encoder_events.csv
  feature_schema.json
  encoder_ready_summary.json
```

The `clipped_qc_no_position` directory name is historical. The original table includes screen-relative `x_norm` and `y_norm`; true no-position comparisons should use an explicit ablation schema such as `configs/features/encoder_trend_only_core.json`.

Current summary:

```text
n_subjects: 160
n_events: 225159
n_features: 13
label_counts: HC 80, SZ 80
features_with_nulls: none
content_fields_used_as_features: none
excluded global position features:
  - subject_event_index_norm
  - segment_index_norm
  - subject_event_index
```

The retained within-segment position feature is `event_index_in_segment_norm`, which describes event order within a local segment rather than global task progress.

## Encoder DataLoader

Dataset/DataLoader code:

```text
src/eyenet/data/encoder_dataset.py
```

Smoke-test command:

```powershell
python scripts/check_encoder_dataloader.py `
  --events data/processed/EMS/encoder_ready/clipped_qc_no_position/ems_encoder_events.csv `
  --schema configs/features/encoder_original_13feature_core.json `
  --split data/splits/EMS/ems_subject_split_60_20_20_seed42.csv `
  --batch-size 8
```

Verified batch output:

```text
x: [batch, seq_len, 13]
mask: [batch, seq_len]
label: [batch]
subject_id: list[str]
dataset_id: list[str]
```

Normalization is fit on train subjects only using median imputation followed by standard scaling. The DataLoader also supports an optional balanced train sampler for downstream supervised training.

## Supervised Encoder Smoke Test

The first supervised smoke-test model is implemented as:

```text
input projection: 13 -> 64
BiGRU encoder: hidden 64, bidirectional
attention pooling
classifier head: 128 -> 1
```

Code:

```text
src/eyenet/models/encoder.py
src/eyenet/training/supervised_encoder.py
scripts/train_supervised_encoder_smoke.py
```

Smoke-test output:

```text
experiments/encoder_smoke/ems_clipped_qc_no_position/
```

Current result:

```text
best_valid_auc: 0.855
test_auc: 0.828
test_balanced_accuracy: 0.719
test_sensitivity: 0.813
test_specificity: 0.625
best_epoch: 8
stopped_epoch: 16
```

Interpretation:

- The encoder/data/mask/checkpoint pipeline works.
- This is a smoke test, not the final model.
- The next modeling step is self-supervised pretraining, not tuning this supervised smoke model for maximum EMS performance.

## Masked Event Modeling Smoke Test

The first self-supervised pretraining task is masked event modeling:

```text
randomly mask valid event positions
encode the corrupted sequence
reconstruct the original standardized feature vector at masked positions
optimize masked MSE loss
```

Code:

```text
src/eyenet/training/masked_event_modeling.py
scripts/train_masked_event_modeling_smoke.py
```

Smoke-test output:

```text
experiments/encoder_pretraining/ems_masked_event_smoke/
```

Current result:

```text
mask_probability: 0.15
effective_mask_rate: about 0.12 because padding positions are excluded
best_epoch: 28
best_valid_loss: 0.307
test_loss: 0.311
```

Training behavior:

```text
valid_loss epoch 1: 0.935
valid_loss epoch 5: 0.499
valid_loss epoch 28: 0.307
```

Interpretation:

- The self-supervised encoder pretraining path works.
- The checkpoint stores both the full masked-event model and `encoder_state_dict`, so the encoder can be transferred into downstream classifier training.
- The next step is to expand pretraining beyond EMS using external datasets.

## Encoder Transfer Smoke Test

The masked-event pretrained encoder was transferred into the supervised EMS downstream classifier.

Code:

```text
src/eyenet/training/supervised_encoder.py
scripts/train_supervised_encoder_smoke.py
scripts/summarize_encoder_transfer_results.py
```

Experiments:

```text
experiments/encoder_smoke/ems_clipped_qc_no_position/
experiments/encoder_smoke/ems_clipped_qc_no_position_masked_pretrained_finetune/
experiments/encoder_smoke/ems_clipped_qc_no_position_masked_pretrained_frozen/
```

Summary output:

```text
experiments/encoder_smoke/encoder_transfer_summary.csv
```

Current test results using validation-selected best balanced-accuracy thresholds:

| Experiment | AUC | Balanced Accuracy | Sensitivity | Specificity | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| from scratch supervised | 0.828 | 0.719 | 0.813 | 0.625 | 0.743 |
| masked pretrained + fine-tune | 0.840 | 0.719 | 0.938 | 0.500 | 0.769 |
| masked pretrained frozen | 0.832 | 0.688 | 0.938 | 0.438 | 0.750 |

Interpretation:

- EMS-only masked-event pretraining gives a small AUC increase over from-scratch training.
- Fine-tuning is better than freezing the encoder, so the recommended downstream protocol is pretrain then fine-tune.
- The current EMS-only pretraining shifts the classifier toward higher sensitivity and lower specificity; this is useful for screening but still needs multi-dataset pretraining and threshold calibration.
