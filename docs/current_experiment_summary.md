# Current Experiment Summary

Last updated: 2026-05-23

This document summarizes the current EyeNet encoder-pretraining state before moving into additional public datasets and model ablation.

## Core Decision

The project uses fixation-event sequences as the universal input. Raw gaze datasets are adapter inputs only and must be converted to fixation events before encoder training.

The current universal encoder feature schema is:

```text
encoder_no_position_core
n_features: 13
```

Feature columns:

```text
x_norm
y_norm
duration_ms
log_duration_ms
saccade_dx_norm
saccade_dy_norm
saccade_amplitude_norm
saccade_angle_sin
saccade_angle_cos
transition_missing
is_first_event_in_segment
is_last_event_in_segment
event_index_in_segment_norm
```

No image, video, task-name, stimulus-category, or content-derived feature is used as an encoder feature.

## Dataset Processing Status

| Dataset | Status | Subjects | Fixation Events | Notes |
| --- | --- | ---: | ---: | --- |
| EMS | Complete | 160 | 225,159 | Main SZ/HC downstream benchmark. Uses clipped-QC no-position encoder table. |
| HBN | Complete | 1,244 usable after QC | 1,684,382 after QC | Raw gaze converted with I-DT. Used for public unlabeled MEM pretraining. |
| GazeBase | Complete | 322 | 843,517 | Raw DVA gaze converted with I-DT. Current adapter uses video tasks `VD1,VD2`. |
| Saliency4ASD | Pending | TBD | TBD | Available locally, not converted yet. Labels must not be merged with EMS SZ/HC. |
| CRCNS eye-1 | Pending | TBD | TBD | Local status still needs verification before adapter work. |

## GazeBase Adapter Result

GazeBase full conversion:

```text
n_subject_zips: 881
n_subjects: 322
n_trials: 3524
n_fixation_events: 843517
tasks: VD1, VD2
valid_sample_rate_mean: 0.955
fixation_duration_median_ms: 155
```

GazeBase schema validation:

```text
passed: true
normalized_coordinate_issues: none
structural_errors: none
```

GazeBase QC:

```text
n_hard_qc_pass: 322 / 322
n_usable_for_self_supervised_pretraining: 322 / 322
```

## Multi-Seed EMS Encoder Baseline

Five EMS stratified subject splits were run for:

- from-scratch supervised encoder
- EMS-only MEM frozen encoder
- EMS-only MEM fine-tuned encoder

Primary threshold: validation-selected best balanced accuracy.

| Experiment | Test AUC Mean | Test AUC Std | Balanced Accuracy Mean | Balanced Accuracy Std | Sensitivity Mean | Specificity Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| from_scratch | 0.784 | 0.069 | 0.700 | 0.114 | 0.800 | 0.600 |
| ems_mem_frozen | 0.716 | 0.059 | 0.619 | 0.081 | 0.813 | 0.425 |
| ems_mem_finetune | 0.832 | 0.065 | 0.763 | 0.052 | 0.800 | 0.725 |

Interpretation:

- EMS-only masked event modeling improves mean AUC and balanced accuracy after supervised fine-tuning.
- Frozen linear probing underperforms from-scratch training, so MEM is useful mainly as initialization for task-specific fine-tuning.
- EMS remains small; results should be reported as initial evidence, not clinical performance.

## Public Dataset Pretraining Results

Single fixed EMS split `seed42` downstream comparison:

| Pretraining Source | Fine-Tune? | Test AUC | Balanced Accuracy | Sensitivity | Specificity | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| none | supervised from scratch | 0.828 | 0.719 | 0.813 | 0.625 | Baseline encoder. |
| EMS-only MEM | yes | 0.891 | 0.750 | 0.938 | 0.563 | Highest single-split AUC so far. |
| EMS-only MEM | frozen | 0.848 | 0.688 | 0.938 | 0.438 | Encoder alone is not enough. |
| HBN+EMS MEM | yes | 0.859 | 0.719 | 0.875 | 0.563 | Did not outperform EMS-only. |
| HBN+EMS MEM | frozen | 0.824 | 0.719 | 0.938 | 0.500 | Some transfer, weak specificity. |
| GazeBase+EMS MEM | yes | 0.863 | 0.813 | 0.750 | 0.875 | More conservative, high specificity. |
| GazeBase+EMS MEM | frozen | 0.863 | 0.688 | 0.938 | 0.438 | Ranking transfers, threshold boundary needs fine-tuning. |

Interpretation:

- EMS-only MEM is the current best AUC route.
- GazeBase+EMS is not higher AUC than EMS-only on seed42, but it improves specificity and balanced accuracy on that split.
- HBN is technically integrated but is not currently the best pretraining source.
- Public data is not assumed useful by default; each source must improve EMS downstream or provide a clear complementary operating point.

## Current Model

Current encoder:

```text
input: [batch, sequence_length, 13]
projection: Linear(13 -> 64) + LayerNorm + ReLU + Dropout(0.3)
temporal encoder: 1-layer bidirectional GRU, hidden_dim=64
event embedding: 128
pooling: masked attention pooling
classifier: supervised binary head
MEM head: reconstruct masked 13-dim event features
```

Current training defaults:

```text
optimizer: AdamW
learning_rate: 1e-3
weight_decay: 1e-4
batch_size: 8
max_seq_len: 1500
dropout: 0.3
gradient_clip_norm: 5.0
mask_probability: 0.30
mask_strategy: span
mask_span_events: 2-8
```

These are baseline settings, not final optimized hyperparameters.

## Next Experimental Order

Do not start broad hyperparameter search yet. The correct order is:

1. Finish remaining dataset source screening:
   - Saliency4ASD
   - CRCNS eye-1, if local raw files are available
2. Run each new source through the fixed baseline configuration:
   - adapter
   - schema validation
   - QC
   - encoder-ready table
   - source+EMS MEM
   - EMS downstream fine-tune/frozen
3. Select one or two promising pretraining routes.
4. Run controlled ablations:
   - hidden_dim: 32 / 64 / 128
   - dropout: 0.1 / 0.3 / 0.5
   - max_seq_len: 1000 / 1500 / 2500
   - optional later: GRU depth and Transformer encoder
5. Report model selection by multi-seed mean/std, not by one best split.

## Current Best Statement

The defensible current statement is:

```text
Across five EMS stratified subject splits, EMS-only masked event modeling followed by supervised fine-tuning improved downstream SZ/HC classification over from-scratch encoder training. Public datasets HBN and GazeBase have been integrated into the same fixation-event pipeline; initial single-split results suggest GazeBase may provide a more conservative high-specificity operating point, while HBN did not improve over EMS-only MEM.
```
