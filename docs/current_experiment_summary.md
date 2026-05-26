# Current Experiment Summary

Last updated: 2026-05-25

This document summarizes the current EyeNet encoder-pretraining state after dataset screening, numerical-stability fixes, and the first controlled aligned multi-seed model-selection run.

## Core Decision

EyeNet uses fixation-event sequences as the universal model input. Raw gaze datasets are adapter inputs only and must be converted to fixation events before encoder training.

The active universal encoder feature schema is:

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

No image, video, task-name, stimulus-category, AOI, or content-derived feature is used as an encoder feature.

## Dataset Processing Status

| Dataset | Status | Subjects | Fixation Events | Notes |
| --- | --- | ---: | ---: | --- |
| EMS | Complete | 160 | 225,159 | Main SZ/HC downstream benchmark. Uses clipped-QC no-position encoder table. |
| HBN | Complete | 1,244 usable after QC | 1,684,382 after QC | Public unlabeled MEM source. Technically integrated; not currently the best transfer source. |
| GazeBase | Complete | 322 | 843,517 | Video tasks `VD1,VD2`; high-specificity single-split behavior. |
| OneStop | Complete | 360 | 2,042,834 | Reading fixation corpus; technically integrated, but less promising for EMS transfer than CRCNS in current runs. |
| CRCNS eye-1 | Complete | 16 | 67,172 | Natural movie-viewing fixations; currently the most useful public source for EMS transfer experiments. |
| Saliency4ASD | Deferred | TBD | TBD | Pseudo-subject/session structure and ASD labels make it lower priority. Do not merge ASD labels with SZ/HC. |

## Protocol Correction

The project now distinguishes exploratory mixed pretraining from strict model-selection pretraining.

For final model selection, the self-supervised mixed pretraining split is **aligned to the EMS downstream split**. EMS subjects assigned to downstream test are also assigned to MEM test, so they are not seen during MEM train. Non-EMS datasets are split independently by subject.

Active aligned split generator:

```text
scripts/create_aligned_self_supervised_subject_split.py
```

Active aligned multi-seed runner:

```text
scripts/run_aligned_encoder_multiseed_ablation.py
```

This means older non-aligned mixed-pretraining runs are retained as exploratory evidence but should not be used as final model-selection results.

## Multi-Seed EMS Baseline

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
- Frozen probing underperforms from-scratch training, so MEM is useful mainly as initialization for task-specific fine-tuning.
- EMS remains small; results should be reported as initial evidence, not clinical performance.

## Public Dataset Single-Split Screening

Single split screening showed that public data is not automatically helpful.

| Pretraining Source | Fine-Tune? | Test AUC | Balanced Accuracy | Sensitivity | Specificity | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| none | supervised from scratch | 0.828 | 0.719 | 0.813 | 0.625 | Seed42 supervised encoder baseline. |
| EMS-only MEM | yes | 0.891 | 0.750 | 0.938 | 0.563 | Strong EMS-only initialization. |
| HBN+EMS MEM | yes | 0.859 | 0.719 | 0.875 | 0.563 | Technically works, but no gain over EMS-only. |
| GazeBase+EMS MEM | yes | 0.863 | 0.813 | 0.750 | 0.875 | More conservative high-specificity operating point. |
| OneStop+EMS MEM | yes | 0.891 | 0.719 | 0.938 | 0.500 | Good AUC, weak specificity under selected threshold. |
| CRCNS eye-1+EMS MEM | yes | 0.910 | 0.781 | 0.750 | 0.813 | Best single-source public AUC candidate. |
| GazeBase+CRCNS+EMS MEM | yes | 0.898 | 0.781 | 0.938 | 0.625 | Fusion did not clearly beat CRCNS-only. |

Interpretation:

- CRCNS eye-1 is currently the most promising public source because it matches the natural viewing/video-like setting better than reading or HBN.
- GazeBase remains useful as a high-specificity reference source.
- HBN and OneStop are technically integrated but not current priority pretraining sources.

## Current Model

Primary encoder:

```text
input: [batch, sequence_length, 13]
projection: Linear(13 -> 64) + LayerNorm + ReLU + Dropout
temporal encoder: 1-layer bidirectional GRU, hidden_dim=64
event embedding: 128
pooling: masked attention pooling
classifier: supervised binary head
MEM head: reconstruct masked 13-dim event features
```

Numerical-stability fixes already applied:

- MEM masking uses a learnable mask token instead of replacing masked values with `0.0`.
- MEM reconstruction operates in the dataloader-standardized feature space, avoiding raw scale domination.
- YAML config loading now uses `pyyaml` rather than a local partial parser.

Current fixed training protocol:

```text
optimizer: AdamW
learning_rate: 1e-3
weight_decay: 1e-4
batch_size: 8
max_seq_len: 1500
gradient_clip_norm: 5.0
mask_strategy: span
mask_span_events: 2-8
```

## Aligned Model-Selection Result

Strict aligned protocol:

```text
pretraining data: EMS + CRCNS eye-1
anchor dataset: EMS
seeds: 0,1,2,3,4
batch_size: 8
max_seq_len: 1500
encoder: BiGRU64 attention
mask_probability: 0.45
compared dropout: 0.3 vs 0.4
downstream mode: supervised fine-tuning
```

Primary threshold: validation-selected best balanced accuracy.

| Experiment | Mode | Seeds | AUC Mean | AUC Std | Balanced Accuracy Mean | Balanced Accuracy Std | Sensitivity Mean | Specificity Mean | Interpretation |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| CRCNS+EMS MEM, dropout 0.3 | fine-tune | 0-4 | 0.813 | 0.062 | 0.725 | 0.060 | 0.788 | 0.663 | Current primary model. Best overall AUC/BA balance. |
| CRCNS+EMS MEM, dropout 0.4 | fine-tune | 0-4 | 0.804 | 0.043 | 0.719 | 0.049 | 0.713 | 0.725 | More conservative; higher specificity, lower sensitivity. |
| CRCNS+EMS MEM, dropout 0.3 | frozen | 0-4 | 0.719 | 0.072 | 0.613 | 0.098 | 0.813 | 0.413 | Frozen probing is not competitive. |
| CRCNS+EMS MEM, dropout 0.4 | frozen | 0-4 | 0.709 | 0.067 | 0.631 | 0.060 | 0.763 | 0.500 | Frozen probing remains weak. |

Decision:

- Use `bigru64_ems_crcns_mask045_aligned` as the current primary model.
- Keep `bigru64_ems_crcns_mask045_dropout04_aligned` as a high-specificity secondary candidate.
- Do not use frozen encoder as the main downstream method.
- Do not include single-seed `mask0.30` or `batch16` runs in final comparisons.

## Current Best Statement

The defensible current statement is:

```text
Across five EMS stratified subject splits, EMS-only masked event modeling followed by supervised fine-tuning improves downstream SZ/HC classification over from-scratch encoder training. After public dataset screening and strict EMS-anchor split alignment, CRCNS eye-1 plus EMS pretraining with a BiGRU64 encoder, span masking at probability 0.45, dropout 0.3, and supervised fine-tuning is the current primary model-selection result. The improvement over EMS-only is not yet decisive, but CRCNS provides the best public-data transfer route tested so far. Frozen encoder probing is consistently weaker than fine-tuning.
```

## Next Experimental Order

1. Stop broad architecture search for now.
2. Compare the aligned CRCNS result directly against EMS-only MEM and from-scratch in one paper table.
3. Update paper draft and project handoff docs with the aligned protocol.
4. Run one final confirmation only if needed:
   - `max_seq_len=3000` for the primary dropout 0.3 model, because 16GB GPU memory can be used there without changing batch size in the main protocol.
5. Move next to engineering cleanup or the next carefully selected public dataset only after the current model-selection result is documented.
