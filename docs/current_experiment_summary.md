# Current Experiment Summary

Last updated: 2026-05-27

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

## Phase-1 Aligned Model-Selection Result

The current source of truth is:

```text
experiments/encoder_downstream/phase1_encoder_summary.csv
experiments/encoder_downstream/phase1_encoder_split_leakage_audit.csv
docs/encoder_model_selection_summary.md
```

Strict aligned split audit:

```text
checked split rows: 35
passed rows: 35
max downstream-test overlap with MEM train: 0
max downstream-test overlap with MEM valid: 0
```

Primary threshold: validation-selected best balanced accuracy.

| Experiment | Mode | Seeds | AUC Mean | AUC Std | Balanced Accuracy Mean | Balanced Accuracy Std | Sensitivity Mean | Specificity Mean |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EMS-only MEM BiGRU | fine-tune | 0-4 | 0.798 | 0.064 | 0.750 | 0.058 | 0.713 | 0.788 |
| EMS+GazeBase+CRCNS+OneStop BiGRU | fine-tune | 0-4 | 0.825 | 0.044 | 0.731 | 0.047 | 0.775 | 0.688 |
| EMS+CRCNS BiGRU | fine-tune | 0-4 | 0.813 | 0.062 | 0.725 | 0.060 | 0.788 | 0.663 |
| EMS+GazeBase+CRCNS BiGRU | fine-tune | 0-4 | 0.800 | 0.049 | 0.725 | 0.026 | 0.725 | 0.725 |
| EMS+CRCNS aligned BiGRU, seq3000 | fine-tune | 0-4 | 0.809 | 0.038 | 0.706 | 0.042 | 0.738 | 0.675 |
| EMS+GazeBase+CRCNS+HBN BiGRU | fine-tune | 0-4 | 0.795 | 0.071 | 0.700 | 0.087 | 0.750 | 0.650 |
| Supervised-only BiGRU | supervised | 0-4 | 0.784 | 0.069 | 0.700 | 0.114 | 0.800 | 0.600 |
| EMS+All-public BiGRU | fine-tune | 0-4 | 0.782 | 0.074 | 0.694 | 0.051 | 0.788 | 0.600 |

Decision:

- Use `bigru64_mask045_fusion_ems_only` as the strongest phase-1 balanced-accuracy model.
- Use `bigru64_mask045_fusion_ems_gazebase_crcns_eye1_onestop` as the strongest public-data AUC candidate.
- Do not claim that adding more public data monotonically improves transfer.
- Do not use frozen encoder probing as the main downstream method.

## Current Best Statement

The defensible current statement is:

```text
Across five EMS stratified subject splits, masked event modeling followed by supervised fine-tuning improves the encoder pipeline over supervised-only training in the strongest phase-1 configuration. EMS-only MEM has the highest mean balanced accuracy, while EMS+GazeBase+CRCNS+OneStop has the highest mean AUC among public-data fusion candidates. Public data is therefore useful as transfer evidence, but more sources do not automatically improve the primary balanced-accuracy criterion. Frozen encoder probing is consistently weaker than fine-tuning.
```

## Phase-2 Dual-Stream Closure

Dual-stream experiments were run after the phase-1 encoder line to test whether a second, content-agnostic subject-summary stream adds stable information beyond the pretrained event encoder.

Current dual-stream source files:

```text
experiments/ems_encoder_dual_stream/old_encoder_dual_stream_summary.csv
experiments/ems_subject_summary_baseline_strict/summary.csv
experiments/ems_summary_encoder_dual_stream/summary.csv
docs/old_encoder_dual_stream_closure.md
docs/new_summary_encoder_dual_stream_closure.md
```

Primary threshold: validation-selected best balanced accuracy.

| Model | Seeds | AUC Mean | AUC Std | Balanced Accuracy Mean | Balanced Accuracy Std | Sensitivity Mean | Specificity Mean | F1 Mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EMS-only MEM BiGRU fine-tune | 0-4 | 0.798 | 0.064 | 0.750 | 0.058 | 0.713 | 0.788 | 0.738 |
| EMS+GazeBase+CRCNS+OneStop BiGRU fine-tune | 0-4 | 0.825 | 0.044 | 0.731 | 0.047 | 0.775 | 0.688 | 0.740 |
| Strict summary-only logistic regression | 0-4 | 0.805 | 0.079 | 0.731 | 0.084 | 0.763 | 0.700 | 0.738 |
| Strict summary + encoder dual-stream gated | 0-4 | 0.806 | 0.043 | 0.725 | 0.046 | 0.675 | 0.775 | 0.699 |
| Strict summary-only SVM-RBF | 0-4 | 0.840 | 0.056 | 0.719 | 0.070 | 0.750 | 0.688 | 0.720 |
| Strict summary + encoder dual-stream residual-logit | 0-4 | 0.795 | 0.047 | 0.713 | 0.041 | 0.775 | 0.650 | 0.727 |
| Strict summary + encoder dual-stream concat | 0-4 | 0.809 | 0.045 | 0.706 | 0.065 | 0.738 | 0.675 | 0.701 |
| Old encoder + segment-GRU dual-stream gated | 0-4 | 0.811 | 0.074 | 0.725 | 0.105 | 0.838 | 0.613 | 0.756 |
| Old encoder + segment-GRU dual-stream concat | 0-4 | 0.799 | 0.058 | 0.694 | 0.078 | 0.675 | 0.713 | 0.688 |

Interpretation:

- Strict subject-summary features have independent predictive signal.
- However, all tested fusion mechanisms failed to improve over the strongest encoder-only balanced-accuracy reference:
  - old segment-GRU macro-stream concat/gated fusion
  - new strict subject-summary concat fusion
  - new strict subject-summary gated fusion
  - new strict subject-summary residual-logit auxiliary fusion
- The current evidence therefore supports keeping `encoder-only MEM BiGRU fine-tuning` as the main model.
- Dual-stream results should be reported as exploratory negative evidence or future-work motivation, not as the main model.

## Next Experimental Order

The next phase separates `research_profile` and `deployment_profile` without training every experiment twice.

```text
research_profile
  publication-facing AUC/F1/balanced-accuracy optimization

deployment_profile
  hospital screening prototype focused on stability, QC, calibration, and maintainability
```

Shared rule:

```text
train once
save checkpoint and predictions
evaluate under both profiles
only final candidates receive profile-specific optimization
```

Immediate order:

1. Recompute existing encoder predictions under F1-oriented and sensitivity-constrained validation thresholds.
2. Keep the current encoder-only MEM BiGRU fine-tune result as the phase-1 main model and deployment baseline.
3. Start research-profile BiGRU hyperparameter ablation only after threshold-only gains are checked.
4. Treat dual-stream as closed exploratory negative evidence under the current EMS label scale.
5. Keep Transformer experiments as future exploratory work unless the research-profile BiGRU ablation saturates.

See:

```text
docs/model_profiles_and_next_steps.md
```
