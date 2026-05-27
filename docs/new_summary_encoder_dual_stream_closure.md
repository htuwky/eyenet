# New Summary Encoder Dual-Stream Closure

Last updated: 2026-05-27

## Purpose

This document closes the phase-2 strict subject-summary dual-stream branch:

```text
pretrained event encoder stream
+ content-agnostic subject-summary stream
+ concat/gated/residual-logit fusion
```

The goal was to test whether subject-level distribution statistics add stable information beyond the pretrained event-sequence encoder.

## Motivation

The old dual-stream branch used a segment-GRU macro stream. That design was useful as an engineering baseline but was not cleanly aligned with the project goal of content-, paradigm-, and device-agnostic screening.

The redesigned second stream therefore used strict subject-level summary features. These features intentionally excluded more paradigm-sensitive features such as absolute spatial layout summaries, total event counts, total segment counts, event-per-segment counts, and scanpath/spatial coverage summaries.

## Strict Summary Feature Families

The strict summary feature set retained:

```text
fixation duration distribution
short fixation ratio
long fixation ratio
saccade amplitude distribution
transition angle entropy
saccade angle entropy
```

It excluded:

```text
summary_n_segments
summary_n_events
events_per_segment
n_fixations
x_norm
y_norm
center_distance
center_bias
spatial range / bbox / coverage
scanpath_length
transition_missing
```

The strict feature table used:

```text
input file: data/processed/EMS/ems_subject_summary_features.csv
subjects: 208
labeled subjects: 160
class balance: 80 control / 80 patient
strict candidate features: 85
```

## Summary-Only Baseline

The strict summary-only baseline was run across EMS splits `0,1,2,3,4`.

| Model | AUC Mean | AUC Std | Balanced Accuracy Mean | Balanced Accuracy Std | Sensitivity Mean | Specificity Mean | F1 Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| strict summary-only logistic regression | 0.805 | 0.079 | 0.731 | 0.084 | 0.763 | 0.700 | 0.738 |
| strict summary-only SVM-RBF | 0.840 | 0.056 | 0.719 | 0.070 | 0.750 | 0.688 | 0.720 |

Interpretation:

```text
Strict subject-summary features contain meaningful predictive signal, even after removing obvious spatial-layout and task-length proxies.
```

## Dual-Stream Variants

Three new fusion mechanisms were tested:

```text
concat:
  concatenate encoder embedding and summary embedding

gated:
  learn a representation-level gate between encoder and summary embeddings

residual_logit:
  encoder_logit + alpha * summary_logit
  alpha <= 0.25
  summary_dim = 16
```

The residual-logit variant was added to address the concern that the summary stream should act only as an auxiliary correction rather than compete with the encoder stream at equal representation capacity.

## Main Results

Primary threshold:

```text
validation-selected best balanced accuracy
```

| Model | AUC Mean | AUC Std | Balanced Accuracy Mean | Balanced Accuracy Std | Sensitivity Mean | Specificity Mean | F1 Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EMS-only MEM BiGRU fine-tune | 0.798 | 0.064 | 0.750 | 0.058 | 0.713 | 0.788 | 0.738 |
| EMS+GazeBase+CRCNS+OneStop BiGRU fine-tune | 0.825 | 0.044 | 0.731 | 0.047 | 0.775 | 0.688 | 0.740 |
| strict summary-only logistic regression | 0.805 | 0.079 | 0.731 | 0.084 | 0.763 | 0.700 | 0.738 |
| strict summary + encoder dual-stream gated | 0.806 | 0.043 | 0.725 | 0.046 | 0.675 | 0.775 | 0.699 |
| strict summary-only SVM-RBF | 0.840 | 0.056 | 0.719 | 0.070 | 0.750 | 0.688 | 0.720 |
| strict summary + encoder dual-stream residual-logit | 0.795 | 0.047 | 0.713 | 0.041 | 0.775 | 0.650 | 0.727 |
| strict summary + encoder dual-stream concat | 0.809 | 0.045 | 0.706 | 0.065 | 0.738 | 0.675 | 0.701 |

## Closure Decision

The strict summary stream is informative as a standalone baseline, but none of the tested fusion mechanisms improved over the strongest encoder-only balanced-accuracy reference.

Final interpretation:

```text
Under the current EMS labeled sample size, subject-level strict summary features provide independent signal but do not produce stable gains when fused end-to-end with the pretrained event-sequence encoder. The main model should remain encoder-only MEM BiGRU fine-tuning. The summary stream can be retained as an interpretability baseline and future-work direction.
```

## Reporting Rule

Do not present the new dual-stream model as the primary model.

Use it as:

```text
exploratory negative evidence
future-work motivation
supporting evidence that strict content-agnostic summary statistics contain signal
```

Suggested paper wording:

```text
Subject-level content-agnostic summary features showed independent predictive signal, but direct fusion with the pretrained sequence encoder did not improve five-seed downstream performance. This suggests that the current encoder already captures much of the discriminative temporal-distribution information, and that future fusion may require stronger regularization or a larger labeled cohort.
```
