# Next Stage Dual-Stream Plan and Closure

Last updated: 2026-05-27

## Status

Dual-stream modeling has now been tested as phase-2 exploratory work. It should not drive the current phase-1 paper/report narrative.

The old dual-stream work remains useful as an engineering baseline, but it is not the current primary model because it did not improve over the phase-1 encoder-only reference.

The new strict subject-summary dual-stream branch was also tested and did not improve over the strongest encoder-only balanced-accuracy reference.

## Phase-2 Goal

Phase 2 should test whether a second stream adds information that is genuinely complementary to the pretrained event encoder.

The preferred design is:

```text
event stream:
  EMS event sequence
  -> pretrained BiGRU MEM encoder
  -> attention pooling
  -> local temporal embedding

summary stream:
  content-agnostic subject-level behavioral statistics
  -> small MLP
  -> global behavior embedding

fusion:
  concat, gated, or residual-logit fusion
  -> EMS classifier
```

## Design Constraint

The second stream must remain content-, paradigm-, and device-agnostic.

The strict summary branch retained:

- Fixation duration distribution.
- Saccade amplitude distribution, preferably normalized.
- Direction entropy or angular dispersion.

The strict summary branch excluded:

- Total event or segment counts.
- Events per segment.
- Number of fixations.
- Normalized x/y position summaries.
- Center-distance and center-bias summaries.
- Spatial range, bounding-box, coverage, and scanpath-length summaries.
- Transition-missing summaries.

## Completed Phase-2 Comparisons

The following comparisons were run on the same EMS splits:

- `encoder-only`
- `summary-only`
- `old encoder + segment-GRU stream`
- `new encoder + strict subject-summary MLP stream`, concat
- `new encoder + strict subject-summary MLP stream`, gated
- `new encoder + strict subject-summary MLP stream`, residual-logit auxiliary correction

Primary comparison metric:

```text
balanced_accuracy_mean across seeds 0,1,2,3,4
```

Secondary metrics:

- AUC mean/std.
- Sensitivity/specificity balance.
- F1.
- Seed-to-seed stability.

## Completed Results

Primary threshold: validation-selected best balanced accuracy.

| Model | AUC Mean | AUC Std | Balanced Accuracy Mean | Balanced Accuracy Std | Sensitivity Mean | Specificity Mean | F1 Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EMS-only MEM BiGRU fine-tune | 0.798 | 0.064 | 0.750 | 0.058 | 0.713 | 0.788 | 0.738 |
| EMS+GazeBase+CRCNS+OneStop BiGRU fine-tune | 0.825 | 0.044 | 0.731 | 0.047 | 0.775 | 0.688 | 0.740 |
| strict summary-only logistic regression | 0.805 | 0.079 | 0.731 | 0.084 | 0.763 | 0.700 | 0.738 |
| new strict summary + encoder gated | 0.806 | 0.043 | 0.725 | 0.046 | 0.675 | 0.775 | 0.699 |
| new strict summary + encoder residual-logit | 0.795 | 0.047 | 0.713 | 0.041 | 0.775 | 0.650 | 0.727 |
| new strict summary + encoder concat | 0.809 | 0.045 | 0.706 | 0.065 | 0.738 | 0.675 | 0.701 |
| old encoder + segment-GRU gated | 0.811 | 0.074 | 0.725 | 0.105 | 0.838 | 0.613 | 0.756 |
| old encoder + segment-GRU concat | 0.799 | 0.058 | 0.694 | 0.078 | 0.675 | 0.713 | 0.688 |

## Current Recommendation

Do not continue dual-stream tuning in the current project phase.

Use the dual-stream branch as exploratory negative evidence:

```text
Strict subject-summary features carry independent predictive signal, but all tested fusion mechanisms failed to improve over the encoder-only MEM BiGRU balanced-accuracy reference.
```

The current mainline remains:

```text
encoder-only MEM BiGRU fine-tuning
```

The dual-stream idea can be revisited only with a larger labeled cohort or a stronger regularization/fusion design.
