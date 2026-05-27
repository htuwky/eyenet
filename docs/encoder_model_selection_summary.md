# Encoder Model Selection Summary

Last updated: 2026-05-27

## Current Scope

This document summarizes the phase-1 encoder-pretraining model-selection result. It intentionally excludes dual-stream models, Transformer reruns, smoke tests, single-split screening, and non-aligned exploratory runs.

Included evidence:

- Supervised-only BiGRU encoder baseline.
- EMS-only masked event modeling (MEM), frozen and fine-tuned.
- Strict EMS-anchor aligned public-data MEM fusion runs, frozen and fine-tuned.
- Five downstream EMS subject splits, seeds `0,1,2,3,4`.

Primary threshold:

```text
validation-selected best balanced accuracy
```

Primary reporting metric:

```text
test balanced accuracy mean across five seeds
```

## Leakage Audit

The aligned split audit passed for all checked pretraining split rows:

```text
audit rows: 35
passed rows: 35
max overlap between downstream test subjects and MEM train: 0
max overlap between downstream test subjects and MEM valid: 0
expected overlap with MEM test: 32 per seed
```

Audit output:

```text
experiments/encoder_downstream/phase1_encoder_split_leakage_audit.csv
```

This supports the claim that EMS downstream test subjects were not seen during MEM train/validation for the aligned phase-1 encoder runs.

## Main Results

Generated from:

```text
experiments/encoder_downstream/phase1_encoder_summary.csv
```

| Experiment | Mode | Pretraining Data | AUC Mean | AUC Std | Balanced Accuracy Mean | Balanced Accuracy Std | Sensitivity Mean | Specificity Mean | F1 Mean |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EMS-only MEM BiGRU | fine-tune | EMS | 0.798 | 0.064 | 0.750 | 0.058 | 0.713 | 0.788 | 0.738 |
| EMS+GazeBase+CRCNS+OneStop BiGRU | fine-tune | EMS + GazeBase + CRCNS eye-1 + OneStop | 0.825 | 0.044 | 0.731 | 0.047 | 0.775 | 0.688 | 0.740 |
| EMS+CRCNS BiGRU | fine-tune | EMS + CRCNS eye-1 | 0.813 | 0.062 | 0.725 | 0.060 | 0.788 | 0.663 | 0.736 |
| EMS+GazeBase+CRCNS BiGRU | fine-tune | EMS + GazeBase + CRCNS eye-1 | 0.800 | 0.049 | 0.725 | 0.026 | 0.725 | 0.725 | 0.722 |
| EMS+CRCNS aligned BiGRU, seq3000 | fine-tune | EMS + CRCNS eye-1 | 0.809 | 0.038 | 0.706 | 0.042 | 0.738 | 0.675 | 0.711 |
| EMS+GazeBase+CRCNS+HBN BiGRU | fine-tune | EMS + GazeBase + CRCNS eye-1 + HBN | 0.795 | 0.071 | 0.700 | 0.087 | 0.750 | 0.650 | 0.716 |
| Supervised-only BiGRU | supervised | none | 0.784 | 0.069 | 0.700 | 0.114 | 0.800 | 0.600 | 0.732 |
| EMS+All-public BiGRU | fine-tune | EMS + GazeBase + CRCNS eye-1 + OneStop + HBN | 0.782 | 0.074 | 0.694 | 0.051 | 0.788 | 0.600 | 0.720 |

## Frozen Encoder Finding

Frozen probing remains consistently weaker than fine-tuning.

| Experiment | Mode | AUC Mean | Balanced Accuracy Mean | Sensitivity Mean | Specificity Mean |
| --- | --- | ---: | ---: | ---: | ---: |
| EMS+GazeBase+CRCNS+OneStop BiGRU | frozen | 0.740 | 0.644 | 0.800 | 0.488 |
| EMS+All-public BiGRU | frozen | 0.741 | 0.631 | 0.800 | 0.463 |
| EMS-only MEM BiGRU | frozen | 0.722 | 0.631 | 0.863 | 0.400 |
| EMS+GazeBase+CRCNS BiGRU | frozen | 0.721 | 0.613 | 0.788 | 0.438 |
| EMS+CRCNS BiGRU | frozen | 0.719 | 0.613 | 0.813 | 0.413 |
| EMS+CRCNS aligned BiGRU, seq3000 | frozen | 0.717 | 0.613 | 0.800 | 0.425 |
| EMS+GazeBase+CRCNS+HBN BiGRU | frozen | 0.715 | 0.613 | 0.788 | 0.438 |

Conclusion:

- Fine-tuning is the main downstream protocol.
- Frozen probing can be reported only as representation-probing evidence.

## Current Interpretation

The most defensible phase-1 conclusion is:

```text
Masked event modeling pretraining is useful mainly as initialization for supervised EMS fine-tuning. EMS-only MEM has the strongest mean balanced accuracy among the current strict five-seed runs. Adding public data is not monotonically beneficial: the EMS+GazeBase+CRCNS+OneStop run has the best mean AUC and competitive balanced accuracy, while HBN and all-public fusion do not improve the main balanced-accuracy criterion. Frozen encoder probing is consistently weaker than fine-tuning.
```

This means the current phase-1 primary model depends on the selection criterion:

- If the primary criterion is balanced accuracy, use `bigru64_mask045_fusion_ems_only`.
- If the primary criterion is AUC with public-data transfer evidence, use `bigru64_mask045_fusion_ems_gazebase_crcns_eye1_onestop` as the strongest public-data candidate.
- Do not claim that more public data always improves transfer.

## Results Excluded From Phase-1 Final Selection

Excluded from the phase-1 final table:

- Smoke tests.
- Single-split screening runs.
- Non-aligned mixed-pretraining runs.
- Transformer exploratory runs.
- Old encoder dual-stream fusion runs.
- New summary dual-stream design work and closure.

These runs can be used for engineering decisions or future-work motivation, but not as phase-1 final evidence.

## Next Status

Phase 1 is now a results/documentation task, not a training task.

Recommended next actions:

1. Use this document and `phase1_encoder_summary.csv` as the current source of truth.
2. Keep dual-stream work in separate phase-2 closure documents.
3. Update paper/report tables from the strict five-seed phase-1 summary only.
