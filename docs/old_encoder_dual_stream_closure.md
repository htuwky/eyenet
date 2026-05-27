# Old Encoder Dual-Stream Closure

Last updated: 2026-05-27

## Purpose

This document closes the exploratory old dual-stream branch:

```text
pretrained encoder stream
+ segment-GRU macro stream
+ concat/gated fusion
```

This branch is not the current main model. It is an exploratory baseline used to decide whether dual-stream fusion deserves a cleaner phase-2 redesign.

## Current Result Files

Expected result layout:

```text
experiments/ems_encoder_dual_stream/
  bigru64_onestop_encoder_dual_seed0/
    concat/metrics.csv
    gated/metrics.csv
  ...
  bigru64_onestop_encoder_dual_seed4/
    concat/metrics.csv
    gated/metrics.csv
```

The result files currently appear complete for:

```text
seeds: 0,1,2,3,4
fusions: concat,gated
```

## Summary Command

Run this locally to generate the old dual-stream closure tables:

```powershell
$env:PYTHONPATH="src"
python scripts/summarize_old_encoder_dual_stream.py `
  --root experiments/ems_encoder_dual_stream `
  --phase1-summary experiments/encoder_downstream/phase1_encoder_summary.csv `
  --output experiments/ems_encoder_dual_stream/old_encoder_dual_stream_summary.csv
```

Expected outputs:

```text
experiments/ems_encoder_dual_stream/old_encoder_dual_stream_summary.csv
experiments/ems_encoder_dual_stream/old_encoder_dual_stream_summary_per_seed.csv
experiments/ems_encoder_dual_stream/old_encoder_dual_stream_summary_comparison.csv
```

## Completed Summary

The summary command was run successfully. The old encoder dual-stream branch has complete five-seed results for both fusion modes.

| Model | AUC Mean | AUC Std | Balanced Accuracy Mean | Balanced Accuracy Std | Sensitivity Mean | Specificity Mean | F1 Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EMS-only MEM BiGRU fine-tune | 0.798 | 0.064 | 0.750 | 0.058 | 0.713 | 0.788 | 0.738 |
| EMS+GazeBase+CRCNS+OneStop BiGRU fine-tune | 0.825 | 0.044 | 0.731 | 0.047 | 0.775 | 0.688 | 0.740 |
| Old encoder dual-stream gated | 0.811 | 0.074 | 0.725 | 0.105 | 0.838 | 0.613 | 0.756 |
| Supervised-only BiGRU | 0.784 | 0.069 | 0.700 | 0.114 | 0.800 | 0.600 | 0.732 |
| Old encoder dual-stream concat | 0.799 | 0.058 | 0.694 | 0.078 | 0.675 | 0.713 | 0.688 |

## Closure Decision

Old dual-stream gated outperformed old dual-stream concat:

```text
old dual gated balanced accuracy: 0.725
old dual concat balanced accuracy: 0.694
```

However, old dual-stream gated did not outperform the phase-1 encoder-only main reference:

```text
EMS-only MEM BiGRU fine-tune balanced accuracy: 0.750
old dual gated balanced accuracy: 0.725
```

It also had larger seed-to-seed variance:

```text
EMS-only MEM BiGRU fine-tune balanced accuracy std: 0.058
EMS+GazeBase+CRCNS+OneStop BiGRU fine-tune balanced accuracy std: 0.047
old dual gated balanced accuracy std: 0.105
```

Final interpretation:

```text
The exploratory encoder dual-stream model did not improve over the phase-1 encoder-only MEM fine-tuning baseline. Although gated fusion outperformed concat fusion and improved sensitivity, it reduced specificity and showed larger seed-to-seed variance. We therefore retain old dual-stream as an engineering baseline and defer the next dual-stream iteration to a content-agnostic subject-summary branch.
```

## Interpretation Rules

Use the old dual-stream result only as exploratory evidence.

The branch is worth continuing only if:

```text
old dual-stream balanced_accuracy_mean > best phase-1 encoder balanced_accuracy_mean
```

or if it shows clearly better sensitivity/specificity balance at similar balanced accuracy.

If it does not improve over the phase-1 encoder reference, the correct conclusion is:

```text
The old segment-GRU macro stream validates the fusion engineering path, but it does not provide enough evidence to replace the encoder-only phase-1 model. Phase 2 should redesign the second stream as a content-agnostic subject-summary MLP stream.
```

## Final Status

Old dual-stream is closed as an exploratory baseline.

Do not use it as the current primary model.

Use it only to motivate the phase-2 redesign:

```text
pretrained encoder stream + content-agnostic subject-summary MLP stream
```
