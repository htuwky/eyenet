# Encoder Model Selection Summary

Last updated: 2026-05-25

## Selected Primary Model

Current primary model:

```text
experiment_group: bigru64_ems_crcns_mask045_aligned
encoder: BiGRU attention
projection_dim: 64
hidden_dim: 64
attention_dim: 64
mask_probability: 0.45
dropout: 0.3
batch_size: 8
max_seq_len: 1500
pretraining: EMS + CRCNS eye-1
split protocol: EMS-anchor aligned self-supervised split
downstream: EMS supervised fine-tuning
```

Primary reason:

- It has the best aligned five-seed balance of AUC and balanced accuracy.
- It keeps sensitivity higher than the dropout 0.4 candidate.
- It avoids the stricter but lower-sensitivity behavior of dropout 0.4.

## Main Comparison

Primary threshold: validation-selected best balanced accuracy.

| Experiment | Mode | Seeds | AUC Mean | AUC Std | Balanced Accuracy Mean | Balanced Accuracy Std | Sensitivity Mean | Specificity Mean |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| from_scratch | supervised | 5 | 0.784 | 0.069 | 0.700 | 0.114 | 0.800 | 0.600 |
| EMS-only MEM | fine-tune | 5 | 0.832 | 0.065 | 0.763 | 0.052 | 0.800 | 0.725 |
| EMS+CRCNS MEM, dropout 0.3 | fine-tune | 5 | 0.813 | 0.062 | 0.725 | 0.060 | 0.788 | 0.663 |
| EMS+CRCNS MEM, dropout 0.4 | fine-tune | 5 | 0.804 | 0.043 | 0.719 | 0.049 | 0.713 | 0.725 |

Interpretation:

- EMS-only MEM remains the strongest mean result among the currently comparable five-seed tables.
- Strict aligned CRCNS+EMS pretraining is useful but does not yet decisively beat EMS-only MEM.
- CRCNS is still the best public-data route tested so far and should remain the primary public source candidate.
- Dropout 0.4 shifts toward specificity at the cost of sensitivity; it is a secondary high-specificity candidate.

## Frozen Encoder Finding

Frozen probing is consistently weaker than fine-tuning.

| Experiment | Mode | Seeds | AUC Mean | Balanced Accuracy Mean | Sensitivity Mean | Specificity Mean |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| EMS-only MEM | frozen | 5 | 0.716 | 0.619 | 0.813 | 0.425 |
| EMS+CRCNS MEM, dropout 0.3 | frozen | 5 | 0.719 | 0.613 | 0.813 | 0.413 |
| EMS+CRCNS MEM, dropout 0.4 | frozen | 5 | 0.709 | 0.631 | 0.763 | 0.500 |

Conclusion:

- Do not use frozen encoder probing as the main downstream strategy.
- Report frozen results only as representation-probing evidence.

## Results Excluded From Final Model Selection

The following runs are useful for exploration but should not enter final tables:

- `mask0.30 aligned seed0`: only one aligned seed and clearly weaker.
- `batch16 aligned seed0`: batch size differs from the fixed protocol.
- Non-aligned mixed-pretraining runs: target EMS test subjects may have been seen during MEM pretraining.
- Single-split seed42 public screening runs: useful for source screening, not final model selection.

## Next Decision

Do not continue broad model search immediately.

Recommended next steps:

1. Use `bigru64_ems_crcns_mask045_aligned` as the current primary model.
2. Keep `bigru64_ems_crcns_mask045_dropout04_aligned` as a high-specificity secondary model.
3. Update paper/report tables to distinguish:
   - EMS-only multi-seed baseline
   - public-source screening
   - strict aligned model selection
4. If one additional experiment is needed, run only `max_seq_len=3000` on the primary model while keeping batch size fixed at 8.
