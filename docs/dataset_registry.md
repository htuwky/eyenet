# Dataset Registry

This registry defines how each dataset should be used before encoder pretraining. The key rule is that all datasets must be converted into the shared EyeNet event schema before training.

## Usage Summary

| Dataset | Current Status | Main Use | Supervised Disease Task | Encoder Pretraining | Notes |
| --- | --- | --- | --- | --- | --- |
| EMS | Available locally | Downstream benchmark | SZ vs HC | Yes, but avoid leaking test labels into model selection | Current main complete clinical dataset. |
| GazeBase | Adapter complete | General eye-movement encoder pretraining | No | Yes | Raw DVA gaze converted to fixation events for video tasks `VD1,VD2`; all 322 subjects pass self-supervised QC. |
| CRCNS eye-1 | Pending local verification | Video-viewing domain pretraining | No | Yes | Useful because the target hospital paradigm is video viewing. |
| HBN | Adapter complete | Youth-domain pretraining or auxiliary phenotype learning | Variable | Yes | Raw gaze converted to fixation events; 1,244 subjects pass self-supervised QC. Use labels only after explicit phenotype review. |
| Saliency4ASD | Available locally, adapter pending | Auxiliary atypical-attention task | ASD vs TD only | Cautious | Parse fixation/scanpath files. Do not merge ASD labels with SZ labels. |

## Dataset Roles

### EMS

EMS is the current downstream benchmark. It is used to evaluate schizophrenia-vs-control screening with a fixed subject-level train/validation/test split.

Allowed uses:

- Supervised SZ vs HC training.
- Downstream fine-tuning.
- Held-out EMS test reporting.

Not allowed:

- Using image identity or image content as model input.
- Selecting final thresholds on the test split.

### GazeBase

GazeBase should be used for self-supervised encoder pretraining. It should not define the clinical decision boundary because it does not provide the target psychiatric label.

Recommended tasks:

- Masked event modeling.
- Segment contrastive learning.
- Device/task robust representation learning.

### CRCNS eye-1

CRCNS eye-1 should be used to expose the encoder to natural video-viewing behavior. This is valuable because the hospital setting is expected to use video paradigms.

Recommended tasks:

- Masked event modeling.
- Temporal order prediction.
- Video-domain alignment.

### HBN

HBN is relevant because the target application is adolescent screening. However, it must be handled cautiously because the label structure and task metadata need review before supervised use.

Recommended tasks:

- Youth-domain self-supervised pretraining.
- Auxiliary phenotype prediction only after label review.

### Saliency4ASD

Saliency4ASD may help the encoder learn atypical visual-attention patterns, but ASD/TD labels must remain an auxiliary task and must not be merged into SZ/HC classification.

Recommended tasks:

- Auxiliary ASD vs TD classification.
- Domain-robust representation learning.

## Common Ingestion Rule

Every dataset must pass:

```text
raw gaze or official fixation table
-> fixation detection when needed
-> dataset adapter
-> shared event table
-> schema validation
-> QC report
-> encoder-ready sequence table
```

Datasets that cannot provide enough metadata for DVA conversion may still be used with normalized-coordinate features, but this limitation must be recorded in the dataset config.

## Encoder Training Policy

Clinical labels from different disorders are not merged into one binary target. Multi-dataset encoder training should first use self-supervised tasks. Supervised disease labels are added later as dataset-specific heads or auxiliary tasks.
