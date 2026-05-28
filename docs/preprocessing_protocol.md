# Preprocessing Protocol

## Principles

- Raw data is immutable.
- Every dataset is converted into a shared fixation-saccade event space.
- Dataset-specific fields are preserved only when useful for auditing or optional analysis.
- Train/test splits are always subject-level.

## Common Steps

1. Validate file counts, labels, and required columns.
2. Normalize coordinates or convert to DVA when screen geometry is available.
3. Remove invalid or out-of-range samples/events.
4. Interpolate only short missing gaps for raw gaze data.
5. Smooth raw gaze before event detection when appropriate.
6. Detect or load fixation/saccade events.
7. Compute transition features within each trial/window only.
8. Generate quality-control summaries.

## EMS-Specific Notes

- EMS is already fixation-level.
- Use `IMAGE` as `trial_id` only.
- Do not connect fixations across different images.
- Do not use image content or image names as model features.
- The current primary evaluation protocol is a subject-level fixed split:

```text
train / valid / test = 60 / 20 / 20
```

- The EMS official 4-fold fields from `Train_Valid.xlsx` are retained only as legacy metadata and diagnostics.
- Do not use official 4-fold cross-validation as the main research or deployment evaluation protocol.
