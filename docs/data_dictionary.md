# Data Dictionary

This document defines the shared EyeNet fixation-event schema. All datasets must be converted into this format before universal encoder pretraining or downstream model training. Datasets with raw gaze samples must first run fixation detection; raw gaze samples are not the primary encoder input.

## Required Event Columns

| Field | Description | Used As Feature |
| --- | --- | --- |
| `dataset_id` | Source dataset identifier, for example `EMS`, `GazeBase`, or `HBN`. | No |
| `subject_id` | Participant identifier within the dataset. | No |
| `label` | Dataset-specific target label. May be unavailable or auxiliary for self-supervised datasets. | No for pretraining; yes only for supervised heads |
| `trial_id` | Stimulus, video clip, task block, or artificial window boundary. | No |
| `event_index` | Event order within a trial/window. | Optional derived position feature only |
| `event_type` | Event type. The universal encoder currently expects `fixation`. | No |
| `x_norm` | Horizontal position normalized to `[0, 1]`. | Yes |
| `y_norm` | Vertical position normalized to `[0, 1]`. | Yes |
| `duration_ms` | Fixation duration in milliseconds. For raw-gaze datasets this is produced by fixation detection; it is not the raw sample interval. | Yes |

## Recommended Event Columns

| Field | Description | Used As Feature |
| --- | --- | --- |
| `split` | Dataset-provided or project-generated split. Current EMS mainline uses `train`, `valid`, and `test`. | No |
| `fold` | Legacy or dataset-provided fold metadata. Not the active EMS evaluation driver. | No |
| `official_fold` | EMS official fold retained as metadata after fixed 60/20/20 split creation. | No |
| `x_dva` | Horizontal visual angle coordinate when screen geometry is known. | Yes, optional |
| `y_dva` | Vertical visual angle coordinate when screen geometry is known. | Yes, optional |
| `saccade_dx_norm` | Transition displacement in normalized horizontal units. | Yes |
| `saccade_dy_norm` | Transition displacement in normalized vertical units. | Yes |
| `saccade_amplitude_norm` | Transition amplitude in normalized units. | Yes |
| `saccade_angle` | Transition direction angle in radians. | Transformed to sine/cosine |
| `saccade_dx_dva` | Transition displacement in horizontal DVA. | Yes, optional |
| `saccade_dy_dva` | Transition displacement in vertical DVA. | Yes, optional |
| `saccade_amplitude_dva` | Transition amplitude in DVA. | Yes, optional |
| `transition_velocity_dva_s_approx` | Approximate transition velocity in DVA/s. | Yes, optional |

## Optional Metadata Columns

| Field | Description |
| --- | --- |
| `session_id` | Recording session identifier. |
| `task_id` | Task or viewing condition. |
| `timestamp_start_ms` | Event start timestamp. |
| `timestamp_end_ms` | Event end timestamp. |
| `pupil_optional` | Pupil measurement when available. Not required by the core model. |
| `age` | Participant age. |
| `sex` | Participant sex. |
| `diagnosis` | Human-readable group or diagnosis. |
| `validity` | Validity or quality marker. |

## Modeling Rules

- Image names, video names, clip ids, and task blocks are segmentation boundaries only.
- Raw gaze is an adapter input only. The universal encoder operates on fixation events.
- Image/video pixels or semantic content are not model inputs.
- Pupil features are optional and must not be required by the core model.
- DVA features are preferred when hardware metadata is available, but the encoder must be able to run without them.
- Clinical labels from different disorders must not be merged into one binary target.
- Current EMS supervised training and final evaluation use `split`, not `fold`.
- `fold` is allowed only for legacy scripts, external dataset metadata, and diagnostics.
