# Data Dictionary

This document defines the project-wide event-level eye movement schema.

## Core Identifiers

| Field | Description |
| --- | --- |
| `subject_id` | Participant identifier within a dataset. |
| `dataset_id` | Source dataset, for example `EMS`, `HBN`, or `GazeBase`. |
| `task_id` | Task or viewing condition. |
| `trial_id` | Stimulus, image, video clip, or window boundary. Not used as a model feature. |
| `event_index` | Event order within a trial/window. |

## Core Event Fields

| Field | Description |
| --- | --- |
| `event_type` | `fixation`, `saccade`, or other project-defined event. |
| `x` | Normalized horizontal gaze/fixation coordinate. |
| `y` | Normalized vertical gaze/fixation coordinate. |
| `duration_ms` | Event duration in milliseconds. |
| `amplitude` | Saccade amplitude in normalized units or DVA when available. |
| `angle` | Saccade direction angle. |
| `velocity` | Approximate saccade/event velocity. |
| `validity_mask` | Whether the event is valid for modeling. |

## Labels and Metadata

| Field | Description |
| --- | --- |
| `label` | Dataset-specific target label. |
| `age` | Participant age when available. |
| `sex` | Participant sex when available. |
| `diagnosis` | Human-readable diagnosis or group name when available. |

## Rules

- Image names, video names, and window ids are segmentation boundaries only.
- Image/video content is not used as a model input.
- Pupil fields are optional and must not be required by the core model.
