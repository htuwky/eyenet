# Engineering Protocol

This project should remain reproducible, content-agnostic, and safe to extend across datasets.

## Core Rules

1. Raw data is never committed.
2. Processed data, checkpoints, experiment outputs, and hospital data are never committed.
3. Small split files, configs, source code, scripts, and documentation are committed.
4. Dataset-specific parsing belongs in `src/eyenet/data/` and `scripts/`.
5. Reusable modeling and training logic belongs in `src/eyenet/`.
6. One-off analysis is allowed in `scripts/`, but reusable logic should move into `src/eyenet/`.
7. Any model used for validation or deployment must save its preprocessing object together with its checkpoint.

## Directory Roles

```text
configs/
  datasets/       Dataset metadata and local path declarations.
  experiments/    Experiment defaults for reproducible training commands.
  features/       Shared feature schemas.

data/
  raw/            Local downloaded raw datasets. Ignored by Git.
  processed/      Generated event tables, QC tables, and encoder-ready data. Ignored by Git.
  splits/         Small reproducibility split files. Tracked by Git.

docs/
  Project design, data dictionaries, experiment protocols, and handoff notes.

experiments/
  Model outputs, logs, checkpoints, predictions, and metric tables. Ignored by Git.

scripts/
  CLI entrypoints for preprocessing, training, validation, and summaries.

src/eyenet/
  Reusable package code.
```

## Data Policy

The model is designed to use eye-movement behavior, not stimulus content.

Allowed as model inputs:

- normalized gaze/fixation coordinates
- event duration
- saccade or transition features
- local event order within a segment
- missingness/validity flags

Allowed for preprocessing/QC but not as direct content features:

- sampling rate
- screen resolution
- physical screen size
- viewing distance
- trial or segment boundaries

Not allowed as model features for the content-agnostic encoder:

- image pixels
- video frames
- stimulus category labels
- dataset-specific task names as direct predictive features
- clinical labels from unrelated disorders merged into one binary disease label

## Cross-Dataset Rule

External datasets should not train separate final encoders. Each dataset is converted and validated independently, then accepted datasets are pooled for one universal self-supervised encoder.

The downstream disease classifier is trained or fine-tuned on clinically relevant labeled data, currently EMS and later the user's hospital cohort.

## Adapter Acceptance Checklist

A dataset adapter is not considered ready until it produces:

1. A raw-to-event conversion command.
2. A schema validation report.
3. A subject-level QC report.
4. An encoder-ready table using a declared feature schema.
5. A masked-event modeling smoke test.

## Reproducible Command Pattern

Use the `eyenet` conda environment:

```powershell
conda activate eyenet
cd D:\CodeProjects\Python\eyenet
$env:PYTHONPATH="D:\CodeProjects\Python\eyenet\src"
```

For Codex-run commands, prefer:

```powershell
conda run -n eyenet cmd /S /C "set PYTHONPATH=D:\CodeProjects\Python\eyenet\src&& python <script>"
```

## Next Engineering Priority

The next code task is HBN adapter development:

```text
HBN raw CSV zip -> shared EyeNet event schema -> QC -> encoder-ready table -> masked-event smoke test
```

GazeBase should follow after HBN because GazeBase is larger and nested by round/subject/task.
