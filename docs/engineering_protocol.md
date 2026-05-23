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

Universal encoder input unit:

- fixation events
- raw gaze is used only to derive fixation events when a dataset does not already provide them

Allowed as model inputs:

- normalized fixation coordinates
- fixation duration
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
python -m pip install -e .
```

Verify the editable install:

```powershell
python -c "import eyenet; print(eyenet.__file__)"
```

Do not rely on ad hoc `PYTHONPATH` exports or per-script `sys.path.insert(...)` blocks as the normal workflow. The project uses a `src/` layout and should be run as an editable Python package during development.

For Codex-run commands, prefer invoking the installed environment directly:

```powershell
conda run -n eyenet python <script>
```

## Next Engineering Priority

The current completed adapters are HBN and GazeBase. The next code task is Saliency4ASD adapter development, followed by CRCNS eye-1 if the local raw files are available.

Each new dataset should use the fixed baseline encoder settings first. Do not begin broad hyperparameter search until candidate data sources have been screened.
