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

Install development tooling before running quality checks:

```powershell
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
```

Do not rely on ad hoc `PYTHONPATH` exports or per-script `sys.path.insert(...)` blocks as the normal workflow. The project uses a `src/` layout and should be run as an editable Python package during development.

For Codex-run commands, prefer invoking the installed environment directly:

```powershell
conda run -n eyenet python <script>
```

## Evaluation Validity Rules

The current EMS mainline does not use official 4-fold cross-validation. It uses a subject-level fixed split:

```text
train / valid / test = 60 / 20 / 20
active split file -> data/splits/EMS/ems_subject_split_60_20_20_seed42.csv
primary split column -> split
legacy metadata column -> official_fold
```

Use `split=train` for fitting, `split=valid` for epoch/threshold/calibration selection, and `split=test` only for final reporting.

The `fold` column is retained only for legacy official-fold baselines, dataset metadata, and diagnostics. Do not use `fold` to drive the current research-profile or deployment-profile evaluation.

Randomized multi-seed evaluation and fixed-split ensembling answer different questions.

Use randomized multi-seed summaries for model-selection stability:

```text
different seed -> different subject split and model initialization
valid summary -> mean/std across seeds
invalid use -> averaging probabilities across seeds as if they shared one test set
```

Use fixed-split late ensembles only when every test subject has one prediction from every seed:

```text
same downstream split -> same test subjects across seeds
required check -> seed_coverage.csv has n_seeds equal to the requested seed count
required command option -> --require-complete-seeds
ensemble grouping key -> split + subject_id + label, with fold retained only if present
```

If `seed_coverage.csv` shows incomplete coverage, the ensemble output is diagnostic only and must not be used as a primary result.

Thresholds must be selected from validation predictions or validation-derived threshold files. Do not select thresholds by inspecting test performance.

## Next Engineering Priority

The next engineering priority is deployment-profile packaging:

1. Calibrate the fixed-split OneStop ensemble or the selected single checkpoint on validation predictions.
2. Add a subject-level inference entrypoint that loads `best.pt` and `preprocessor.joblib`.
3. Emit a stable risk-score output contract with QC warnings and model-version metadata.
4. Keep Saliency4ASD deferred unless the project explicitly needs ASD auxiliary analysis.
