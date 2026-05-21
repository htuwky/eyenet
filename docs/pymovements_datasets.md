# PyMovements Dataset Integration

`pymovements` bundles dataset definitions and readers for several public eye-tracking datasets. This means the project can use the package to manage download URLs, metadata, and raw file parsing rules, but the raw data still has to be downloaded locally.

## Current Environment

Verified package:

```text
pymovements 0.26.2
```

Datasets confirmed in the local package:

```text
GazeBase
HBN
GazeBaseVR
```

Datasets not found in the local package:

```text
CRCNS eye-1
Saliency4ASD
```

## GazeBase Metadata from PyMovements

The local `pymovements` definition reports:

```text
sampling_rate_hz: 1000
screen_width_px: 1680
screen_height_px: 1050
screen_width_cm: 47.4
screen_height_cm: 29.7
viewing_distance_cm: 55
raw source: figshare GazeBase_v2_0.zip
```

GazeBase provides gaze samples in degrees of visual angle. It is not already an EyeNet fixation/saccade event table. The adapter must therefore convert continuous gaze samples into the shared event schema before encoder pretraining.

## Commands

List available `pymovements` datasets:

```powershell
$env:PYTHONPATH="D:\CodeProjects\Python\eyenet\src"
python scripts/inspect_pymovements_dataset.py --list
```

Inspect GazeBase metadata without downloading:

```powershell
$env:PYTHONPATH="D:\CodeProjects\Python\eyenet\src"
python scripts/inspect_pymovements_dataset.py `
  --dataset GazeBase `
  --root data/raw/GazeBase `
  --output data/raw/GazeBase/pymovements_metadata.json
```

Dry-run the GazeBase downloader:

```powershell
$env:PYTHONPATH="D:\CodeProjects\Python\eyenet\src"
python scripts/download_pymovements_dataset.py `
  --dataset GazeBase `
  --root data/raw/GazeBase `
  --metadata-output data/raw/GazeBase/pymovements_metadata.json `
  --dry-run
```

Download GazeBase:

```powershell
$env:PYTHONPATH="D:\CodeProjects\Python\eyenet\src"
python scripts/download_pymovements_dataset.py `
  --dataset GazeBase `
  --root data/raw/GazeBase `
  --metadata-output data/raw/GazeBase/pymovements_metadata.json
```

## Current Local Status

Metadata inspection and downloader dry-run are verified.

The automated GazeBase download was attempted from:

```text
https://figshare.com/ndownloader/files/27039812
```

but the local environment produced an empty/corrupted archive:

```text
data/raw/GazeBase/downloads/GazeBase_v2_0.zip
```

This means the code path is ready, but the large-file download should be completed manually if the automated downloader continues to fail.

Manual fallback:

1. Download `GazeBase_v2_0.zip` from the URL above in a browser or other download manager.
2. Place it at:

```text
data/raw/GazeBase/downloads/GazeBase_v2_0.zip
```

3. Run the downloader again. `pymovements` should verify and extract the existing file if the archive is complete.

## Next Adapter Step

After GazeBase files are downloaded, implement raw sample conversion:

```text
raw gaze samples -> fixation/saccade detection or fixed temporal windows -> shared EyeNet event schema -> QC -> encoder-ready table
```

The first GazeBase adapter should start with a small subset, for example one task and a few subjects, before processing the whole dataset.
