# Server Training Workflow

This project should use the remote Linux server as the training host and keep the local Windows machine as a lightweight analysis and code-editing client.

## What Goes To GitHub

Track:

- Source code under `src/` and `scripts/`
- Configs under `configs/`
- Documentation under `docs/`
- Small reproducibility splits under `data/splits/`

Do not track:

- `data/raw/`
- `data/processed/`
- `Hospital_Data/`
- `experiments/`
- `experiments_remote/`
- Model checkpoints and generated tables

## First Server Setup

Clone the repository on the server:

```bash
mkdir -p ~/projects
cd ~/projects
git clone git@github.com:htuwky/eyenet.git
cd eyenet
```

Create the environment:

```bash
conda env create -f environment.yml
conda activate eyenet
python -m pip install -e .
python -c "import eyenet; print(eyenet.__file__)"
```

Check CUDA:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## Minimal Data Upload

Upload only the processed training data, split files, and existing fusion results. Do not upload the full 40+ GB workspace unless rebuilding processed data from raw files on the server.

From WSL or Git Bash on the local machine:

```bash
cd /mnt/d/CodeProjects/Python/eyenet

rsync -avh --progress data/processed/EMS/ USER@HOST:/home/USER/projects/eyenet/data/processed/EMS/
rsync -avh --progress data/processed/mixed/ USER@HOST:/home/USER/projects/eyenet/data/processed/mixed/
rsync -avh --progress data/splits/EMS/ USER@HOST:/home/USER/projects/eyenet/data/splits/EMS/
rsync -avh --progress experiments/encoder_pretraining/fusion_ablation/ USER@HOST:/home/USER/projects/eyenet/experiments/encoder_pretraining/fusion_ablation/
rsync -avh --progress experiments/encoder_downstream/fusion_ablation/ USER@HOST:/home/USER/projects/eyenet/experiments/encoder_downstream/fusion_ablation/
```

Replace `USER` and `HOST` with the server username and address. Re-running the same `rsync` commands is safe; unchanged files are skipped.

## Long Training Sessions

Always run long experiments inside `tmux`:

```bash
ssh USER@HOST
cd ~/projects/eyenet
conda activate eyenet
tmux new -s eyenet
```

Detach without stopping training:

```text
Ctrl+B, then D
```

Reattach:

```bash
tmux attach -t eyenet
```

## Sync Lightweight Results Back

Do not download full checkpoints or datasets to the local machine. Pull only metrics, configs, predictions, and summaries:

```bash
cd /mnt/d/CodeProjects/Python/eyenet

rsync -avh --progress \
  --include "*/" \
  --include "metrics.csv" \
  --include "training_log.csv" \
  --include "config.json" \
  --include "predictions.csv" \
  --include "*summary*.csv" \
  --include "*summary*.json" \
  --exclude "*" \
  USER@HOST:/home/USER/projects/eyenet/experiments/ \
  experiments_remote/
```

Use `experiments_remote/` for local inspection only. It is ignored by Git.

## Current Encoder Selection State

The current BiGRU fusion pass indicates that `EMS + GazeBase + CRCNS_eye1 + OneStop` is the strongest BiGRU pretraining mixture by mean downstream AUC. HBN and full public fusion underperformed in that pass, so future Transformer and downstream experiments should compare against that mixture rather than assuming more datasets are always better.

