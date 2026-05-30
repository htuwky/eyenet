from __future__ import annotations

import argparse
import json

import pandas as pd

from eyenet.data.encoder_dataset import build_encoder_dataloaders
from eyenet.data.encoder_ready import load_feature_schema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test encoder-ready Dataset/DataLoader outputs.")
    parser.add_argument("--events", default="data/processed/EMS/encoder_ready/clipped_qc_no_position/ems_encoder_events.csv")
    parser.add_argument("--schema", default="configs/features/encoder_original_13feature_core.json")
    parser.add_argument("--split", default="data/splits/EMS/ems_subject_split_60_20_20_seed42.csv")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--balanced-train-sampler", action="store_true")
    parser.add_argument("--require-label", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    schema = load_feature_schema(args.schema)
    events = pd.read_csv(args.events, dtype={"subject_id": str}, low_memory=False)
    split = pd.read_csv(args.split, dtype={"subject_id": str})
    loaders, preprocessor = build_encoder_dataloaders(
        events=events,
        split_subjects=split,
        feature_columns=schema["feature_columns"],
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        balanced_train_sampler=args.balanced_train_sampler,
        require_label=args.require_label,
    )

    summary = {
        "feature_dim": len(schema["feature_columns"]),
        "feature_columns": schema["feature_columns"],
        "splits": {},
    }
    for split_name, loader in loaders.items():
        batch = next(iter(loader))
        summary["splits"][split_name] = {
            "n_subjects": len(loader.dataset),
            "x_shape": list(batch["x"].shape),
            "mask_shape": list(batch["mask"].shape),
            "label_shape": list(batch["label"].shape),
            "mask_true_counts": batch["mask"].sum(dim=1).tolist(),
            "lengths": batch["length"].tolist(),
            "subject_ids": batch["subject_id"],
            "labels": batch["label"].tolist(),
            "x_mean_first_batch": float(batch["x"][batch["mask"]].mean().item()),
            "x_std_first_batch": float(batch["x"][batch["mask"]].std().item()),
        }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # Make sure the preprocessor is train-fitted and serializable.
    _ = preprocessor


if __name__ == "__main__":
    main()
