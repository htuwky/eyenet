from __future__ import annotations

import argparse
import json
from pathlib import Path

from eyenet.data.pymovements_registry import get_pymovements_dataset_info, make_pymovements_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a pymovements-managed dataset.")
    parser.add_argument("--dataset", default="GazeBase")
    parser.add_argument("--root", default=None)
    parser.add_argument("--no-extract", action="store_true")
    parser.add_argument("--remove-finished", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--metadata-output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root or f"data/raw/{args.dataset}")
    info = get_pymovements_dataset_info(args.dataset, root=root)
    metadata_text = json.dumps(info.to_dict(), ensure_ascii=False, indent=2, default=str)
    print(metadata_text)

    if args.metadata_output is not None:
        metadata_output = Path(args.metadata_output)
        metadata_output.parent.mkdir(parents=True, exist_ok=True)
        metadata_output.write_text(metadata_text, encoding="utf-8")

    if args.dry_run:
        print(f"Dry run only. Dataset would be downloaded to: {root}")
        return

    root.mkdir(parents=True, exist_ok=True)
    dataset = make_pymovements_dataset(args.dataset, root=root)
    dataset.download(
        extract=not args.no_extract,
        remove_finished=args.remove_finished,
        resume=not args.no_resume,
        verbose=1,
    )
    print(f"Download finished: {root}")


if __name__ == "__main__":
    main()
