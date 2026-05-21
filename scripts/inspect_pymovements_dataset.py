from __future__ import annotations

import argparse
import json
from pathlib import Path

from eyenet.data.pymovements_registry import get_pymovements_dataset_info, list_pymovements_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a dataset definition bundled with pymovements.")
    parser.add_argument("--dataset", default="GazeBase")
    parser.add_argument("--root", default=None)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list:
        print("\n".join(list_pymovements_datasets()))
        return

    root = args.root or f"data/raw/{args.dataset}"
    info = get_pymovements_dataset_info(args.dataset, root=root)
    payload = info.to_dict()
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    print(text)
    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
