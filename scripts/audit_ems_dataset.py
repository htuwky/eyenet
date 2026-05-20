from __future__ import annotations

import argparse
import json
from pathlib import Path

from eyenet.data.ems import (
    build_audit_summary,
    build_test_manifest,
    build_train_manifest,
    load_ems_config,
    save_audit_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the EMS eye movement dataset.")
    parser.add_argument("--config", default="configs/datasets/ems.yaml")
    parser.add_argument("--output-dir", default="outputs/ems_audit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_ems_config(args.config)

    train_manifest = build_train_manifest(cfg)
    test_manifest = build_test_manifest(cfg)
    summary = build_audit_summary(cfg, train_manifest, test_manifest)
    save_audit_outputs(Path(args.output_dir), train_manifest, test_manifest, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
