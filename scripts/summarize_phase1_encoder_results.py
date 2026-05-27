from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


METRICS = ["auc", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1"]


@dataclass(frozen=True)
class Phase1Experiment:
    group: str
    display_name: str
    pretrain_data: str
    protocol: str
    root: str
    path_template: str
    modes: tuple[str, ...]
    role: str
    order: int


EXPERIMENTS = [
    Phase1Experiment(
        group="bigru64_supervised_only",
        display_name="Supervised-only BiGRU",
        pretrain_data="none",
        protocol="EMS downstream five-seed baseline",
        root="experiments/encoder_downstream/architecture_ablation",
        path_template="bigru64_supervised_only_seed{seed}/supervised/metrics.csv",
        modes=("supervised",),
        role="baseline",
        order=10,
    ),
    Phase1Experiment(
        group="bigru64_ems_crcns_mask045_seq3000_aligned",
        display_name="EMS+CRCNS aligned BiGRU, seq3000",
        pretrain_data="EMS + CRCNS eye-1",
        protocol="strict EMS-anchor aligned five-seed, max_seq_len=3000",
        root="experiments/encoder_downstream/architecture_ablation",
        path_template="bigru64_ems_crcns_mask045_seq3000_aligned_seed{seed}/{mode}/metrics.csv",
        modes=("finetune", "frozen"),
        role="aligned public-source candidate",
        order=20,
    ),
    Phase1Experiment(
        group="bigru64_mask045_fusion_ems_only",
        display_name="EMS-only MEM BiGRU",
        pretrain_data="EMS",
        protocol="strict aligned fusion-ablation five-seed, max_seq_len=1500",
        root="experiments/encoder_downstream/fusion_ablation",
        path_template="bigru64_mask045_fusion_ems_only_seed{seed}/{mode}/metrics.csv",
        modes=("finetune", "frozen"),
        role="EMS-only pretraining baseline",
        order=30,
    ),
    Phase1Experiment(
        group="bigru64_mask045_fusion_ems_crcns_eye1",
        display_name="EMS+CRCNS BiGRU",
        pretrain_data="EMS + CRCNS eye-1",
        protocol="strict aligned fusion-ablation five-seed, max_seq_len=1500",
        root="experiments/encoder_downstream/fusion_ablation",
        path_template="bigru64_mask045_fusion_ems_crcns_eye1_seed{seed}/{mode}/metrics.csv",
        modes=("finetune", "frozen"),
        role="public-source candidate",
        order=40,
    ),
    Phase1Experiment(
        group="bigru64_mask045_fusion_ems_gazebase_crcns_eye1",
        display_name="EMS+GazeBase+CRCNS BiGRU",
        pretrain_data="EMS + GazeBase + CRCNS eye-1",
        protocol="strict aligned fusion-ablation five-seed, max_seq_len=1500",
        root="experiments/encoder_downstream/fusion_ablation",
        path_template="bigru64_mask045_fusion_ems_gazebase_crcns_eye1_seed{seed}/{mode}/metrics.csv",
        modes=("finetune", "frozen"),
        role="public-source candidate",
        order=50,
    ),
    Phase1Experiment(
        group="bigru64_mask045_fusion_ems_gazebase_crcns_eye1_onestop",
        display_name="EMS+GazeBase+CRCNS+OneStop BiGRU",
        pretrain_data="EMS + GazeBase + CRCNS eye-1 + OneStop",
        protocol="strict aligned fusion-ablation five-seed, max_seq_len=1500",
        root="experiments/encoder_downstream/fusion_ablation",
        path_template="bigru64_mask045_fusion_ems_gazebase_crcns_eye1_onestop_seed{seed}/{mode}/metrics.csv",
        modes=("finetune", "frozen"),
        role="public-source candidate",
        order=60,
    ),
    Phase1Experiment(
        group="bigru64_mask045_fusion_ems_gazebase_crcns_eye1_hbn",
        display_name="EMS+GazeBase+CRCNS+HBN BiGRU",
        pretrain_data="EMS + GazeBase + CRCNS eye-1 + HBN",
        protocol="strict aligned fusion-ablation five-seed, max_seq_len=1500",
        root="experiments/encoder_downstream/fusion_ablation",
        path_template="bigru64_mask045_fusion_ems_gazebase_crcns_eye1_hbn_seed{seed}/{mode}/metrics.csv",
        modes=("finetune", "frozen"),
        role="public-source screening",
        order=70,
    ),
    Phase1Experiment(
        group="bigru64_mask045_fusion_ems_all_public",
        display_name="EMS+All-public BiGRU",
        pretrain_data="EMS + GazeBase + CRCNS eye-1 + OneStop + HBN",
        protocol="strict aligned fusion-ablation five-seed, max_seq_len=1500",
        root="experiments/encoder_downstream/fusion_ablation",
        path_template="bigru64_mask045_fusion_ems_all_public_seed{seed}/{mode}/metrics.csv",
        modes=("finetune", "frozen"),
        role="public-source screening",
        order=80,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the phase-1 encoder model-selection results.")
    parser.add_argument("--output", default="experiments/encoder_downstream/phase1_encoder_summary.csv")
    parser.add_argument("--split", default="test")
    parser.add_argument("--threshold-name", default="valid_best_balanced_accuracy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    per_seed = collect_phase1_rows(args)
    if per_seed.empty:
        raise FileNotFoundError("No phase-1 encoder metrics were found.")
    summary = summarize(per_seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(output_path.with_name(output_path.stem + "_per_seed.csv"), index=False, encoding="utf-8-sig")
    summary.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("Phase-1 per-seed metrics")
    print(per_seed.to_string(index=False))
    print("\nPhase-1 summary")
    print(summary.to_string(index=False))


def collect_phase1_rows(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict] = []
    for exp in EXPERIMENTS:
        for seed in range(5):
            for mode in exp.modes:
                metrics_path = Path(exp.root) / exp.path_template.format(seed=seed, mode=mode)
                if not metrics_path.exists():
                    rows.append(metadata_row(exp, seed, mode, metrics_path, status="missing"))
                    continue
                metrics = pd.read_csv(metrics_path)
                selected = metrics[
                    (metrics["split"] == args.split)
                    & (metrics["threshold_name"] == args.threshold_name)
                ]
                if selected.empty:
                    rows.append(metadata_row(exp, seed, mode, metrics_path, status="missing_threshold"))
                    continue
                row = selected.iloc[0].to_dict()
                row.update(metadata_row(exp, seed, mode, metrics_path, status="complete"))
                rows.append(row)
    return pd.DataFrame(rows)


def metadata_row(exp: Phase1Experiment, seed: int, mode: str, metrics_path: Path, status: str) -> dict:
    return {
        "order": exp.order,
        "experiment_group": exp.group,
        "display_name": exp.display_name,
        "pretrain_data": exp.pretrain_data,
        "protocol": exp.protocol,
        "role": exp.role,
        "mode": mode,
        "seed": seed,
        "metrics_path": str(metrics_path),
        "status": status,
    }


def summarize(per_seed: pd.DataFrame) -> pd.DataFrame:
    complete = per_seed[per_seed["status"] == "complete"].copy()
    rows = []
    group_cols = [
        "order",
        "experiment_group",
        "display_name",
        "pretrain_data",
        "protocol",
        "role",
        "mode",
    ]
    for keys, group in complete.groupby(group_cols, sort=True):
        row = dict(zip(group_cols, keys))
        row["n_seeds"] = int(group["seed"].nunique())
        row["seeds"] = ",".join(str(seed) for seed in sorted(group["seed"].unique()))
        row["complete_5seed"] = row["n_seeds"] == 5
        for metric in METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
            row[f"{metric}_min"] = float(group[metric].min())
            row[f"{metric}_max"] = float(group[metric].max())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["mode", "balanced_accuracy_mean", "auc_mean"],
        ascending=[True, False, False],
    )


if __name__ == "__main__":
    main()
