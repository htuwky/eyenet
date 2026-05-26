from __future__ import annotations

import json
import re
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd

from eyenet.utils.config import load_yaml_config


REQUIRED_FIXATION_COLUMNS = [
    "IMAGE",
    "FIX_INDEX",
    "FIX_DURATION",
    "FIX_X",
    "FIX_Y",
    "FIX_PUPIL",
]
SPREADSHEET_NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


@dataclass(frozen=True)
class EMSConfig:
    root: Path
    train_valid_dir: str = "Train_Valid/Fixations"
    test_dir: str = "Test/Fixations"
    official_split_file: str = "Train_Valid.xlsx"
    images_dir: str = "Images"
    control_max_subject_id: int = 199
    control_label: int = 0
    schizophrenia_label: int = 1

    @property
    def train_valid_path(self) -> Path:
        return self.root / self.train_valid_dir

    @property
    def test_path(self) -> Path:
        return self.root / self.test_dir

    @property
    def split_path(self) -> Path:
        return self.root / self.official_split_file

    @property
    def images_path(self) -> Path:
        return self.root / self.images_dir


def load_ems_config(config_path: str | Path) -> EMSConfig:
    raw = load_yaml_config(config_path)

    dataset = raw["dataset"]
    labels = raw["labels"]
    return EMSConfig(
        root=Path(dataset["root"]),
        train_valid_dir=dataset.get("train_valid_dir", "Train_Valid/Fixations"),
        test_dir=dataset.get("test_dir", "Test/Fixations"),
        official_split_file=dataset.get("official_split_file", "Train_Valid.xlsx"),
        images_dir=dataset.get("images_dir", "Images"),
        control_max_subject_id=int(labels.get("control_max_subject_id", 199)),
        control_label=int(labels.get("control_label", 0)),
        schizophrenia_label=int(labels.get("schizophrenia_label", 1)),
    )

def subject_id_from_train_file(path: Path) -> str:
    return path.stem


def label_from_subject_id(subject_id: str, cfg: EMSConfig) -> int:
    numeric_id = int(subject_id)
    if numeric_id <= cfg.control_max_subject_id:
        return cfg.control_label
    return cfg.schizophrenia_label


def read_official_folds(cfg: EMSConfig) -> pd.DataFrame:
    split_df = read_xlsx_first_sheet(cfg.split_path)
    rows: list[dict] = []
    for fold_name in split_df.columns:
        for subject_id in split_df[fold_name].dropna().astype(str):
            subject_id = subject_id.strip().zfill(3)
            rows.append(
                {
                    "subject_id": subject_id,
                    "fold": fold_name,
                    "label": label_from_subject_id(subject_id, cfg),
                }
            )
    folds = pd.DataFrame(rows)
    if folds["subject_id"].duplicated().any():
        duplicates = folds.loc[folds["subject_id"].duplicated(), "subject_id"].tolist()
        raise ValueError(f"Duplicate subject ids in official split: {duplicates}")
    return folds


def read_fixation_file(path: Path) -> pd.DataFrame:
    df = read_xlsx_first_sheet(path)
    missing = [col for col in REQUIRED_FIXATION_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    out = df[REQUIRED_FIXATION_COLUMNS].copy()
    for col in ["FIX_INDEX", "FIX_DURATION", "FIX_X", "FIX_Y", "FIX_PUPIL"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["IMAGE"] = out["IMAGE"].astype(str)
    return out


def read_xlsx_first_sheet(path: Path) -> pd.DataFrame:
    rows = read_xlsx_rows(path)
    if not rows:
        return pd.DataFrame()
    header = [str(value) for value in rows[0]]
    normalized_rows = [pad_row(row, len(header)) for row in rows[1:]]
    return pd.DataFrame(normalized_rows, columns=header)


def read_xlsx_rows(path: Path) -> list[list[str]]:
    with zipfile.ZipFile(path) as archive:
        shared_strings = read_shared_strings(archive)
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in relationships}
        first_sheet = workbook.find("a:sheets/a:sheet", SPREADSHEET_NS)
        if first_sheet is None:
            return []

        relationship_id = first_sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
        target = rel_map[relationship_id].lstrip("/")
        if not target.startswith("xl/"):
            target = f"xl/{target}"

        sheet = ET.fromstring(archive.read(target))
        parsed_rows: list[list[str]] = []
        for row in sheet.findall("a:sheetData/a:row", SPREADSHEET_NS):
            values: dict[int, str] = {}
            for cell in row.findall("a:c", SPREADSHEET_NS):
                col_index = column_index_from_cell_ref(cell.attrib.get("r", "A1"))
                values[col_index] = read_cell_value(cell, shared_strings)
            if values:
                max_col = max(values)
                parsed_rows.append([values.get(idx, "") for idx in range(max_col + 1)])
        return parsed_rows


def read_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    strings: list[str] = []
    for item in root.findall("a:si", SPREADSHEET_NS):
        strings.append("".join(text.text or "" for text in item.findall(".//a:t", SPREADSHEET_NS)))
    return strings


def read_cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    value = cell.find("a:v", SPREADSHEET_NS)
    text = "" if value is None else (value.text or "")
    if cell.attrib.get("t") == "s" and text:
        return shared_strings[int(text)]
    return text


def column_index_from_cell_ref(cell_ref: str) -> int:
    match = re.match(r"([A-Z]+)", cell_ref)
    if not match:
        return 0
    index = 0
    for char in match.group(1):
        index = index * 26 + (ord(char) - ord("A") + 1)
    return index - 1


def pad_row(row: list[str], length: int) -> list[str]:
    if len(row) >= length:
        return row[:length]
    return row + [""] * (length - len(row))


def summarize_fixations(df: pd.DataFrame) -> dict:
    duration = df["FIX_DURATION"]
    x = df["FIX_X"]
    y = df["FIX_Y"]
    pupil = df["FIX_PUPIL"]
    images = df["IMAGE"].dropna()

    return {
        "n_rows": int(len(df)),
        "n_images": int(images.nunique()),
        "n_fixations": int(df["FIX_INDEX"].notna().sum()),
        "fix_duration_mean": float(duration.mean()),
        "fix_duration_std": float(duration.std()),
        "fix_duration_min": float(duration.min()),
        "fix_duration_max": float(duration.max()),
        "x_min": float(x.min()),
        "x_max": float(x.max()),
        "y_min": float(y.min()),
        "y_max": float(y.max()),
        "pupil_mean": float(pupil.mean()),
        "pupil_std": float(pupil.std()),
        "missing_values": int(df.isna().sum().sum()),
    }


def build_train_manifest(cfg: EMSConfig) -> pd.DataFrame:
    folds = read_official_folds(cfg)
    fold_map = folds.set_index("subject_id")
    rows: list[dict] = []

    for path in sorted(cfg.train_valid_path.glob("*.xlsx")):
        subject_id = subject_id_from_train_file(path)
        if subject_id not in fold_map.index:
            raise ValueError(f"{subject_id} is not present in official split file.")

        df = read_fixation_file(path)
        row = {
            "subject_id": subject_id,
            "split": "train_valid",
            "fold": str(fold_map.loc[subject_id, "fold"]),
            "label": int(fold_map.loc[subject_id, "label"]),
            "file_path": str(path),
        }
        row.update(summarize_fixations(df))
        rows.append(row)

    manifest = pd.DataFrame(rows).sort_values("subject_id").reset_index(drop=True)
    expected = set(folds["subject_id"])
    observed = set(manifest["subject_id"])
    if expected != observed:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"Mismatch between split file and fixation files. Missing={missing}, extra={extra}")
    return manifest


def build_test_manifest(cfg: EMSConfig) -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted(cfg.test_path.glob("*.xlsx")):
        df = read_fixation_file(path)
        row = {
            "subject_id": path.stem,
            "split": "test",
            "fold": "test",
            "label": pd.NA,
            "file_path": str(path),
        }
        row.update(summarize_fixations(df))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("subject_id").reset_index(drop=True)


def count_images(cfg: EMSConfig) -> int:
    return sum(1 for _ in cfg.images_path.rglob("*.jpg"))


def build_fold_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    train = manifest[manifest["split"] == "train_valid"].copy()
    grouped = train.groupby(["fold", "label"], dropna=False).size().unstack(fill_value=0)
    grouped = grouped.rename(columns={0: "n_control", 1: "n_schizophrenia"})
    for col in ["n_control", "n_schizophrenia"]:
        if col not in grouped.columns:
            grouped[col] = 0
    grouped["n_subjects"] = grouped["n_control"] + grouped["n_schizophrenia"]
    return grouped.reset_index()[["fold", "n_subjects", "n_control", "n_schizophrenia"]]


def build_audit_summary(cfg: EMSConfig, train_manifest: pd.DataFrame, test_manifest: pd.DataFrame) -> dict:
    fold_summary = build_fold_summary(train_manifest)
    return {
        "config": asdict(cfg) | {"root": str(cfg.root)},
        "n_train_valid_subjects": int(len(train_manifest)),
        "n_test_subjects": int(len(test_manifest)),
        "n_images": int(count_images(cfg)),
        "train_valid_label_counts": {
            str(key): int(value) for key, value in train_manifest["label"].value_counts().sort_index().items()
        },
        "fold_summary": fold_summary.to_dict(orient="records"),
        "train_valid_rows": describe_numeric(train_manifest["n_rows"]),
        "train_valid_images_per_subject": describe_numeric(train_manifest["n_images"]),
        "train_valid_fix_duration_mean": describe_numeric(train_manifest["fix_duration_mean"]),
        "test_rows": describe_numeric(test_manifest["n_rows"]),
    }


def describe_numeric(series: pd.Series) -> dict:
    desc = series.astype(float).describe().to_dict()
    return {key: float(value) for key, value in desc.items()}


def save_audit_outputs(
    output_dir: str | Path,
    train_manifest: pd.DataFrame,
    test_manifest: pd.DataFrame,
    summary: dict,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    full_manifest = pd.concat([train_manifest, test_manifest], ignore_index=True)
    full_manifest.to_csv(output_path / "ems_manifest.csv", index=False, encoding="utf-8-sig")
    build_fold_summary(train_manifest).to_csv(output_path / "ems_fold_summary.csv", index=False, encoding="utf-8-sig")
    (output_path / "ems_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
