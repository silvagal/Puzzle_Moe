#!/usr/bin/env python3
"""Download and preprocess PTB-XL into the format expected by PTBXLDataset.

This script:
1) Downloads PTB-XL (if requested).
2) Extracts raw WFDB files.
3) Builds train/val/test splits using strat_fold.
4) Saves preprocessed records as NPZ files:
   {output_dir}/{train,val,test}/records.npz

The resulting directory can be used as the dataset.path in config YAMLs.
"""
from __future__ import annotations

import argparse
import ast
import json
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import wfdb

PTBXL_URLS = [
    "https://physionet.org/content/ptb-xl/1.0.3/ptb-xl-1.0.3.zip",
    "https://physionet.org/content/ptb-xl/1.0.3/ptbxl.zip",
    "https://physionet.org/content/ptb-xl/1.0.3/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip",
    "https://physionet.org/download/ptb-xl/1.0.3/ptb-xl-1.0.3.zip?filename=ptb-xl-1.0.3.zip",
    "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-1.0.3.zip",
]

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def download_ptbxl(destination: Path, overwrite: bool = False, url: str | None = None) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        return destination
    urls = [url] if url else PTBXL_URLS
    last_error: Exception | None = None
    for candidate in urls:
        print(f"Downloading PTB-XL from {candidate} to {destination} ...")
        try:
            urlretrieve(candidate, destination)
            return destination
        except Exception as exc:  # noqa: BLE001 - surface any download failure
            last_error = exc
            if destination.exists():
                destination.unlink()
            print(f"Download failed for {candidate}: {exc}")
    raise RuntimeError(
        "Failed to download PTB-XL. Try providing --url or download manually "
        "from PhysioNet and re-run without --download."
    ) from last_error


def download_ptbxl_wfdb(raw_dir: Path, overwrite: bool = False) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    print("Downloading PTB-XL using wfdb.dl_database (may take a while) ...")
    import wfdb

    wfdb.dl_database("ptb-xl", dl_dir=str(raw_dir), keep_subdirs=True, overwrite=overwrite)
    wfdb.dl_files(
        "ptb-xl",
        ["ptbxl_database.csv", "scp_statements.csv"],
        dl_dir=str(raw_dir),
        overwrite=overwrite,
    )
    return raw_dir
    return destination


def extract_zip(zip_path: Path, extract_to: Path, overwrite: bool = False) -> Path:
    if extract_to.exists() and not overwrite:
        return extract_to
    if extract_to.exists():
        shutil.rmtree(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path} to {extract_to} ...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    return extract_to


def load_metadata(raw_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    database_path = raw_root / "ptbxl_database.csv"
    scp_path = raw_root / "scp_statements.csv"
    if not database_path.exists() or not scp_path.exists():
        raise FileNotFoundError(
            "Missing ptbxl_database.csv or scp_statements.csv. "
            "Check that the PTB-XL archive was extracted correctly."
        )
    database = pd.read_csv(database_path)
    database["scp_codes"] = database["scp_codes"].apply(ast.literal_eval)
    scp_statements = pd.read_csv(scp_path, index_col=0)
    return database, scp_statements


def build_superclass_mapping(scp_statements: pd.DataFrame) -> Dict[str, str]:
    diagnostic = scp_statements[scp_statements["diagnostic"] == 1]
    mapping = {}
    for code, row in diagnostic.iterrows():
        diagnostic_class = row.get("diagnostic_class")
        if pd.isna(diagnostic_class):
            continue
        mapping[code] = diagnostic_class
    return mapping


def extract_labels(scp_codes: Dict[str, float], mapping: Dict[str, str]) -> np.ndarray:
    labels = np.zeros(len(SUPERCLASSES), dtype=np.float32)
    for code in scp_codes.keys():
        superclass = mapping.get(code)
        if superclass in SUPERCLASSES:
            labels[SUPERCLASSES.index(superclass)] = 1.0
    return labels


def load_record(record_path: Path) -> np.ndarray:
    signal, _ = wfdb.rdsamp(str(record_path))
    signal = signal.T
    if signal.shape[0] != 12:
        raise ValueError(f"Expected 12 leads, got {signal.shape[0]} for {record_path}")
    if signal.shape[1] < 5000:
        pad_width = 5000 - signal.shape[1]
        signal = np.pad(signal, ((0, 0), (0, pad_width)), mode="edge")
    signal = signal[:, :5000]
    mean = signal.mean(axis=1, keepdims=True)
    std = signal.std(axis=1, keepdims=True) + 1e-6
    signal = (signal - mean) / std
    return signal.astype(np.float32)


def build_split(database: pd.DataFrame, split_name: str) -> pd.DataFrame:
    if split_name == "train":
        return database[database["strat_fold"].between(1, 8)]
    if split_name == "val":
        return database[database["strat_fold"] == 9]
    if split_name == "test":
        return database[database["strat_fold"] == 10]
    raise ValueError(f"Unknown split: {split_name}")


def process_split(
    database: pd.DataFrame,
    split_name: str,
    mapping: Dict[str, str],
    records_dir: Path,
    output_dir: Path,
) -> None:
    split_df = build_split(database, split_name)
    signals: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    patient_ids: List[int] = []

    for _, row in split_df.iterrows():
        record_name = row["filename_hr"]
        record_path = records_dir / record_name
        signal = load_record(record_path)
        label = extract_labels(row["scp_codes"], mapping)
        signals.append(signal)
        labels.append(label)
        patient_ids.append(int(row["patient_id"]))

    split_output = output_dir / split_name
    split_output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        split_output / "records.npz",
        signals=np.stack(signals),
        labels=np.stack(labels),
        patient_ids=np.array(patient_ids, dtype=np.int64),
    )

    summary = {
        "split": split_name,
        "records": len(signals),
        "class_counts": np.stack(labels).sum(axis=0).tolist(),
    }
    with open(split_output / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved {split_name} split to {split_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare PTB-XL for Puzzle-MoE.")
    parser.add_argument(
        "--raw_dir",
        type=Path,
        default=Path("data/raw/ptbxl"),
        help="Directory to store the raw PTB-XL files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/processed/ptbxl"),
        help="Output directory for processed splits.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download PTB-XL from PhysioNet before processing.",
    )
    parser.add_argument(
        "--download_method",
        choices=["auto", "zip", "wfdb"],
        default="auto",
        help="Download strategy (auto tries ZIP then wfdb).",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Optional PTB-XL ZIP URL override.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing raw or processed data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir
    output_dir = args.output_dir

    if args.download:
        zip_path = raw_dir / "ptbxl.zip"
        if args.download_method in ("auto", "zip"):
            try:
                download_ptbxl(zip_path, overwrite=args.overwrite, url=args.url)
                extract_zip(zip_path, raw_dir, overwrite=args.overwrite)
            except RuntimeError as exc:
                if args.download_method == "zip":
                    raise
                print(f"ZIP download failed ({exc}); falling back to wfdb.")
                download_ptbxl_wfdb(raw_dir, overwrite=args.overwrite)
        else:
            download_ptbxl_wfdb(raw_dir, overwrite=args.overwrite)

    raw_root = raw_dir / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
    if not raw_root.exists():
        raw_root = raw_dir / "ptb-xl"
    if not raw_root.exists():
        raw_root = raw_dir
    if not raw_root.exists():
        raise FileNotFoundError(
            "Raw PTB-XL directory not found. Use --download or set --raw_dir."
        )

    database, scp_statements = load_metadata(raw_root)
    mapping = build_superclass_mapping(scp_statements)
    records_dir = raw_root / "records500"
    if not records_dir.exists():
        raise FileNotFoundError("Expected records500 directory with WFDB files.")

    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        process_split(database, split, mapping, records_dir, output_dir)

    print("PTB-XL preprocessing complete.")


if __name__ == "__main__":
    main()
