"""Shared paths and helpers for SICdb-Ext dataset statistics.

Mirrors data/mimic-iv-ext/statistics/config.py, adapted to SICdb-Ext's schema:
no radiology_reports.csv / microbiology.csv (SICdb has neither modality), and
no separate `d_icd_diagnoses` dictionary — diagnosis titles already live
(in German) inside primary_diagnosis_index.json itself.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))

DATASET_DIR = os.environ.get("CLINVERSE_SICDB_NORMALIZED_DIR", os.path.join(_REPO_ROOT, "data", "normalized", "sicdb-ext"))
OUTPUT_DIR = os.path.join(_HERE, "output")

PRIMARY_DX_INDEX = os.path.join(DATASET_DIR, "primary_diagnosis_index.json")
DEMOGRAPHICS_CSV = os.path.join(DATASET_DIR, "patient_demographics.csv")
LABS_CSV = os.path.join(DATASET_DIR, "laboratory_tests.csv")
MEDS_CSV = os.path.join(DATASET_DIR, "discharge_medications.csv")
ICD_PROC_CSV = os.path.join(DATASET_DIR, "icd_procedures.csv")

# Matplotlib CJK font fallbacks (macOS / Linux / Windows)
PLOT_FONT_FAMILY = [
    "PingFang SC",
    "Arial Unicode MS",
    "Heiti SC",
    "SimHei",
    "Microsoft YaHei",
    "DejaVu Sans",
]


def ensure_output_dir() -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


@lru_cache(maxsize=1)
def load_accepted_hadm_ids() -> set[int]:
    with open(PRIMARY_DX_INDEX, encoding="utf-8") as f:
        index = json.load(f)
    return {int(h) for h in index}


@lru_cache(maxsize=1)
def load_primary_diagnosis_df() -> pd.DataFrame:
    with open(PRIMARY_DX_INDEX, encoding="utf-8") as f:
        index = json.load(f)
    rows = [
        {
            "hadm_id": int(hadm_id),
            "icd_code": info["icd_code"],
            "icd_version": info["icd_version"],
            "long_title": info["long_title"],
        }
        for hadm_id, info in index.items()
        if info.get("icd_code")
    ]
    return pd.DataFrame(rows)


def filter_accepted(df: pd.DataFrame, hadm_col: str = "hadm_id") -> pd.DataFrame:
    accepted = load_accepted_hadm_ids()
    return df[df[hadm_col].isin(accepted)].copy()


def count_per_hadm(csv_path: str, hadm_col: str = "hadm_id") -> pd.Series:
    """Return row counts per admission (accepted only) for a long-format CSV."""
    accepted = load_accepted_hadm_ids()
    counts: dict[int, int] = {}
    for chunk in pd.read_csv(csv_path, usecols=[hadm_col], chunksize=500_000):
        sub = chunk[chunk[hadm_col].isin(accepted)]
        vc = sub[hadm_col].value_counts()
        for hadm_id, n in vc.items():
            counts[int(hadm_id)] = counts.get(int(hadm_id), 0) + int(n)
    # admissions with zero rows in this table
    base = pd.Series(0, index=sorted(accepted), dtype=int)
    if counts:
        base.update(pd.Series(counts, dtype=int))
    return base
