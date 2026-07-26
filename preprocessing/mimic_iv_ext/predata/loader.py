"""
Data loading utilities for MIMIC-IV 3.1.
All source files are .csv.gz; this module provides lightweight helpers
that return pandas DataFrames with only the columns needed downstream.
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from config import (
    ADMISSIONS_FILE, PATIENTS_FILE, PRESCRIPTIONS_FILE,
    DIAGNOSES_ICD_FILE, D_ICD_DIAGNOSES_FILE,
    PROCEDURES_ICD_FILE, D_ICD_PROCEDURES_FILE,
    LABEVENTS_FILE, D_LABITEMS_FILE,
    MICROBIOLOGY_FILE,
    DISCHARGE_FILE, RADIOLOGY_FILE, RADIOLOGY_DETAIL_FILE,
    CHUNK_SIZE,
)

log = logging.getLogger(__name__)


def load_admissions() -> pd.DataFrame:
    log.info("Loading admissions…")
    df = pd.read_csv(
        ADMISSIONS_FILE,
        usecols=["subject_id", "hadm_id", "admittime", "dischtime",
                 "admission_type", "admission_location", "discharge_location",
                 "race", "marital_status", "insurance", "hospital_expire_flag"],
        parse_dates=["admittime", "dischtime"],
    )
    log.info("  %d admissions loaded", len(df))
    return df


def load_patients() -> pd.DataFrame:
    log.info("Loading patients…")
    df = pd.read_csv(
        PATIENTS_FILE,
        usecols=["subject_id", "gender", "anchor_age", "anchor_year"],
    )
    log.info("  %d patients loaded", len(df))
    return df


def load_diagnoses_icd() -> pd.DataFrame:
    """Returns diagnoses merged with long_title from d_icd_diagnoses.
    Loads the raw dx table first, then joins description in-memory.
    The raw dx table (~6.3M rows, integers only) is compact; desc is tiny (~100K rows).
    """
    log.info("Loading diagnoses_icd + d_icd_diagnoses…")
    dx = pd.read_csv(DIAGNOSES_ICD_FILE,
                     usecols=["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"])
    desc = pd.read_csv(D_ICD_DIAGNOSES_FILE,
                       usecols=["icd_code", "icd_version", "long_title"])
    merged = dx.merge(desc, on=["icd_code", "icd_version"], how="left")
    log.info("  %d diagnosis rows, %d unique hadm_id",
             len(merged), merged["hadm_id"].nunique())
    return merged


def load_primary_diagnoses() -> pd.DataFrame:
    """Load only seq_num==1 (primary) diagnoses merged with long_title.
    Uses streaming to avoid holding 6.3M rows in memory at once.
    Peak memory: one chunk (100K rows) + filtered primaries (~8K per chunk).
    """
    log.info("Loading primary diagnoses (seq_num=1, streaming)…")
    desc = pd.read_csv(D_ICD_DIAGNOSES_FILE,
                       usecols=["icd_code", "icd_version", "long_title"])
    primaries = []
    for chunk in pd.read_csv(
        DIAGNOSES_ICD_FILE,
        usecols=["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"],
        chunksize=CHUNK_SIZE,
    ):
        p = chunk[chunk["seq_num"] == 1]
        if not p.empty:
            primaries.append(p)
    primary_df = pd.concat(primaries, ignore_index=True)
    del primaries
    merged = primary_df.merge(desc, on=["icd_code", "icd_version"], how="left")
    del primary_df, desc
    log.info("  %d primary diagnosis rows", len(merged))
    return merged


def stream_diagnoses_icd(hadm_ids: set):
    """Stream diagnoses_icd in chunks, joining long_title on-the-fly.
    Yields DataFrames with columns: hadm_id, seq_num, icd_code, icd_version, long_title
    """
    log.info("Streaming diagnoses_icd for %d admissions…", len(hadm_ids))
    desc = pd.read_csv(D_ICD_DIAGNOSES_FILE,
                       usecols=["icd_code", "icd_version", "long_title"])
    for chunk in pd.read_csv(
        DIAGNOSES_ICD_FILE,
        usecols=["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"],
        chunksize=CHUNK_SIZE,
    ):
        filtered = chunk[chunk["hadm_id"].isin(hadm_ids)]
        if not filtered.empty:
            merged = filtered.merge(desc, on=["icd_code", "icd_version"], how="left")
            yield merged


def stream_procedures_icd(hadm_ids: set):
    """Stream procedures_icd in chunks, joining long_title on-the-fly."""
    log.info("Streaming procedures_icd for %d admissions…", len(hadm_ids))
    desc = pd.read_csv(D_ICD_PROCEDURES_FILE,
                       usecols=["icd_code", "icd_version", "long_title"])
    for chunk in pd.read_csv(
        PROCEDURES_ICD_FILE,
        usecols=["subject_id", "hadm_id", "seq_num",
                 "icd_code", "icd_version", "chartdate"],
        chunksize=CHUNK_SIZE,
    ):
        filtered = chunk[chunk["hadm_id"].isin(hadm_ids)]
        if not filtered.empty:
            merged = filtered.merge(desc, on=["icd_code", "icd_version"], how="left")
            yield merged


def load_d_labitems() -> pd.DataFrame:
    log.info("Loading d_labitems…")
    return pd.read_csv(D_LABITEMS_FILE,
                       usecols=["itemid", "label", "fluid", "category"])


def load_discharge(hadm_ids: Optional[set] = None) -> pd.DataFrame:
    """Load discharge notes. If hadm_ids given, filter to that set."""
    log.info("Loading discharge notes%s…",
             f" (filtering to {len(hadm_ids)} admissions)" if hadm_ids else "")
    chunks = []
    for chunk in pd.read_csv(
        DISCHARGE_FILE,
        usecols=["note_id", "subject_id", "hadm_id", "charttime", "text"],
        chunksize=CHUNK_SIZE,
    ):
        if hadm_ids is not None:
            chunk = chunk[chunk["hadm_id"].isin(hadm_ids)]
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    log.info("  %d discharge notes loaded", len(df))
    return df


def load_radiology(hadm_ids: Optional[set] = None) -> pd.DataFrame:
    log.info("Loading radiology notes%s…",
             f" (filtering to {len(hadm_ids)} admissions)" if hadm_ids else "")
    chunks = []
    for chunk in pd.read_csv(
        RADIOLOGY_FILE,
        usecols=["note_id", "subject_id", "hadm_id", "charttime", "text"],
        parse_dates=["charttime"],
        chunksize=CHUNK_SIZE,
    ):
        if hadm_ids is not None:
            chunk = chunk[chunk["hadm_id"].isin(hadm_ids)]
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    log.info("  %d radiology notes loaded", len(df))
    return df


def load_radiology_detail() -> pd.DataFrame:
    """Load radiology_detail; returns exam_name and parent_note_id mappings."""
    log.info("Loading radiology_detail…")
    df = pd.read_csv(
        RADIOLOGY_DETAIL_FILE,
        usecols=["note_id", "field_name", "field_value"],
    )
    # pivot to wide: one row per note_id with columns exam_name, parent_note_id
    wanted = df[df["field_name"].isin(["exam_name", "parent_note_id"])].copy()
    wide = wanted.pivot_table(
        index="note_id",
        columns="field_name",
        values="field_value",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    # ensure both columns exist
    for col in ["exam_name", "parent_note_id"]:
        if col not in wide.columns:
            wide[col] = None
    log.info("  %d note_id entries in radiology_detail", len(wide))
    return wide


def stream_labevents(hadm_ids: set):
    """
    Generator that yields chunks of labevents filtered to hadm_ids.
    Also fills NaN hadm_id rows using subject_id + admissions time window
    is done separately in labs.py.
    """
    log.info("Streaming labevents for %d admissions…", len(hadm_ids))
    for chunk in pd.read_csv(
        LABEVENTS_FILE,
        usecols=["labevent_id", "subject_id", "hadm_id", "itemid",
                 "charttime", "value", "valuenum", "valueuom",
                 "ref_range_lower", "ref_range_upper", "flag", "comments"],
        parse_dates=["charttime"],
        chunksize=CHUNK_SIZE,
    ):
        filtered = chunk[chunk["hadm_id"].isin(hadm_ids)]
        if not filtered.empty:
            yield filtered


def stream_microbiology(hadm_ids: set):
    """Generator yielding microbiology chunks filtered to hadm_ids.
    Includes spec_type_desc (human-readable specimen type) and antibiogram fields.
    """
    log.info("Streaming microbiologyevents for %d admissions…", len(hadm_ids))
    for chunk in pd.read_csv(
        MICROBIOLOGY_FILE,
        usecols=["subject_id", "hadm_id", "charttime",
                 "test_itemid", "test_name",
                 "org_itemid", "org_name",
                 "spec_itemid", "spec_type_desc",
                 "ab_name", "interpretation", "comments"],
        parse_dates=["charttime"],
        chunksize=CHUNK_SIZE,
    ):
        # exclude cancelled tests (org_itemid == 90760)
        chunk = chunk[chunk["org_itemid"] != 90760.0]
        filtered = chunk[chunk["hadm_id"].isin(hadm_ids)]
        if not filtered.empty:
            yield filtered


def stream_prescriptions(hadm_ids: set):
    """Generator yielding discharge-relevant prescription rows filtered to hadm_ids."""
    log.info("Streaming prescriptions for %d admissions…", len(hadm_ids))
    for chunk in pd.read_csv(
        PRESCRIPTIONS_FILE,
        usecols=["subject_id", "hadm_id", "drug_type", "drug",
                 "prod_strength", "dose_val_rx", "dose_unit_rx",
                 "route", "doses_per_24_hrs"],
        chunksize=CHUNK_SIZE,
    ):
        # Keep only MAIN drug orders (exclude BASE/ADDITIVE)
        chunk = chunk[chunk["drug_type"] == "MAIN"]
        filtered = chunk[chunk["hadm_id"].isin(hadm_ids)]
        if not filtered.empty:
            yield filtered
