"""
Data loading utilities for SICdb v1.0.8 (.csv.gz).
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Callable

import pandas as pd

from config import (
    CASES_FILE, LABORATORY_FILE, MEDICATION_FILE,
    DATA_FLOAT_H_FILE, DATA_RANGE_FILE, DATA_REF_FILE,
    CHUNK_SIZE, PE_HOURS_AFTER_ICU, VITAL_DATA_IDS,
)
from clinical import _s, icd_version_from_code, sex_to_gender
from references import load_references, ref_label, ref_unit, ref_value

log = logging.getLogger(__name__)


def load_cases(limit: int = 0) -> pd.DataFrame:
    log.info("Loading cases…")
    df = pd.read_csv(CASES_FILE)
    df = df.rename(columns={"CaseID": "hadm_id", "PatientID": "subject_id"})
    df["hadm_id"] = df["hadm_id"].astype(int)
    df["subject_id"] = df["subject_id"].astype(int)
    df["CaseID"] = df["hadm_id"]  # alias for data_ref / data_range joins
    if limit and limit > 0:
        df = df.head(limit)
    log.info("  %d cases loaded", len(df))
    return df


def build_demographics(cases: pd.DataFrame, refs: dict) -> pd.DataFrame:
    rows = []
    for _, r in cases.iterrows():
        offset_death = r.get("OffsetOfDeath")
        expire = 0
        if pd.notna(offset_death):
            expire = 1
        rows.append({
            "hadm_id": int(r["hadm_id"]),
            "subject_id": int(r["subject_id"]),
            "gender": sex_to_gender(r.get("Sex")),
            "anchor_age": int(r["AgeOnAdmission"]) if pd.notna(r.get("AgeOnAdmission")) else None,
            "race": None,
            "marital_status": None,
            "insurance": None,
            "admission_type": ref_label(r.get("SurgicalAdmissionType"), refs) or ref_label(r.get("AdmissionUrgency"), refs),
            "admission_location": ref_label(r.get("ReferringUnit"), refs),
            "discharge_location": ref_label(r.get("DischargeUnit"), refs),
            "hospital_expire_flag": expire,
        })
    return pd.DataFrame(rows)


def stream_laboratory():
    for chunk in pd.read_csv(
        LABORATORY_FILE,
        usecols=["CaseID", "LaboratoryID", "Offset", "LaboratoryValue", "LaboratoryType"],
        chunksize=CHUNK_SIZE,
    ):
        yield chunk


def stream_medications(case_ids: set, refs: dict, write_callback: Callable):
    log.info("Streaming medication for %d cases…", len(case_ids))
    batch_rows = []
    for chunk in pd.read_csv(
        MEDICATION_FILE,
        usecols=["CaseID", "DrugID", "Offset", "OffsetDrugEnd", "IsSingleDose", "Amount", "AmountPerMinute"],
        chunksize=CHUNK_SIZE,
    ):
        chunk = chunk[chunk["CaseID"].isin(case_ids)]
        if chunk.empty:
            continue
        for _, row in chunk.iterrows():
            drug = ref_value(row.get("DrugID"), refs)
            if not drug:
                continue
            batch_rows.append({
                "hadm_id": int(row["CaseID"]),
                "drug": drug,
                "prod_strength": None,
                "dose_val_rx": row.get("Amount"),
                # NOTE: was `ref_label(DrugID)`, which just repeats the drug
                # name itself (label/value fallback) — d_references stores the
                # real unit (g / IE / ml / drops / …) for "Drug" entries in
                # ReferenceUnit, fetched via ref_unit, not ref_label.
                "dose_unit_rx": ref_unit(row.get("DrugID"), refs),
                "route": "bolus" if row.get("IsSingleDose") == 1 else "infusion",
                "doses_per_24_hrs": None,
            })
        if len(batch_rows) >= 50_000:
            write_callback(pd.DataFrame(batch_rows))
            batch_rows = []
    if batch_rows:
        write_callback(pd.DataFrame(batch_rows))


def write_icd_diagnoses(cases: pd.DataFrame, write_callback: Callable):
    rows = []
    for _, r in cases.iterrows():
        code = _s(r.get("ICD10Main"))
        title = _s(r.get("ICD10MainText")) or _s(r.get("DiagnosisT2"))
        if not code and not title:
            continue
        rows.append({
            "hadm_id": int(r["hadm_id"]),
            "seq_num": 1,
            "icd_code": code or None,
            "icd_version": icd_version_from_code(code) if code else 10,
            "icd_diagnosis": title or code,
        })
        dx2 = _s(r.get("DiagnosisT2"))
        if dx2 and dx2 != title:
            rows.append({
                "hadm_id": int(r["hadm_id"]),
                "seq_num": 2,
                "icd_code": None,
                "icd_version": None,
                "icd_diagnosis": dx2,
            })
    if rows:
        df = pd.DataFrame(rows)
        # Mixing int (seq_num=1, coded) and None (seq_num=2, free-text-only
        # DiagnosisT2) rows makes pandas upcast icd_version to float64
        # ("10.0" instead of "10") — use the nullable Int64 dtype instead.
        df["icd_version"] = df["icd_version"].astype("Int64")
        write_callback(df)


def write_icd_procedures(case_ids: set, refs: dict, write_callback: Callable):
    log.info("Streaming data_range → icd_procedures proxy…")
    rows = []
    for chunk in pd.read_csv(
        DATA_RANGE_FILE,
        usecols=["CaseID", "DataID"],
        chunksize=CHUNK_SIZE,
    ):
        chunk = chunk[chunk["CaseID"].isin(case_ids)]
        for _, row in chunk.iterrows():
            title = ref_label(row.get("DataID"), refs)
            if not title:
                continue
            rows.append({
                "hadm_id": int(row["CaseID"]),
                "icd_code": None,
                "icd_title": title,
                "icd_version": None,
            })
        if len(rows) >= 50_000:
            write_callback(pd.DataFrame(rows))
            rows = []
    if rows:
        write_callback(pd.DataFrame(rows))


def load_data_ref(case_ids: set | None = None) -> pd.DataFrame:
    parts = []
    for chunk in pd.read_csv(
        DATA_REF_FILE,
        usecols=["CaseID", "FieldID", "RefID"],
        chunksize=CHUNK_SIZE,
    ):
        if case_ids is not None:
            chunk = chunk[chunk["CaseID"].isin(case_ids)]
        if not chunk.empty:
            parts.append(chunk)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["CaseID", "FieldID", "RefID"],
    )


def load_data_range(case_ids: set | None = None) -> pd.DataFrame:
    parts = []
    for chunk in pd.read_csv(
        DATA_RANGE_FILE,
        usecols=["CaseID", "DataID", "Offset", "OffsetEnd"],
        chunksize=CHUNK_SIZE,
    ):
        if case_ids is not None:
            chunk = chunk[chunk["CaseID"].isin(case_ids)]
        if not chunk.empty:
            parts.append(chunk)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["CaseID", "DataID", "Offset", "OffsetEnd"],
    )


def build_vitals_index(
    case_ids: set,
    icu_offsets: dict[int, int],
) -> dict[int, dict[int, list[float]]]:
    """
    One pass over data_float_h: aggregate vital values per CaseID / DataID
    within PE window [ICUOffset, ICUOffset + PE_HOURS_AFTER_ICU].
    """
    log.info("Streaming data_float_h for vitals (%d cases)…", len(case_ids))
    accum: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    pe_seconds = PE_HOURS_AFTER_ICU * 3600
    vital_ids = set(VITAL_DATA_IDS)
    chunk_n = 0

    for chunk in pd.read_csv(
        DATA_FLOAT_H_FILE,
        usecols=["CaseID", "DataID", "Offset", "Val", "cnt"],
        chunksize=CHUNK_SIZE,
    ):
        chunk = chunk[chunk["CaseID"].isin(case_ids) & chunk["DataID"].isin(vital_ids)]
        if chunk.empty:
            continue
        chunk["Offset"] = pd.to_numeric(chunk["Offset"], errors="coerce")
        chunk["Val"] = pd.to_numeric(chunk["Val"], errors="coerce")
        chunk["cnt"] = pd.to_numeric(chunk["cnt"], errors="coerce")
        chunk = chunk.dropna(subset=["Offset", "Val"])

        for row in chunk.itertuples(index=False):
            case_id = int(row.CaseID)
            icu_off = int(icu_offsets.get(case_id, 0) or 0)
            if row.Offset < icu_off or row.Offset > icu_off + pe_seconds:
                continue
            # data_float_h.Val is ALREADY the representative value for the hour
            # (verified against the raw per-minute `rawdata` stream: Val equals
            # the mean of the `cnt` non-zero raw samples for continuous signals
            # like BP/HR/SpO2, and the hourly total for single-entry/cumulative
            # fields like urine output). Dividing by cnt again — as the
            # previous version did — silently divided every vital by its
            # sample count (e.g. systolic BP 146 mmHg / 46 samples -> "2.4"),
            # corrupting every PE value built from this index.
            val = float(row.Val)
            accum[case_id][int(row.DataID)].append(val)

        chunk_n += 1
        if chunk_n % 50 == 0:
            log.info("  vitals pass chunk %d…", chunk_n)

    log.info("  vitals collected for %d cases", len(accum))
    return {k: dict(v) for k, v in accum.items()}
