"""MIMIC-IV-Ext dataset adapter. Loads dataset/mimic-iv-ext/*.csv into the
canonical Admission schema (core.schema). This is pipeline/mimic_loader.py's
former _load_tables/_build_admission logic, unchanged, just wrapped behind
core.ports.DatasetAdapter and routed through admission_from_parts() so the
icd10_code field (ICD-9->10 conversion) is computed automatically instead of
left for every consumer to do itself.
"""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

from core.json_catalog import JsonLabCatalog, JsonRadiologyCatalog
from core.schema import Admission, admission_from_parts

from .condition_priors import extract_condition_priors

_REPO_ROOT = Path(__file__).resolve().parents[3]  # parent workspace
_DEFAULT_DATA_DIR = _REPO_ROOT / "dataset" / "mimic-iv-ext"


def _load_tables(data_dir: Path) -> dict:
    cache_path = data_dir / "_tables_cache.pkl"
    disable_cache = os.environ.get("CLINVERSE_DISABLE_TABLE_CACHE") == "1"
    if cache_path.exists() and not disable_cache:
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    demo   = pd.read_csv(data_dir / "patient_demographics.csv", dtype={"hadm_id": str, "subject_id": str})
    hpi    = pd.read_csv(data_dir / "history_of_present_illness.csv", dtype={"hadm_id": str})
    pe     = pd.read_csv(data_dir / "physical_examination.csv", dtype={"hadm_id": str})
    course = pd.read_csv(data_dir / "hospital_course.csv", dtype={"hadm_id": str})
    dxdis  = pd.read_csv(data_dir / "discharge_diagnosis.csv", dtype={"hadm_id": str})
    meds   = pd.read_csv(data_dir / "discharge_medications.csv", dtype={"hadm_id": str})
    icd_dx = pd.read_csv(data_dir / "icd_diagnosis.csv", dtype={"hadm_id": str})
    icd_pr = pd.read_csv(data_dir / "icd_procedures.csv", dtype={"hadm_id": str})
    micro  = pd.read_csv(data_dir / "microbiology.csv", dtype={"hadm_id": str})
    rad    = pd.read_csv(data_dir / "radiology_reports.csv", dtype={"hadm_id": str})
    labs   = pd.read_csv(data_dir / "laboratory_tests.csv", dtype={"hadm_id": str, "itemid": str})

    with open(data_dir / "primary_diagnosis_index.json") as f:
        dx_index: dict = json.load(f)

    tables = {
        "demo":    demo.set_index("hadm_id"),
        "hpi":     hpi.set_index("hadm_id"),
        "pe":      pe.set_index("hadm_id"),
        "course":  course.set_index("hadm_id"),
        "dxdis":   dxdis.set_index("hadm_id"),
        "meds":    meds.groupby("hadm_id"),
        "icd_dx":  icd_dx.groupby("hadm_id"),
        "icd_pr":  icd_pr.groupby("hadm_id"),
        "micro":   micro.groupby("hadm_id"),
        "rad":     rad.groupby("hadm_id"),
        "labs":    labs.groupby("hadm_id"),
        "dx_index": dx_index,
    }

    if not disable_cache:
        with open(cache_path, "wb") as f:
            pickle.dump(tables, f, protocol=pickle.HIGHEST_PROTOCOL)

    return tables


def _extract_vitals(pe_text: str) -> str:
    for line in pe_text.split("\n"):
        if any(kw in line for kw in ("VS:", "T ", "HR ", "BP ", "RR ", "O2")):
            return line.strip()
    return pe_text[:200]


def _rows(grouped, hadm_id: str) -> list[dict]:
    try:
        return grouped.get_group(hadm_id).to_dict("records")
    except KeyError:
        return []


def _build_admission(hadm_id: str, tables: dict) -> Optional[Admission]:
    dx_index = tables["dx_index"]
    if hadm_id not in dx_index:
        return None

    dx = dx_index[hadm_id]
    demo_row = tables["demo"].loc[hadm_id] if hadm_id in tables["demo"].index else None
    if demo_row is None:
        return None

    subject_id = str(int(demo_row["subject_id"]))

    hpi_text = tables["hpi"].loc[hadm_id, "hpi"] if hadm_id in tables["hpi"].index else ""
    pe_text = tables["pe"].loc[hadm_id, "pe"] if hadm_id in tables["pe"].index else ""
    course_text = tables["course"].loc[hadm_id, "hospital_course"] if hadm_id in tables["course"].index else ""
    discharge_dx_text = tables["dxdis"].loc[hadm_id, "discharge_diagnosis"] if hadm_id in tables["dxdis"].index else ""

    labs = [
        {
            "itemid": r["itemid"],
            "valuestr": r["valuestr"],
            "flag": r.get("flag", ""),
            "ref_range_lower": r.get("ref_range_lower", ""),
            "ref_range_upper": r.get("ref_range_upper", ""),
        }
        for r in _rows(tables["labs"], hadm_id)
    ]
    micro = [
        {
            "test_name": r.get("test_name", ""),
            "spec_type_desc": r.get("spec_type_desc", ""),
            "valuestr": r.get("valuestr", ""),
            "antibiogram": r.get("antibiogram", ""),
        }
        for r in _rows(tables["micro"], hadm_id)
    ]
    radiology = [
        {
            "note_id": r.get("note_id", ""),
            "modality": r.get("modality", ""),
            "region": r.get("region", ""),
            "exam_name": r.get("exam_name", ""),
            "text": r.get("text", ""),
        }
        for r in _rows(tables["rad"], hadm_id)
    ]
    medications = [
        {"drug": r.get("drug", ""), "prod_strength": r.get("prod_strength", ""), "route": r.get("route", "")}
        for r in _rows(tables["meds"], hadm_id)
    ]
    procedures = [
        {"icd_code": r.get("icd_code", ""), "icd_title": r.get("icd_title", "")}
        for r in _rows(tables["icd_pr"], hadm_id)
    ]
    icd_diagnoses = [
        {
            "seq_num": int(r.get("seq_num", 1)),
            "icd_code": str(r.get("icd_code", "")),
            "icd_version": int(r.get("icd_version", 9)),
            "icd_diagnosis": str(r.get("icd_diagnosis", "")),
        }
        for r in _rows(tables["icd_dx"], hadm_id)
    ]

    admission = admission_from_parts(
        hadm_id=hadm_id,
        subject_id=subject_id,
        source="mimic-iv-ext",
        language="en",
        icd_code=dx.get("icd_code", ""),
        icd_version=dx.get("icd_version", 9),
        long_title=dx.get("long_title", ""),
        demographics={
            "gender": demo_row.get("gender", ""),
            "anchor_age": int(demo_row["anchor_age"]) if str(demo_row.get("anchor_age", "")).isdigit() else None,
            "race": demo_row.get("race", ""),
            "marital_status": demo_row.get("marital_status", ""),
            "insurance": demo_row.get("insurance", ""),
            "admission_type": demo_row.get("admission_type", ""),
            "admission_location": demo_row.get("admission_location", ""),
            "discharge_location": demo_row.get("discharge_location", ""),
            "hospital_expire_flag": int(demo_row.get("hospital_expire_flag", 0)),
            "admittime": str(demo_row.get("admittime", "")),
        },
        hpi=hpi_text,
        physical_exam=pe_text,
        vital_signs=_extract_vitals(pe_text) if pe_text else "",
        hospital_course=course_text,
        discharge_diagnosis=discharge_dx_text,
        labs=labs,
        microbiology=micro,
        radiology=radiology,
        medications=medications,
        procedures=procedures,
        icd_diagnoses=icd_diagnoses,
    )
    return admission


class MimicAdapter:
    """DatasetAdapter for MIMIC-IV-Ext."""

    name = "mimic-iv-ext"
    language = "en"
    has_radiology = True
    has_microbiology = True

    def __init__(
        self,
        data_dir: str | Path | None = None,
        lab_catalog_path: str | Path | None = None,
        radiology_catalog_path: str | Path | None = None,
    ):
        self._data_dir = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR
        self.lab_catalog = JsonLabCatalog(
            lab_catalog_path or self._data_dir / "lab_provider_catalog.json",
            mapping_csv_path=self._data_dir / "lab_test_mapping.csv",
        )
        self.radiology_catalog = JsonRadiologyCatalog(
            radiology_catalog_path or self._data_dir / "radiology_catalog.json"
        )
        self._tables: dict | None = None

    def _get_tables(self) -> dict:
        if self._tables is None:
            self._tables = _load_tables(self._data_dir)
        return self._tables

    def load_admissions(self, limit: Optional[int] = None) -> Iterator[Admission]:
        tables = self._get_tables()
        hadm_ids = list(tables["dx_index"].keys())
        if limit:
            hadm_ids = hadm_ids[:limit]
        for hadm_id in hadm_ids:
            admission = _build_admission(hadm_id, tables)
            if admission:
                yield admission

    def load_admission(self, case_id: str) -> Admission:
        tables = self._get_tables()
        admission = _build_admission(case_id, tables)
        assert admission is not None, f"hadm_id {case_id} not found or missing in dx_index"
        return admission

    def extract_condition_priors(self, admission: Admission) -> dict[str, bool]:
        return extract_condition_priors(admission)
