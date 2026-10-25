"""SICdb-Ext dataset adapter. Loads dataset/sicdb-ext/*.csv into the
canonical Admission schema (core.schema) — the SICdb-side equivalent of
adapters/mimic_iv_ext/adapter.py.

Differences from MIMIC that show up here (see data/sicdb-ext/predata/README.md
for the full list):
  - icd_version is always 10 (SICdb is ICD-10-GM only).
  - language is "de" (HPI/PE/hospital_course are built from coded German
    fields, not English free text).
  - no radiology, no microbiology — those lists are always empty and
    has_radiology=False / radiology_catalog=None (see core.ports.DatasetAdapter).
  - physical_exam IS the vital signs (a flat "Label: value unit; ..." string;
    SICdb has no narrative organ-system exam findings), so vital_signs is
    just an alias for physical_exam rather than something extracted from it.
"""
from __future__ import annotations
from core.icd import block

import json
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

from core.json_catalog import JsonLabCatalog
from core.schema import Admission, admission_from_parts

from .condition_priors import extract_condition_priors as _extract_condition_priors

_REPO_ROOT = Path(__file__).resolve().parents[3]  # .../train-platform
_DEFAULT_DATA_DIR = _REPO_ROOT / "dataset" / "sicdb-ext"


def _load_tables(data_dir: Path) -> dict:
    demo   = pd.read_csv(data_dir / "patient_demographics.csv", dtype={"hadm_id": str, "subject_id": str})
    hpi    = pd.read_csv(data_dir / "history_of_present_illness.csv", dtype={"hadm_id": str})
    pe     = pd.read_csv(data_dir / "physical_examination.csv", dtype={"hadm_id": str})
    course = pd.read_csv(data_dir / "hospital_course.csv", dtype={"hadm_id": str})
    dxdis  = pd.read_csv(data_dir / "discharge_diagnosis.csv", dtype={"hadm_id": str})
    meds   = pd.read_csv(data_dir / "discharge_medications.csv", dtype={"hadm_id": str})
    icd_dx = pd.read_csv(data_dir / "icd_diagnosis.csv", dtype={"hadm_id": str})
    icd_pr = pd.read_csv(data_dir / "icd_procedures.csv", dtype={"hadm_id": str})
    labs   = pd.read_csv(data_dir / "laboratory_tests.csv", dtype={"hadm_id": str, "itemid": str})
    comorb = pd.read_csv(data_dir / "comorbidity_flags.csv", dtype={"hadm_id": str})

    with open(data_dir / "primary_diagnosis_index.json", encoding="utf-8") as f:
        dx_index: dict = json.load(f)

    return {
        "demo":    demo.set_index("hadm_id"),
        "hpi":     hpi.set_index("hadm_id"),
        "pe":      pe.set_index("hadm_id"),
        "course":  course.set_index("hadm_id"),
        "dxdis":   dxdis.set_index("hadm_id"),
        "meds":    meds.groupby("hadm_id"),
        "icd_dx":  icd_dx.groupby("hadm_id"),
        "icd_pr":  icd_pr.groupby("hadm_id"),
        "labs":    labs.groupby("hadm_id"),
        "comorb":  comorb.groupby("hadm_id"),
        "dx_index": dx_index,
    }


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
    discharge_dx_text = (
        tables["dxdis"].loc[hadm_id, "discharge_diagnosis"] if hadm_id in tables["dxdis"].index else ""
    )
    if pd.isna(discharge_dx_text):
        discharge_dx_text = ""

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
            "icd_code": str(r.get("icd_code") or ""),
            "icd_version": int(r["icd_version"]) if pd.notna(r.get("icd_version")) else 10,
            "icd_diagnosis": str(r.get("icd_diagnosis", "")),
        }
        for r in _rows(tables["icd_dx"], hadm_id)
    ]
    # comorbidity_flags.csv rows -> {field: value}, attached separately (not
    # part of the canonical Admission schema) for condition_priors.py to read.
    comorbidities = {
        r["field"]: r["value"] for r in _rows(tables["comorb"], hadm_id)
    }

    admission: Admission = admission_from_parts(
        hadm_id=hadm_id,
        subject_id=subject_id,
        source="sicdb-ext",
        language="de",
        icd_code=dx.get("icd_code", ""),
        icd_version=10,
        long_title=dx.get("long_title", ""),
        demographics={
            "gender": demo_row.get("gender", ""),
            "anchor_age": int(demo_row["anchor_age"]) if str(demo_row.get("anchor_age", "")).isdigit() else None,
            "race": None,
            "marital_status": None,
            "insurance": None,
            "admission_type": demo_row.get("admission_type", ""),
            "admission_location": demo_row.get("admission_location", ""),
            "discharge_location": demo_row.get("discharge_location", ""),
            "hospital_expire_flag": int(demo_row.get("hospital_expire_flag", 0)),
            "admittime": "",   # SICdb has no real wall-clock admission date (offset-only)
        },
        hpi=hpi_text,
        physical_exam=pe_text,
        vital_signs=pe_text,   # PE *is* the vitals string for SICdb — see module docstring
        hospital_course=course_text,
        discharge_diagnosis=discharge_dx_text,
        labs=labs,
        microbiology=[],   # SICdb has no microbiology table
        radiology=[],      # SICdb has no radiology table
        medications=medications,
        procedures=procedures,
        icd_diagnoses=icd_diagnoses,
    )
    # Stash the raw comorbidity dict for condition_priors.py — not part of the
    # shared schema (every other adapter has no equivalent concept), but
    # admission dicts may carry extra dataset-private keys beyond Admission's
    # declared fields without breaking any consumer (they only read declared keys).
    admission["_sicdb_comorbidities"] = comorbidities
    return admission


class SicdbAdapter:
    """DatasetAdapter for SICdb-Ext."""

    name = "sicdb-ext"
    language = "de"
    has_radiology = False
    has_microbiology = False

    def __init__(
        self,
        data_dir: str | Path | None = None,
        lab_catalog_path: str | Path | None = None,
    ):
        self._data_dir = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR
        self.lab_catalog = JsonLabCatalog(
            lab_catalog_path or self._data_dir / "lab_provider_catalog.json",
            mapping_csv_path=self._data_dir / "lab_test_mapping.csv",
        )
        self.radiology_catalog = None
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
            if admission and block(admission["primary_diagnosis"].get("icd10_code")):
                yield admission

    def load_admission(self, case_id: str) -> Admission:
        tables = self._get_tables()
        admission = _build_admission(case_id, tables)
        assert admission is not None, f"hadm_id {case_id} not found or missing in dx_index"
        return admission

    def extract_condition_priors(self, admission: Admission) -> dict[str, bool]:
        return _extract_condition_priors(admission)
