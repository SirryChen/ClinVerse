"""Canonical admission schema."""
from __future__ import annotations

from typing import Optional, TypedDict

from . import icd as icd_mod


class PrimaryDiagnosis(TypedDict):
    icd_code: str          # original code, as coded by the source hospital system
    icd_version: int        # 9 or 10, as coded by the source
    long_title: str         # native-language diagnosis title (English / German / ...)
    icd10_code: Optional[str]  # ALWAYS ICD-10 — converted from icd_code/icd_version via core.icd.to_icd10()


class Demographics(TypedDict, total=False):
    gender: str
    anchor_age: Optional[int]
    race: Optional[str]
    marital_status: Optional[str]
    insurance: Optional[str]
    admission_type: str
    admission_location: str
    discharge_location: str
    hospital_expire_flag: int
    admittime: str          # "" when the source has no real wall-clock date (MIMIC de-id, SICdb offsets)


class LabResult(TypedDict, total=False):
    itemid: str              # dataset-local id; meaning only resolvable via that dataset's LabCatalogPort
    valuestr: str
    flag: Optional[str]
    ref_range_lower: Optional[float]
    ref_range_upper: Optional[float]


class IcdDiagnosisRow(TypedDict, total=False):
    seq_num: int
    icd_code: str
    icd_version: int
    icd_diagnosis: str


class Admission(TypedDict, total=False):
    hadm_id: str
    subject_id: str
    case_id: str
    source: str              # "mimic-iv-ext" | "sicdb-ext" | ... — which adapter produced this
    language: str             # language of hpi/physical_exam/hospital_course free text: "en" | "de"
    primary_diagnosis: PrimaryDiagnosis
    demographics: Demographics
    hpi: str
    physical_exam: str
    vital_signs: str
    hospital_course: str       # moderator-only
    discharge_diagnosis: str   # moderator-only
    labs: list[LabResult]
    microbiology: list[dict]
    radiology: list[dict]
    medications: list[dict]
    procedures: list[dict]
    icd_diagnoses: list[IcdDiagnosisRow]


def admission_from_parts(
    *,
    hadm_id: str,
    subject_id: str,
    source: str,
    language: str,
    icd_code: str,
    icd_version: int,
    long_title: str,
    demographics: Demographics,
    hpi: str = "",
    physical_exam: str = "",
    vital_signs: str = "",
    hospital_course: str = "",
    discharge_diagnosis: str = "",
    labs: list[LabResult] | None = None,
    microbiology: list[dict] | None = None,
    radiology: list[dict] | None = None,
    medications: list[dict] | None = None,
    procedures: list[dict] | None = None,
    icd_diagnoses: list[IcdDiagnosisRow] | None = None,
) -> Admission:
    """Assemble an Admission dict, computing the canonical icd10_code field
    so every adapter gets ICD-9->10 normalization for free instead of each
    reimplementing it."""
    icd10_code = icd_mod.to_icd10(icd_code, icd_version)
    return {
        "hadm_id": hadm_id,
        "subject_id": subject_id,
        "case_id": f"{subject_id}_{hadm_id}",
        "source": source,
        "language": language,
        "primary_diagnosis": {
            "icd_code": icd_code,
            "icd_version": icd_version,
            "long_title": long_title,
            "icd10_code": icd10_code,
        },
        "demographics": demographics,
        "hpi": hpi,
        "physical_exam": physical_exam,
        "vital_signs": vital_signs,
        "hospital_course": hospital_course,
        "discharge_diagnosis": discharge_diagnosis,
        "labs": labs or [],
        "microbiology": microbiology or [],
        "radiology": radiology or [],
        "medications": medications or [],
        "procedures": procedures or [],
        "icd_diagnoses": icd_diagnoses or [],
    }
