"""
Central configuration for SICdb-Ext pipeline.
Source: dataset/SICdb/ (Salzburg ICU Database v1.0.8)
Output mirrors data/mimic-iv-ext/ schema (hadm_id = CaseID, subject_id = PatientID).
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))

BASE_SICDB = os.environ.get("CLINVERSE_SICDB_DIR", os.path.join(_ROOT, "dataset", "SICdb"))
BASE_OUT = os.environ.get("CLINVERSE_SICDB_NORMALIZED_DIR", os.path.join(_ROOT, "data", "normalized", "sicdb-ext"))

# ── Source tables (.csv.gz) ───────────────────────────────────────────────────
CASES_FILE        = os.path.join(BASE_SICDB, "cases.csv.gz")
REFERENCES_FILE   = os.path.join(BASE_SICDB, "d_references.csv.gz")
LABORATORY_FILE   = os.path.join(BASE_SICDB, "laboratory.csv.gz")
MEDICATION_FILE   = os.path.join(BASE_SICDB, "medication.csv.gz")
DATA_FLOAT_H_FILE = os.path.join(BASE_SICDB, "data_float_h.csv.gz")
DATA_RANGE_FILE   = os.path.join(BASE_SICDB, "data_range.csv.gz")
DATA_REF_FILE     = os.path.join(BASE_SICDB, "data_ref.csv.gz")

# ── Output paths ────────────────────────────────────────────────────────────────
OUT_HPI          = os.path.join(BASE_OUT, "history_of_present_illness.csv")
OUT_PE           = os.path.join(BASE_OUT, "physical_examination.csv")
OUT_DISCHARGE_DX = os.path.join(BASE_OUT, "discharge_diagnosis.csv")
OUT_DISCHARGE_PR = os.path.join(BASE_OUT, "discharge_procedures.csv")
OUT_LABS         = os.path.join(BASE_OUT, "laboratory_tests.csv")
OUT_MICRO        = os.path.join(BASE_OUT, "microbiology.csv")
OUT_ICD_DX       = os.path.join(BASE_OUT, "icd_diagnosis.csv")
OUT_ICD_PROC     = os.path.join(BASE_OUT, "icd_procedures.csv")
OUT_LAB_MAPPING  = os.path.join(BASE_OUT, "lab_test_mapping.csv")
OUT_DEMOGRAPHICS = os.path.join(BASE_OUT, "patient_demographics.csv")
OUT_MEDICATIONS  = os.path.join(BASE_OUT, "discharge_medications.csv")
OUT_HOSP_COURSE  = os.path.join(BASE_OUT, "hospital_course.csv")
# Structured comorbidity/precondition flags from data_ref (Precondition*/Premedication*/NYHA/...),
# the same dict clinical.data_ref_map() already computes for HPI/hospital_course text — exported
# separately (not just baked into free text) so downstream consumers (e.g. ClinVerse's
# Synthea condition-prior extraction) have structured access without having to regex-parse HPI.
OUT_COMORBIDITY  = os.path.join(BASE_OUT, "comorbidity_flags.csv")
OUT_DX_INDEX     = os.path.join(BASE_OUT, "primary_diagnosis_index.json")
OUT_STATS        = os.path.join(BASE_OUT, "stats.json")

# ── Processing parameters ───────────────────────────────────────────────────────
CHUNK_SIZE = 200_000
CLINICAL_BATCH_SIZE = 500

# Vital-sign window after ICU admission for PE (hours)
PE_HOURS_AFTER_ICU = 24

MIN_HPI_LENGTH = 25
MIN_PE_LENGTH  = 40

# SignalFloat / ventilator DataIDs used for PE assembly (d_references.ReferenceGlobalID)
VITAL_DATA_IDS = frozenset({
    701, 702, 703, 704, 705, 706, 707, 708, 709, 710,
    711, 712, 713, 715, 716, 724, 725, 2018, 2274, 2278, 2279, 2280, 2283,
})

# Sex reference IDs in d_references
SEX_MALE_REF   = 735
SEX_FEMALE_REF = 736
