"""
Central configuration for MIMIC-IV-Ext full-dataset pipeline.
Mirrors the structure of MIMIC-IV-Ext CDM (paulhager/MIMIC-Clinical-Decision-Making-Dataset)
but generalised to ALL primary diagnoses instead of 4 abdominal pathologies.
"""
import os

# ── Source paths ──────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))

BASE_MIMIC = os.path.join(_ROOT, "dataset", "mimic-iv-3.1")
BASE_NOTE  = os.path.join(_ROOT, "dataset", "mimic-iv-note", "note")
BASE_OUT   = os.path.join(_ROOT, "dataset", "mimic-iv-ext")

# hosp tables (all .csv.gz)
HOSP = os.path.join(BASE_MIMIC, "hosp")

ADMISSIONS_FILE          = os.path.join(HOSP, "admissions.csv.gz")
PATIENTS_FILE            = os.path.join(HOSP, "patients.csv.gz")
PRESCRIPTIONS_FILE       = os.path.join(HOSP, "prescriptions.csv.gz")
DIAGNOSES_ICD_FILE       = os.path.join(HOSP, "diagnoses_icd.csv.gz")
D_ICD_DIAGNOSES_FILE     = os.path.join(HOSP, "d_icd_diagnoses.csv.gz")
PROCEDURES_ICD_FILE      = os.path.join(HOSP, "procedures_icd.csv.gz")
D_ICD_PROCEDURES_FILE    = os.path.join(HOSP, "d_icd_procedures.csv.gz")
LABEVENTS_FILE           = os.path.join(HOSP, "labevents.csv.gz")
D_LABITEMS_FILE          = os.path.join(HOSP, "d_labitems.csv.gz")
MICROBIOLOGY_FILE        = os.path.join(HOSP, "microbiologyevents.csv.gz")

# note tables
DISCHARGE_FILE           = os.path.join(BASE_NOTE, "discharge.csv.gz")
DISCHARGE_DETAIL_FILE    = os.path.join(BASE_NOTE, "discharge_detail.csv.gz")
RADIOLOGY_FILE           = os.path.join(BASE_NOTE, "radiology.csv.gz")
RADIOLOGY_DETAIL_FILE    = os.path.join(BASE_NOTE, "radiology_detail.csv.gz")

# ── Output paths ──────────────────────────────────────────────────────────────
OUT_HPI          = os.path.join(BASE_OUT, "history_of_present_illness.csv")
OUT_PE           = os.path.join(BASE_OUT, "physical_examination.csv")
OUT_DISCHARGE_DX = os.path.join(BASE_OUT, "discharge_diagnosis.csv")
OUT_DISCHARGE_PR = os.path.join(BASE_OUT, "discharge_procedures.csv")
OUT_LABS         = os.path.join(BASE_OUT, "laboratory_tests.csv")
OUT_MICRO        = os.path.join(BASE_OUT, "microbiology.csv")
OUT_RADIOLOGY    = os.path.join(BASE_OUT, "radiology_reports.csv")
OUT_ICD_DX       = os.path.join(BASE_OUT, "icd_diagnosis.csv")
OUT_ICD_PROC     = os.path.join(BASE_OUT, "icd_procedures.csv")
OUT_LAB_MAPPING  = os.path.join(BASE_OUT, "lab_test_mapping.csv")
OUT_DEMOGRAPHICS = os.path.join(BASE_OUT, "patient_demographics.csv")
OUT_MEDICATIONS  = os.path.join(BASE_OUT, "discharge_medications.csv")
OUT_HOSP_COURSE  = os.path.join(BASE_OUT, "hospital_course.csv")
OUT_DX_INDEX     = os.path.join(BASE_OUT, "primary_diagnosis_index.json")
OUT_STATS        = os.path.join(BASE_OUT, "stats.json")

# ── Processing parameters ─────────────────────────────────────────────────────
# pandas chunk size for large files
CHUNK_SIZE = 100_000

# Lab/radiology time window: include results up to N hours before admission
LAB_HOURS_BEFORE_ADMIT  = 24
RAD_HOURS_BEFORE_ADMIT  = 24

# Minimum PE string length to be considered valid
MIN_PE_LENGTH = 40

# Sections of radiology report to EXCLUDE (contain radiologist's conclusion)
RAD_BAD_FIELDS = {
    "CLINICAL HISTORY", "MEDICAL HISTORY", "CLINICAL INFORMATION",
    "COMPARISON", "COMPARISONS", "COMMENT", "CONCLUSION",
    "HISTORY", "IMPRESSION", "CLINICAL INDICATION", "INDICATION",
    "OPERATORS", "REASON", "REFERENCE", "DATE",
}

# Radiology modality keyword mapping  (exam_name token → canonical modality)
RAD_MODALITY_MAP = {
    "CT":          "CT",
    "CAT":         "CT",
    "CTU":         "CT",
    "COMPUTED":    "CT",
    "MR":          "MRI",
    "MRI":         "MRI",
    "MRA":         "MRI",
    "MRE":         "MRI",
    "MRCP":        "MRI",
    "US":          "Ultrasound",
    "ULTRASOUND":  "Ultrasound",
    "SONO":        "Ultrasound",
    "ECHO":        "Ultrasound",
    "XR":          "Radiograph",
    "X-RAY":       "Radiograph",
    "XRAY":        "Radiograph",
    "CHEST":       "Radiograph",
    "RADIOGRAPH":  "Radiograph",
    "FLUOROSCOPY": "Fluoroscopy",
    "FLUORO":      "Fluoroscopy",
    "ERCP":        "ERCP",
    "PTC":         "PTC",
    "DRAINAGE":    "Drainage",
    "NM":          "Nuclear Medicine",
    "NUCLEAR":     "Nuclear Medicine",
    "PET":         "Nuclear Medicine",
    "ANGIO":       "Angiography",
}

# Radiology region keyword mapping
RAD_REGION_MAP = {
    "ABD":       "Abdomen",
    "ABDOMEN":   "Abdomen",
    "ABDOMINAL": "Abdomen",
    "PELV":      "Pelvis",
    "PELVIS":    "Pelvis",
    "PELVIC":    "Pelvis",
    "CHEST":     "Chest",
    "THORAX":    "Chest",
    "THORACIC":  "Chest",
    "HEAD":      "Head",
    "BRAIN":     "Head",
    "CRANIAL":   "Head",
    "NECK":      "Neck",
    "CERVICAL":  "Neck",
    "SPINE":     "Spine",
    "SPINAL":    "Spine",
    "LUMBAR":    "Spine",
    "THORACOLUMBAR": "Spine",
    "EXTREMIT":  "Extremity",
    "KNEE":      "Extremity",
    "HIP":       "Extremity",
    "SHOULDER":  "Extremity",
    "ANKLE":     "Extremity",
    "FOOT":      "Extremity",
    "WRIST":     "Extremity",
    "HAND":      "Extremity",
    "VENOUS":    "Venous",
    "VEIN":      "Venous",
    "BILIARY":   "Abdomen",
    "HEPATIC":   "Abdomen",
    "RENAL":     "Abdomen",
    "CARDIAC":   "Chest",
    "CAROTID":   "Neck",
}
