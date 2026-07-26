"""Extract comparable feature vectors from MIMIC admissions and Synthea personas."""

from __future__ import annotations

import math
import re

import numpy as np


def _s(val) -> str:
    """Coerce a possibly-NaN pandas cell to a string. MIMIC CSV columns with
    missing values load as float('nan'), not "", so any field read straight
    off admission/demographics dicts needs this before .upper()/string ops."""
    if val is None:
        return ""
    if isinstance(val, float) and math.isnan(val):
        return ""
    return str(val)

# Chronic disease categories: label → keywords matched against condition displays
_CONDITION_CATEGORIES: dict[str, list[str]] = {
    "diabetes":         ["diabetes", "diabetic", "hyperglycemia"],
    "hypertension":     ["hypertension", "hypertensive"],
    "ckd":              ["chronic kidney", "renal failure", "ckd", "renal disease"],
    "copd":             ["copd", "emphysema", "chronic obstructive"],
    "heart_failure":    ["heart failure", "cardiac failure", "congestive heart"],
    "liver_disease":    ["cirrhosis", "hepatitis", "liver disease", "liver failure"],
    "malignancy":       ["cancer", "carcinoma", "malignant", "neoplasm", "tumor", "lymphoma", "leukemia"],
    "afib":             ["atrial fibrillation", "a-fib", "afib"],
    "cad":              ["coronary artery", "ischemic heart", "angina", "myocardial infarction"],
    "stroke":           ["stroke", "cerebrovascular", "tia", "transient ischemic"],
    "depression":       ["depression", "depressive"],
    "anxiety":          ["anxiety"],
    "obesity":          ["obesity", "obese", "morbid obesity"],
    "hypothyroid":      ["hypothyroid", "thyroid deficiency"],
    "anemia":           ["anemia", "anaemia"],
    "hyperlipidemia":   ["hyperlipidemia", "hypercholesterol", "dyslipidemia"],
}

# Medication class categories: label → keywords matched against med displays
_MED_CATEGORIES: dict[str, list[str]] = {
    "antidiabetic":     ["metformin", "insulin", "glipizide", "glimepiride", "sitagliptin", "empagliflozin", "liraglutide"],
    "antihypertensive": ["amlodipine", "lisinopril", "losartan", "metoprolol", "atenolol", "carvedilol", "hydrochlorothiazide", "furosemide"],
    "anticoagulant":    ["warfarin", "heparin", "apixaban", "rivaroxaban", "dabigatran", "enoxaparin"],
    "inhaler":          ["albuterol", "tiotropium", "fluticasone", "budesonide", "salmeterol", "ipratropium"],
    "statin":           ["atorvastatin", "simvastatin", "rosuvastatin", "pravastatin", "lovastatin"],
    "immunosuppressant":["prednisone", "prednisolone", "tacrolimus", "cyclosporine", "mycophenolate", "azathioprine"],
    "antidepressant":   ["sertraline", "fluoxetine", "escitalopram", "citalopram", "venlafaxine", "duloxetine"],
    "proton_pump":      ["omeprazole", "pantoprazole", "lansoprazole", "esomeprazole"],
    "antiplatelet":     ["aspirin", "clopidogrel", "ticagrelor", "prasugrel"],
}

_RISK_FACTORS = {
    "smoking":          ["smoker", "smoking", "tobacco", "cigarette"],
    "alcohol":          ["alcohol", "drink", "etoh"],
    "obesity":          ["obesity", "obese", "bmi > 30", "bmi>30"],
    "pregnancy_related":["pregnant", "pregnancy", "gravid", "postpartum"],
}

# Race mapping for MIMIC-IV (numeric codes → canonical)
_RACE_CANON = {
    "WHITE": "white",
    "BLACK": "black",
    "HISPANIC": "hispanic",
    "ASIAN": "asian",
    "OTHER": "other",
    "UNKNOWN": "unknown",
}


def _match_any(text: str, keywords: list[str]) -> bool:
    t = text.lower()
    return any(kw in t for kw in keywords)


def _conditions_multihot(displays: list[str]) -> np.ndarray:
    keys = list(_CONDITION_CATEGORIES.keys())
    vec = np.zeros(len(keys), dtype=float)
    for i, cat in enumerate(keys):
        kws = _CONDITION_CATEGORIES[cat]
        if any(_match_any(d, kws) for d in displays):
            vec[i] = 1.0
    return vec


def _meds_multihot(displays: list[str]) -> np.ndarray:
    keys = list(_MED_CATEGORIES.keys())
    vec = np.zeros(len(keys), dtype=float)
    for i, cat in enumerate(keys):
        kws = _MED_CATEGORIES[cat]
        if any(_match_any(d, kws) for d in displays):
            vec[i] = 1.0
    return vec


def _risk_multihot(texts: list[str]) -> np.ndarray:
    keys = list(_RISK_FACTORS.keys())
    vec = np.zeros(len(keys), dtype=float)
    for i, cat in enumerate(keys):
        kws = _RISK_FACTORS[cat]
        if any(_match_any(t, kws) for t in texts):
            vec[i] = 1.0
    return vec


def _race_vec(race_str: str) -> np.ndarray:
    keys = list(_RACE_CANON.values())
    vec = np.zeros(len(keys), dtype=float)
    canon = _RACE_CANON.get(race_str.upper(), "other")
    if canon in keys:
        vec[keys.index(canon)] = 1.0
    return vec


def extract_mimic_features(admission: dict) -> dict:
    demo = admission["demographics"]
    age = demo.get("anchor_age") or 0
    gender_m = 1.0 if _s(demo.get("gender", "")).upper() == "M" else 0.0
    deceased = float(demo.get("hospital_expire_flag", 0) or 0)
    race_v = _race_vec(_s(demo.get("race", "")))

    # derive conditions from HPI + hospital_course text
    context_text = " ".join([
        _s(admission.get("hpi", "")),
        _s(admission.get("hospital_course", "")),
    ])
    # pull lab-based derived states
    abnormal_flags = {
        row["itemid"]: row.get("flag", "")
        for row in admission.get("labs", [])
    }
    # anemia: hemoglobin itemids 51222 / 50811
    derived_conditions = []
    if abnormal_flags.get("51222") == "abnormal" or abnormal_flags.get("50811") == "abnormal":
        derived_conditions.append("anemia")
    if abnormal_flags.get("50912") == "abnormal":
        derived_conditions.append("renal dysfunction")

    all_condition_text = [context_text] + derived_conditions
    all_med_text = [_s(m.get("drug", "")) for m in admission.get("medications", [])]
    risk_texts = [_s(admission.get("hpi", ""))]

    cond_v  = _conditions_multihot(all_condition_text)
    med_v   = _meds_multihot(all_med_text)
    risk_v  = _risk_multihot(risk_texts)

    return {
        "age": age,
        "gender_m": gender_m,
        "deceased": deceased,
        "race": race_v,
        "conditions": cond_v,
        "medications": med_v,
        "risk_factors": risk_v,
    }


def extract_synthea_features(persona: dict) -> dict:
    age = persona.get("age") or 0
    gender_m = 1.0 if persona.get("gender", "").lower() == "male" else 0.0
    deceased = 1.0 if persona.get("deceased") else 0.0
    race_str = persona.get("race", "").upper()
    race_v = _race_vec(race_str)

    condition_displays = [c["display"] for c in persona.get("conditions", [])]
    med_displays = [m["display"] for m in persona.get("medications", [])]

    obs_texts = [
        f"{o['display']} {o.get('value', '')} {o.get('unit', '')}"
        for o in persona.get("observations", [])
    ]
    encounter_types = [e.get("type", "") for e in persona.get("encounters", [])]
    risk_texts = condition_displays + obs_texts + encounter_types

    cond_v = _conditions_multihot(condition_displays)
    med_v  = _meds_multihot(med_displays)
    risk_v = _risk_multihot(risk_texts)

    return {
        "age": age,
        "gender_m": gender_m,
        "deceased": deceased,
        "race": race_v,
        "conditions": cond_v,
        "medications": med_v,
        "risk_factors": risk_v,
    }
