"""MIMIC comorbidity-prior extraction: secondary (seq_num>=2) coded ICD
diagnoses -> mimic_force_X Synthea injection flags.

Moved unchanged from the old pipeline/mimic_prior_extractor.py — MIMIC's
diagnoses_icd table genuinely has coded secondary diagnoses, so prefix-
matching ICD codes works here. (SICdb does not have this; see
adapters/sicdb_ext/condition_priors.py for why it needs a completely
different source signal.)
"""
from __future__ import annotations

from core.schema import Admission

# ICD code → Synthea mimic_force_X attribute mapping.
# Key = ICD-9/10 prefix (or exact code), value = attribute name injected into Synthea.
_ICD_TO_FORCE_FLAG: list[tuple[str, str]] = [
    # Hypertension
    ("4019", "mimic_force_hypertension"),
    ("I10", "mimic_force_hypertension"),
    # Hyperlipidemia
    ("2724", "mimic_force_hyperlipidemia"),
    ("E785", "mimic_force_hyperlipidemia"),
    ("E780", "mimic_force_hyperlipidemia"),
    # GERD / reflux esophagitis
    ("53081", "mimic_force_gerd"),
    ("K219", "mimic_force_gerd"),
    ("K210", "mimic_force_gerd"),
    # Depression / MDD
    ("311", "mimic_force_depression"),
    ("2962", "mimic_force_depression"),  # MDD single episode
    ("2963", "mimic_force_depression"),  # MDD recurrent
    ("F329", "mimic_force_depression"),
    ("F330", "mimic_force_depression"),
    ("F319", "mimic_force_depression"),  # BD depressive
    # Anxiety
    ("30000", "mimic_force_anxiety"),
    ("30001", "mimic_force_anxiety"),
    ("30009", "mimic_force_anxiety"),
    ("F419", "mimic_force_anxiety"),
    ("F411", "mimic_force_anxiety"),  # GAD
    ("F410", "mimic_force_anxiety"),  # Panic
    # Diabetes type 2
    ("25000", "mimic_force_diabetes"),
    ("25002", "mimic_force_diabetes"),
    ("E119", "mimic_force_diabetes"),
    ("E11", "mimic_force_diabetes"),
    # CAD / ischemic heart disease
    ("41401", "mimic_force_cad"),
    ("I2510", "mimic_force_cad"),
    ("I259", "mimic_force_cad"),
    # CHF / heart failure
    ("4280", "mimic_force_heart_failure"),
    ("42833", "mimic_force_heart_failure"),
    ("I509", "mimic_force_heart_failure"),
    ("I5032", "mimic_force_heart_failure"),  # HFrEF
    # Atrial fibrillation
    ("42731", "mimic_force_afib"),
    ("I4891", "mimic_force_afib"),
    # COPD / emphysema
    ("4960", "mimic_force_copd"),
    ("4961", "mimic_force_copd"),
    ("4968", "mimic_force_copd"),
    ("J449", "mimic_force_copd"),
    ("J440", "mimic_force_copd"),
    # CKD
    ("5853", "mimic_force_ckd"),  # stage 3
    ("5854", "mimic_force_ckd"),  # stage 4
    ("5855", "mimic_force_ckd"),  # stage 5
    ("N183", "mimic_force_ckd"),
    ("N184", "mimic_force_ckd"),
    ("N185", "mimic_force_ckd"),
    # Hypothyroidism
    ("2440", "mimic_force_hypothyroidism"),
    ("E039", "mimic_force_hypothyroidism"),
    # Hepatitis C
    ("07054", "mimic_force_hepatitis_c"),
    ("B182", "mimic_force_hepatitis_c"),
    # Liver cirrhosis / portal hypertension / ascites
    ("5715", "mimic_force_liver_cirrhosis"),
    ("5712", "mimic_force_liver_cirrhosis"),
    ("K703", "mimic_force_liver_cirrhosis"),  # alcoholic cirrhosis
    ("K744", "mimic_force_liver_cirrhosis"),  # secondary biliary cirrhosis
    ("5723", "mimic_force_liver_cirrhosis"),  # portal hypertension
    ("K766", "mimic_force_liver_cirrhosis"),  # portal hypertension (ICD-10)
    ("78959", "mimic_force_liver_cirrhosis"),  # ascites (ICD-9)
    ("R18", "mimic_force_liver_cirrhosis"),    # ascites (ICD-10)
    # Bipolar disorder
    ("2960", "mimic_force_bipolar_disorder"),
    ("2961", "mimic_force_bipolar_disorder"),
    ("2964", "mimic_force_bipolar_disorder"),
    ("2967", "mimic_force_bipolar_disorder"),
    ("F310", "mimic_force_bipolar_disorder"),
    ("F311", "mimic_force_bipolar_disorder"),
    ("F312", "mimic_force_bipolar_disorder"),
    ("F313", "mimic_force_bipolar_disorder"),
    # PTSD
    ("30981", "mimic_force_ptsd"),
    ("F431", "mimic_force_ptsd"),
    # Alcohol use disorder
    ("30390", "mimic_force_alcohol_use_disorder"),
    ("30391", "mimic_force_alcohol_use_disorder"),
    ("3050", "mimic_force_alcohol_use_disorder"),
    ("F102", "mimic_force_alcohol_use_disorder"),
    ("F101", "mimic_force_alcohol_use_disorder"),
    # Nicotine / smoking
    ("3051", "mimic_force_smoking"),
    ("V1582", "mimic_force_smoking"),
    ("F17", "mimic_force_smoking"),
    ("Z8701", "mimic_force_smoking"),
    # Obesity
    ("27800", "mimic_force_obesity"),
    ("E669", "mimic_force_obesity"),
    ("E6601", "mimic_force_obesity"),
    # Sleep apnea
    ("32723", "mimic_force_sleep_apnea"),
    ("G4733", "mimic_force_sleep_apnea"),
    # BPH
    ("60000", "mimic_force_bph"),
    ("N400", "mimic_force_bph"),
    # Stroke / TIA
    ("4369", "mimic_force_stroke"),
    ("I639", "mimic_force_stroke"),
    ("43491", "mimic_force_stroke"),  # TIA
]


def _icd_matches(icd_code: str, prefix: str) -> bool:
    return icd_code.upper().startswith(prefix.upper())


def extract_condition_priors(admission: Admission) -> dict[str, bool]:
    """Return mimic_force_X flags from this admission's secondary
    (seq_num >= 2) coded ICD diagnoses. seq_num 1 is the primary/target
    diagnosis and must never be forced as background (that would be leakage)."""
    flags: dict[str, bool] = {}
    for row in admission.get("icd_diagnoses", []):
        if int(row.get("seq_num", 1)) < 2:
            continue
        code = str(row.get("icd_code") or "").strip()
        if not code:
            continue
        for prefix, attr in _ICD_TO_FORCE_FLAG:
            if _icd_matches(code, prefix):
                flags[attr] = True
                break
    return flags
