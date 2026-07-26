"""SICdb comorbidity-prior extraction.

SICdb's icd_diagnosis.csv has at most ONE coded ICD entry per admission (the
primary diagnosis) — DiagnosisT2, the only "secondary diagnosis" field, is
free text with no ICD code (see data/sicdb-ext/predata/loader.py). Reusing
MIMIC's approach (scan seq_num>=2 *coded* rows) would therefore always
return {} for every SICdb case, silently disabling Synthea condition
injection. The actual comorbidity signal SICdb has is the boolean
Precondition*/Premedication*/NYHA fields from data_ref, exported by predata
into comorbidity_flags.csv and attached to each Admission by
adapters/sicdb_ext/adapter.py as admission["_sicdb_comorbidities"].
"""
from __future__ import annotations

from core.schema import Admission

# data_ref field name -> Synthea mimic_force_X attribute.
# Field values are already pre-filtered by predata's clinical._meaningful()
# (junk/"No"/"Unknown" values never make it into comorbidity_flags.csv), so
# presence of the key is itself informative for the boolean fields below —
# we still check the value defensively rather than assume.
_FIELD_TO_FORCE_FLAG: dict[str, str] = {
    "PreconditionDiabetes":         "mimic_force_diabetes",
    "PreconditionArtHypertension":  "mimic_force_hypertension",
    "PreconditionLungDisease":      "mimic_force_copd",
    "PreconditionRenalDysfunction": "mimic_force_ckd",
    "PremedicationStatin":          "mimic_force_hyperlipidemia",
    "PremedicationACEInhibitor":    "mimic_force_hypertension",
    "PremedicationBetablocker":     "mimic_force_hypertension",
}

_TRUTHY = {"yes", "ja", "true", "1"}


def extract_condition_priors(admission: Admission) -> dict[str, bool]:
    comorbidities: dict[str, str] = admission.get("_sicdb_comorbidities", {})  # type: ignore[assignment]
    if not comorbidities:
        return {}

    flags: dict[str, bool] = {}
    for field, attr in _FIELD_TO_FORCE_FLAG.items():
        value = str(comorbidities.get(field, "")).strip().lower()
        if value in _TRUTHY:
            flags[attr] = True

    nyha = comorbidities.get("NYHA")
    if nyha is not None:
        try:
            if int(str(nyha).strip()) >= 2:
                flags["mimic_force_heart_failure"] = True
        except ValueError:
            pass

    return flags
