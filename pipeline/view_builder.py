"""Build the four simulation views from admission + trimmed persona."""

from __future__ import annotations


def _social_summary(trimmed_persona: dict) -> dict:
    lifestyle_kws = ["smoking", "tobacco", "alcohol", "bmi", "body mass", "employment", "education", "marital"]
    summary = {}
    for obs in trimmed_persona.get("observations", []):
        display = obs.get("display", "")
        if any(kw in display.lower() for kw in lifestyle_kws):
            value = f"{obs.get('value', '')} {obs.get('unit', '')}".strip()
            summary[display] = value
    return summary


def _prior_encounters_summary(trimmed_persona: dict) -> str:
    encounters = trimmed_persona.get("encounters", [])
    if not encounters:
        return ""
    # Group by encounter type
    type_counts: dict[str, int] = {}
    for enc in encounters:
        t = enc.get("type", "Unknown encounter")
        type_counts[t] = type_counts.get(t, 0) + 1
    lines = [f"- {t}: {n} visit(s)" for t, n in sorted(type_counts.items(), key=lambda x: -x[1])[:10]]
    return "\n".join(lines)


def build_views(
    admission: dict,
    trimmed_persona: dict,
    personality: dict | None = None,
    patient_information_units: list[dict] | None = None,
) -> dict:
    demo = admission["demographics"]

    # ---------------------------------------------------------------
    # patient_actor_view  – used by Patient Actor LLM
    # ---------------------------------------------------------------
    patient_actor_view = {
        "initial_presentation": {
            "age": demo.get("anchor_age"),
            "gender": demo.get("gender"),
            "admission_type": demo.get("admission_type"),
            "vital_signs": admission.get("vital_signs", ""),
        },
        "symptom_context": admission.get("hpi", ""),
        "medical_background": {
            "past_medical_history": [
                c["display"] for c in trimmed_persona.get("conditions", [])
            ],
            "medications": [
                m["display"] for m in trimmed_persona.get("medications", [])
            ],
            "allergies": [
                a["substance"] for a in trimmed_persona.get("allergies", [])
            ],
            "immunizations": [
                i["vaccine"] for i in trimmed_persona.get("immunizations", [])
            ],
            "social_history": _social_summary(trimmed_persona),
        },
        "prior_encounters_summary": _prior_encounters_summary(trimmed_persona),
        "personality": personality or {},
        "patient_information_units": patient_information_units or [],
    }

    # ---------------------------------------------------------------
    # exam_provider_view  – full PE text, returned on request
    # ---------------------------------------------------------------
    exam_provider_view = {
        "physical_exam_full": admission.get("physical_exam", ""),
        "vital_signs": admission.get("vital_signs", ""),
    }

    # ---------------------------------------------------------------
    # lab_provider_view  – lookup tables for lab/micro/radiology
    # ---------------------------------------------------------------
    # Index labs by itemid for O(1) lookup
    labs_by_itemid: dict[str, list[dict]] = {}
    for lab in admission.get("labs", []):
        labs_by_itemid.setdefault(str(lab["itemid"]), []).append(lab)

    lab_provider_view = {
        "labs_by_itemid": labs_by_itemid,
        "microbiology": admission.get("microbiology", []),
        "radiology": admission.get("radiology", []),
    }

    # ---------------------------------------------------------------
    # moderator_view  – ground truth + hospital course (never shown to doctor)
    # ---------------------------------------------------------------
    moderator_view = {
        "ground_truth": admission["primary_diagnosis"],
        "hospital_expire_flag": demo.get("hospital_expire_flag", 0),
        "hospital_course": admission.get("hospital_course", ""),
        "discharge_diagnosis": admission.get("discharge_diagnosis", ""),
        "discharge_medications": admission.get("medications", []),
        "icd_procedures": admission.get("procedures", []),
        "icd_diagnoses": admission.get("icd_diagnoses", []),
        "synthea_patient_id": trimmed_persona.get("patient_id", ""),
        "synthea_name": trimmed_persona.get("name", ""),
    }

    return {
        "patient_actor_view": patient_actor_view,
        "exam_provider_view": exam_provider_view,
        "lab_provider_view": lab_provider_view,
        "moderator_view": moderator_view,
    }
