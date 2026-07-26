"""Dataset-agnostic half of Synthea conditioning priors.

Splits what used to be pipeline/mimic_prior_extractor.py into:
  - hard demographic priors + forbidden clues: generic, live here (they only
    read admission["demographics"] / admission["primary_diagnosis"], which
    every adapter populates identically via core.schema.admission_from_parts)
  - comorbidity condition flags: genuinely dataset-specific (MIMIC has coded
    secondary ICD diagnoses; SICdb has boolean precondition fields instead),
    so that part is adapter.extract_condition_priors() — see
    adapters/mimic_iv_ext/condition_priors.py and adapters/sicdb_ext/condition_priors.py.
"""
from __future__ import annotations

from typing import Any

from core.ports import DatasetAdapter
from core.schema import Admission


def extract_hard_priors(admission: Admission) -> dict[str, Any]:
    """Return demographic priors for Synthea fixedRecordPath Entity JSON.

    DOB is approximated as Jan 1 of (reference_year - anchor_age) because
    real admission dates are unavailable in every adapter so far (MIMIC
    de-identifies them; SICdb only has second-offsets from an undisclosed
    reference, not a wall-clock date — see core.schema.Demographics.admittime).
    """
    demo = admission.get("demographics", {})
    anchor_age = demo.get("anchor_age") or 50
    reference_year = 2020
    birth_year = reference_year - int(anchor_age)
    date_of_birth = f"{birth_year}-06-15"  # midyear approximation

    gender = "M" if str(demo.get("gender", "")).upper() in ("M", "MALE") else "F"
    state = _pick_state()

    return {
        "date_of_birth": date_of_birth,
        "gender": gender,
        "state": state,
        "anchor_age": int(anchor_age),
        "race": str(demo.get("race") or ""),
        "mimic_hadm_id": admission.get("hadm_id", ""),
        "mimic_subject_id": admission.get("subject_id", ""),
    }


def _pick_state() -> str:
    """Synthea only ships US population modules, so every adapter's core
    record gets paired with a US-synthetic background regardless of the
    source hospital's actual country (Boston for MIMIC, Salzburg/Austria for
    SICdb, ...) — a known, accepted limitation, not something this function
    can fix. Default to Massachusetts everywhere for now."""
    return "Massachusetts"


def extract_forbidden_clues(admission: Admission) -> list[str]:
    """Return ICD disease names that are leakage-adjacent to the primary
    diagnosis and must be suppressed from the generated persona's background
    (pipeline/trimmer.py does the actual filtering)."""
    primary = admission.get("primary_diagnosis", {})
    return [primary.get("long_title", "")]


def extract_all_priors(admission: Admission, adapter: DatasetAdapter) -> dict:
    """Top-level function: extract all priors needed for conditioned Synthea
    generation, combining the generic hard_priors/forbidden_clues above with
    this admission's dataset-specific condition_flags.

    Returns:
        {
          "hard_priors": {...demographics...},
          "condition_flags": {"mimic_force_hypertension": True, ...},
          "forbidden_clues": ["Primary Diagnosis Title"],
          "hadm_id": "...",
          "subject_id": "...",
          "n_conditions_forced": int,
        }
    """
    condition_flags = adapter.extract_condition_priors(admission)
    hard_priors = extract_hard_priors(admission)
    forbidden = extract_forbidden_clues(admission)

    return {
        "hard_priors": hard_priors,
        "condition_flags": condition_flags,
        "forbidden_clues": forbidden,
        "hadm_id": admission.get("hadm_id", ""),
        "subject_id": admission.get("subject_id", ""),
        "n_conditions_forced": len(condition_flags),
    }
