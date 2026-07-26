"""Consistency audit: compute clinical_consistency_score and enumerate conflicts."""

from __future__ import annotations

import re


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def _check_gender_conflict(admission: dict, persona: dict) -> list[str]:
    mimic_g = admission["demographics"].get("gender", "").upper()
    synthea_g = "M" if persona.get("gender", "").lower() == "male" else "F"
    if mimic_g and synthea_g and mimic_g != synthea_g:
        return [f"gender: MIMIC={mimic_g} Synthea={synthea_g}"]
    return []


def _age_tolerance(age: int) -> int:
    """Mirrors matcher.py's tiered tolerance and config/default.yaml's
    pipeline.age_tolerance_adult/age_tolerance_elderly (5/8) — was a flat
    10 here, inconsistent with both."""
    return 8 if age > 75 else 5


def _check_age_conflict(admission: dict, persona: dict) -> list[str]:
    mimic_age = admission["demographics"].get("anchor_age")
    synthea_age = persona.get("age")
    if mimic_age and synthea_age:
        diff = abs(mimic_age - synthea_age)
        tol = _age_tolerance(mimic_age)
        if diff > tol:
            return [f"age_gap: MIMIC={mimic_age} Synthea={synthea_age} diff={diff} tolerance={tol}"]
    return []


def _check_death_conflict(admission: dict, persona: dict, trimmed_persona: dict) -> list[str]:
    mimic_dead = bool(admission["demographics"].get("hospital_expire_flag", 0))
    synthea_dead = persona.get("deceased", False)
    post_count = len(trimmed_persona.get("encounters", []))  # already trimmed
    if mimic_dead and not synthea_dead and post_count > 10:
        return ["death_conflict: MIMIC=deceased Synthea=alive with many future encounters"]
    return []


def _check_physiological_conflicts(admission: dict, persona: dict) -> list[str]:
    conflicts = []
    gender = admission["demographics"].get("gender", "").upper()
    cond_displays = [_normalize(c["display"]) for c in persona.get("conditions", [])]
    if gender == "M" and any("pregnancy" in d or "gravid" in d for d in cond_displays):
        conflicts.append("physiology: male MIMIC patient with Synthea pregnancy condition")
    if gender == "F" and any("prostate" in d for d in cond_displays):
        conflicts.append("physiology: female MIMIC patient with Synthea prostate condition")
    return conflicts


def audit(admission: dict, persona: dict, trimmed_persona: dict, match_score: float) -> dict:
    """Run all consistency checks. Returns audit result dict."""
    conflicts = []
    conflicts += _check_gender_conflict(admission, persona)
    conflicts += _check_age_conflict(admission, persona)
    conflicts += _check_death_conflict(admission, persona, trimmed_persona)
    conflicts += _check_physiological_conflicts(admission, persona)

    # clinical_consistency_score: 1.0 minus penalty per conflict
    consistency_score = max(0.0, 1.0 - 0.2 * len(conflicts))

    # leakage_risk_score: based on number of trimmed events that were leakage
    # (will be filled in by case_builder from trimmer output)
    return {
        "match_score": round(match_score, 4),
        "clinical_consistency_score": round(consistency_score, 4),
        "leakage_risk_score": 0.0,    # updated by case_builder
        "conflict_count": len(conflicts),
        "conflicts": conflicts,
        "usable": consistency_score >= 0.6 and match_score >= 0.0,
    }
