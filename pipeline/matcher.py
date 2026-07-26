"""Hard-filter + soft-score Synthea personas against a MIMIC admission."""

from __future__ import annotations

import numpy as np

from .feature_extractor import extract_mimic_features, extract_synthea_features
from .trimmer import trim_temporal_only


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    intersection = float(np.sum(np.minimum(a, b)))
    union = float(np.sum(np.maximum(a, b)))
    return intersection / union if union > 0 else 0.0


def _age_tolerance(age: int) -> int:
    return 8 if age > 75 else 5


def hard_filter(admission: dict, persona: dict) -> bool:
    """Return True if persona passes all hard constraints."""
    demo = admission["demographics"]
    mimic_gender = demo.get("gender", "").upper()
    synthea_gender = "M" if persona.get("gender", "").lower() == "male" else "F"
    if mimic_gender and synthea_gender and mimic_gender != synthea_gender:
        return False

    mimic_age = demo.get("anchor_age")
    synthea_age = persona.get("age")
    if mimic_age and synthea_age:
        tol = _age_tolerance(mimic_age)
        if abs(mimic_age - synthea_age) > tol:
            return False

    mimic_deceased = bool(demo.get("hospital_expire_flag", 0))
    synthea_deceased = persona.get("deceased", False)
    if mimic_deceased and not synthea_deceased and len(persona.get("encounters", [])) > 5:
        return False

    return True


def score_match(
    admission: dict,
    persona: dict,
    mf: dict | None = None,
    pf: dict | None = None,
) -> dict:
    """Compute soft match score between admission and persona. Returns score breakdown."""
    if mf is None:
        mf = extract_mimic_features(admission)
    if pf is None:
        pf = extract_synthea_features(persona)

    # age distance score (0–1, higher is better)
    mimic_age = mf["age"] or 50
    synthea_age = pf["age"] or 50
    age_dist = abs(mimic_age - synthea_age)
    tol = _age_tolerance(mimic_age)
    age_score = max(0.0, 1.0 - age_dist / tol) if tol > 0 else 0.0

    gender_score = 1.0 if mf["gender_m"] == pf["gender_m"] else 0.0

    race_score = _jaccard(mf["race"], pf["race"])

    demo_score = 0.5 * age_score + 0.3 * gender_score + 0.2 * race_score

    cond_score = _jaccard(mf["conditions"], pf["conditions"])
    med_score  = _jaccard(mf["medications"], pf["medications"])
    risk_score = _jaccard(mf["risk_factors"], pf["risk_factors"])

    # conflict: deceased mismatch
    conflict_penalty = 0.0
    if mf["deceased"] != pf["deceased"]:
        conflict_penalty = 0.3

    w = dict(demo=0.25, cond=0.30, med=0.20, risk=0.15, conflict=0.10)
    match_score = (
        w["demo"] * demo_score
        + w["cond"] * cond_score
        + w["med"]  * med_score
        + w["risk"] * risk_score
        - w["conflict"] * conflict_penalty
    )
    match_score = max(0.0, min(1.0, match_score))

    return {
        "match_score": round(match_score, 4),
        "demographics_similarity": round(demo_score, 4),
        "condition_similarity": round(cond_score, 4),
        "medication_similarity": round(med_score, 4),
        "risk_factor_similarity": round(risk_score, 4),
        "conflict_penalty": round(conflict_penalty, 4),
        "leakage_penalty": 0.0,       # computed in trimmer after context review
        "leakage_risk_score": 0.0,
        "conflict_count": 1 if conflict_penalty > 0 else 0,
    }


def match_admission(
    admission: dict,
    personas: list[dict],
    top_k: int = 3,
    min_score: float = 0.0,
) -> list[dict]:
    """Return top-k Synthea personas ranked by match score (descending).
    Each element: {synthea_patient_id, rank, persona, **score_breakdown}
    """
    mimic_feat = extract_mimic_features(admission)
    index_date = admission["demographics"].get("admittime", "")

    candidates = []
    for persona in personas:
        if not hard_filter(admission, persona):
            continue
        # Trim to pre-index history before scoring so match_score reflects only
        # the background that will actually be available after full trim.
        pre_trimmed = trim_temporal_only(persona, index_date)
        synthea_feat = extract_synthea_features(pre_trimmed)
        breakdown = score_match(admission, pre_trimmed, mimic_feat, synthea_feat)
        if breakdown["match_score"] >= min_score:
            candidates.append({
                "synthea_patient_id": persona["patient_id"],
                "persona": persona,
                **breakdown,
            })

    candidates.sort(key=lambda x: x["match_score"], reverse=True)
    ranked = []
    for rank, c in enumerate(candidates[:top_k], start=1):
        c["rank"] = rank
        ranked.append(c)
    return ranked
