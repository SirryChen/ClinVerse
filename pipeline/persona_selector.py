"""Select the best Synthea persona from conditioned generation candidates.

Scores candidates on:
  1. Prior satisfaction: how many forced conditions appear in the generated persona
  2. Safety: no forbidden clues (leakage of primary diagnosis)
  3. Utility: richness of background history (more conditions = more realistic)
"""

from __future__ import annotations

import re

from .trimmer import trim_persona


# ---------------------------------------------------------------------------
# Condition flag → expected Synthea condition display keyword mapping
# ---------------------------------------------------------------------------
_FLAG_TO_KEYWORDS: dict[str, list[str]] = {
    "mimic_force_hypertension": ["hypertension", "hypertensive"],
    "mimic_force_hyperlipidemia": ["hyperlipidemia", "hypercholesterolemia", "cholesterol"],
    "mimic_force_gerd": ["reflux", "gerd", "esophagitis"],
    "mimic_force_depression": ["depressive", "depression", "mdd"],
    "mimic_force_anxiety": ["anxiety", "anxious", "generalized"],
    "mimic_force_diabetes": ["diabetes", "diabetic"],
    "mimic_force_cad": ["coronary", "atherosclerosis", "ischemic heart"],
    "mimic_force_heart_failure": ["heart failure", "cardiac failure", "chf"],
    "mimic_force_afib": ["atrial fibrillation", "afib", "atrial flutter"],
    "mimic_force_copd": ["copd", "emphysema", "chronic obstructive"],
    "mimic_force_ckd": ["chronic kidney", "ckd", "renal failure"],
    "mimic_force_hypothyroidism": ["hypothyroid", "thyroid"],
    "mimic_force_hepatitis_c": ["hepatitis c", "hcv", "chronic hepatitis"],
    "mimic_force_liver_cirrhosis": ["cirrhosis", "portal hypertension", "ascites"],
    "mimic_force_bipolar_disorder": ["bipolar", "manic"],
    "mimic_force_ptsd": ["posttraumatic", "ptsd", "post-traumatic"],
    "mimic_force_alcohol_use_disorder": ["alcohol", "alcoholism", "aud"],
    "mimic_force_smoking": ["tobacco", "nicotine", "smoking", "smoker"],
    "mimic_force_obesity": ["obese", "obesity", "overweight"],
    "mimic_force_sleep_apnea": ["sleep apnea", "apnea"],
    "mimic_force_bph": ["prostatic hyperplasia", "bph", "benign prostatic"],
    "mimic_force_stroke": ["stroke", "cerebral infarction", "tia", "transient ischemic"],
}


def _condition_displays(persona: dict) -> list[str]:
    return [c.get("display", "").lower() for c in persona.get("conditions", [])]


def _medication_displays(persona: dict) -> list[str]:
    return [m.get("display", "").lower() for m in persona.get("medications", [])]


def prior_satisfaction_score(persona: dict, condition_flags: dict[str, bool]) -> float:
    """Return fraction of forced conditions that appear in the generated persona."""
    if not condition_flags:
        return 1.0

    all_text = " ".join(_condition_displays(persona) + _medication_displays(persona))
    matched = 0
    for flag, forced in condition_flags.items():
        if not forced:
            continue
        keywords = _FLAG_TO_KEYWORDS.get(flag, [])
        if any(kw in all_text for kw in keywords):
            matched += 1

    return matched / len([v for v in condition_flags.values() if v])


def safety_score(persona: dict, forbidden_clues: list[str]) -> float:
    """Return 0.0 if persona contains any forbidden clue words, else 1.0."""
    all_text = " ".join(_condition_displays(persona))
    for clue in forbidden_clues:
        clue_words = set(re.findall(r"\w{4,}", clue.lower()))
        text_words = set(re.findall(r"\w{4,}", all_text))
        overlap = clue_words & text_words - {"with", "without", "chronic", "acute",
                                              "unspecified", "disorder", "disease"}
        if len(overlap) >= 3:
            return 0.0
    return 1.0


def utility_score(persona: dict) -> float:
    """Reward richer backgrounds."""
    n_cond = min(len(persona.get("conditions", [])), 15)
    n_med = min(len(persona.get("medications", [])), 10)
    return (n_cond / 15.0) * 0.6 + (n_med / 10.0) * 0.4


def score_candidate(
    persona: dict,
    condition_flags: dict[str, bool],
    forbidden_clues: list[str],
    weights: dict[str, float] | None = None,
) -> dict:
    if weights is None:
        weights = {"prior_satisfaction": 0.50, "safety": 0.30, "utility": 0.20}

    sat = prior_satisfaction_score(persona, condition_flags)
    safe = safety_score(persona, forbidden_clues)
    util = utility_score(persona)

    composite = (
        weights["prior_satisfaction"] * sat
        + weights["safety"] * safe
        + weights["utility"] * util
    )

    return {
        "prior_satisfaction": round(sat, 4),
        "safety": round(safe, 4),
        "utility": round(util, 4),
        "composite": round(composite, 4),
        "usable": safe > 0.0 and composite >= 0.3,
    }


def select_persona(
    admission: dict,
    generation_result: dict,
    top_k: int = 3,
) -> list[dict]:
    """Select top-k candidates from conditioned generation output.

    Returns an empty list if generation produced no usable personas
    (both Tier 1 and Tier 2 failed). The caller should skip the case.
    """
    personas = generation_result.get("personas", [])
    priors = generation_result.get("priors", {})
    mode = generation_result.get("mode", "skipped")
    condition_flags = priors.get("condition_flags", {})
    forbidden_clues = priors.get("forbidden_clues", [])
    index_date = admission.get("demographics", {}).get("admittime", "")

    candidates = []
    for persona in personas:
        trimmed_persona = trim_persona(persona, index_date,
                                       admission.get("primary_diagnosis", {}))
        scored = score_candidate(
            trimmed_persona["trimmed_persona"],
            condition_flags,
            forbidden_clues,
        )
        if scored["usable"]:
            candidates.append({
                "persona": persona,
                "trimmed_persona": trimmed_persona["trimmed_persona"],
                "trimmed_events": trimmed_persona["trimmed_events"],
                "mode": mode,
                "priors": priors,
                **scored,
            })

    candidates.sort(key=lambda x: x["composite"], reverse=True)
    for rank, c in enumerate(candidates[:top_k], start=1):
        c["rank"] = rank
    return candidates[:top_k]
