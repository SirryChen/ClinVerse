"""Trim Synthea persona timeline: remove post-index events and leakage-risk items."""

from __future__ import annotations

import re
from datetime import datetime


def _to_date(s: str) -> datetime | None:
    if not s or not s.strip():
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s[:19], fmt[:min(19, len(fmt))])
        except (ValueError, TypeError):
            pass
    return None


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def _dedup_by_display(items: list[dict]) -> list[dict]:
    """Drop repeat entries with the same display text (keep first occurrence).

    Synthea emits a separate Condition/MedicationRequest resource per
    encounter, so administrative conditions like "Medication review due"
    that recur at every wellness visit otherwise show up many times in
    past_medical_history/long_term_medications for no informational gain.
    """
    seen: set[str] = set()
    out = []
    for item in items:
        key = _normalize(item.get("display", "") or item.get("vaccine", ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _leakage_risk(item_display: str, diagnosis_title: str, icd_code: str) -> bool:
    """Return True if a Synthea condition/encounter seems to directly name the diagnosis.

    Threshold raised to ≥3 content words so generic comorbidities (e.g. "liver
    disease", "hepatitis") are not removed when only 1-2 words overlap with a
    specific ICD title like "Alcoholic cirrhosis of liver".
    """
    diag_words = set(re.findall(r"\w{4,}", _normalize(diagnosis_title)))
    item_words  = set(re.findall(r"\w{4,}", _normalize(item_display)))
    overlap = diag_words & item_words
    stopwords = {"with", "without", "unspecified", "chronic", "acute", "disease", "disorder", "syndrome"}
    content_overlap = overlap - stopwords
    return len(content_overlap) >= 3


def trim_persona(persona: dict, index_date: str, diagnosis: dict) -> dict:
    """Return a trimmed copy of persona with leakage and post-index events removed.

    Returns dict with keys:
      - trimmed_persona: the sanitized background
      - trimmed_events: list of {resource_type, display, reason, leakage_risk}
    """
    diag_title = diagnosis.get("long_title", "")
    icd_code   = diagnosis.get("icd_code", "")
    idx_dt = _to_date(index_date)

    trimmed_events = []

    def keep_condition(c: dict) -> bool:
        onset = _to_date(c.get("onset_date", ""))
        # remove post-index conditions
        if idx_dt and onset and onset > idx_dt:
            trimmed_events.append({
                "resource_type": "Condition",
                "display": c["display"],
                "reason": "post_index_time",
                "leakage_risk": False,
            })
            return False
        # remove leakage-risk conditions
        if _leakage_risk(c["display"], diag_title, icd_code):
            trimmed_events.append({
                "resource_type": "Condition",
                "display": c["display"],
                "reason": "leakage_risk",
                "leakage_risk": True,
            })
            return False
        return True

    def keep_encounter(e: dict) -> bool:
        start = _to_date(e.get("start", ""))
        if idx_dt and start and start > idx_dt:
            trimmed_events.append({
                "resource_type": "Encounter",
                "display": e.get("type", ""),
                "reason": "post_index_time",
                "leakage_risk": False,
            })
            return False
        if _leakage_risk(e.get("type", ""), diag_title, icd_code):
            trimmed_events.append({
                "resource_type": "Encounter",
                "display": e.get("type", ""),
                "reason": "leakage_risk",
                "leakage_risk": True,
            })
            return False
        return True

    def keep_medication(m: dict) -> bool:
        authored = _to_date(m.get("authored_date", ""))
        if idx_dt and authored and authored > idx_dt:
            trimmed_events.append({
                "resource_type": "MedicationRequest",
                "display": m["display"],
                "reason": "post_index_time",
                "leakage_risk": False,
            })
            return False
        return True

    def keep_observation(o: dict) -> bool:
        obs_dt = _to_date(o.get("date", ""))
        if idx_dt and obs_dt and obs_dt > idx_dt:
            return False
        return True

    trimmed_persona = {
        **persona,
        "conditions":    _dedup_by_display([c for c in persona.get("conditions", [])    if keep_condition(c)]),
        "medications":   _dedup_by_display([m for m in persona.get("medications", [])    if keep_medication(m)]),
        "encounters":    [e for e in persona.get("encounters", [])     if keep_encounter(e)],
        "observations":  [o for o in persona.get("observations", [])  if keep_observation(o)],
        "allergies":     persona.get("allergies", []),
        "immunizations": _dedup_by_display(persona.get("immunizations", [])),
    }

    return {
        "trimmed_persona": trimmed_persona,
        "trimmed_events": trimmed_events,
    }


def trim_temporal_only(persona: dict, index_date: str) -> dict:
    """Return a persona with only post-index events removed (no leakage filtering).

    Used during matching so that feature vectors reflect only the history that
    will actually be available after the full trim, avoiding inflated match scores
    caused by future conditions.
    """
    idx_dt = _to_date(index_date)
    if idx_dt is None:
        return persona

    def before_index(date_str: str) -> bool:
        dt = _to_date(date_str)
        return dt is None or dt <= idx_dt

    return {
        **persona,
        "conditions":   [c for c in persona.get("conditions", [])   if before_index(c.get("onset_date", ""))],
        "medications":  [m for m in persona.get("medications", [])   if before_index(m.get("authored_date", ""))],
        "encounters":   [e for e in persona.get("encounters", [])    if before_index(e.get("start", ""))],
        "observations": [o for o in persona.get("observations", []) if before_index(o.get("date", ""))],
        "allergies":    persona.get("allergies", []),
        "immunizations": persona.get("immunizations", []),
    }
