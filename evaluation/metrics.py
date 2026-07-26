"""Individual scoring metric functions."""

from __future__ import annotations

from core import icd as icd_mod

# Synthea's social-determinants-of-health modules tag administrative/SDOH
# findings onto a persona's "conditions" list alongside real diagnoses. These
# are excluded from collectible history so they do not pad the denominator.
_NON_CLINICAL_FINDINGS = {
    "Risk activity involvement (finding)",
    "Received higher education (finding)",
    "Has a criminal record (finding)",
    "Full-time employment (finding)",
    "Part-time employment (finding)",
    "Unemployed (finding)",
    "Victim of intimate partner abuse (finding)",
    "Medication review due (situation)",
    "Stress (finding)",
    "Limited social contact (finding)",
    "Social isolation (finding)",
}


_INFO_WEIGHTS = {
    "key": 3.0,
    "supporting": 2.0,
    "background": 1.0,
}

_COLLECTIBLE_INFO_CATEGORIES = {
    "chief_complaint",
    "symptom",
    "negative_symptom",
    "past_medical_history",
    "surgical_history",
    "family_history",
    "social_history",
    "medication",
    "allergy",
    "immunization",
    "profile",
    "demographic_profile",
}


def _info_id(item: dict) -> str:
    return item.get("info_id") or item.get("evidence_id") or ""


def _info_text(item: dict) -> str:
    return item.get("text") or item.get("content") or ""


def _info_category(item: dict) -> str:
    return item.get("category") or item.get("type") or ""


def _info_weight(item: dict) -> float:
    return _INFO_WEIGHTS.get(str(item.get("importance", "supporting")).lower(), 2.0)


def history_collection_score(collected_ids: list[str], patient_information_units: list[dict]) -> float:
    """Weighted fraction of collectible patient-known atomic information units.

    The denominator is the integrated patient information set, not raw MIMIC
    test evidence. Initially visible units are excluded because the doctor did
    not need to elicit them through questioning.
    """
    available = [
        item for item in patient_information_units
        if _info_id(item)
        and not item.get("initially_visible")
        and item.get("patient_visible", True)
        and _info_category(item) in _COLLECTIBLE_INFO_CATEGORIES
        and _info_text(item) not in _NON_CLINICAL_FINDINGS
    ]
    if not available:
        return 1.0
    collected = set(collected_ids)
    total = sum(_info_weight(item) for item in available)
    got = sum(_info_weight(item) for item in available if _info_id(item) in collected)
    return round(got / total, 4) if total else 1.0


def test_order_coverage(returned_ids: list[str], evidence_inventory: list[dict]) -> float:
    """Fraction of available test/imaging evidence revealed by the doctor's orders."""
    available = [
        item for item in evidence_inventory
        if item.get("evidence_id")
        and not item.get("initially_visible")
        and item.get("type") in ("lab", "microbiology", "radiology")
    ]
    if not available:
        return 1.0
    available_ids = {item["evidence_id"] for item in available}
    ordered = {eid for eid in returned_ids if eid in available_ids}
    return round(len(ordered) / len(available_ids), 4)


def abnormal_test_coverage(returned_abnormal_ids: list[str], evidence_inventory: list[dict]) -> float:
    """Fraction of abnormal tests the doctor successfully surfaced."""
    abnormal = [
        item for item in evidence_inventory
        if item.get("evidence_id")
        and not item.get("initially_visible")
        and item.get("is_abnormal")
        and item.get("type") in ("lab", "microbiology")
    ]
    if not abnormal:
        return 1.0
    abnormal_ids = {item["evidence_id"] for item in abnormal}
    revealed = {eid for eid in returned_abnormal_ids if eid in abnormal_ids}
    return round(len(revealed) / len(abnormal_ids), 4)


def diagnosis_category_accuracy(predicted_icd10_code: str, ground_truth_icd10_code: str) -> int:
    """1 if the doctor's predicted ICD-10 code falls in the same "block" as
    the ground truth (e.g. both in 'I30-I5A | Other forms of heart disease'),
    else 0.

    This used to be an English keyword-overlap match against a free-text
    category label (e.g. "heart_failure") against the ground-truth title —
    which only worked when both were in English and happened to share
    vocabulary (see git history / _CATEGORY_ALIASES if you need the old
    behaviour). Comparing by ICD-10 code instead works identically for any
    dataset/language: MIMIC's English titles, SICdb's German titles, and
    any future dataset, because the doctor's icd10_code answer (see
    agents/tools.py finalize_case) is always normalized to ICD-10 the same
    way the ground truth is (core.schema.admission_from_parts).
    """
    if not predicted_icd10_code or not ground_truth_icd10_code:
        return 0
    return 1 if icd_mod.same_block(predicted_icd10_code, ground_truth_icd10_code) else 0


def diagnosis_subtype_accuracy(predicted_icd10_code: str, ground_truth_icd10_code: str) -> int:
    """1 if the doctor's predicted ICD-10 code shares the same 3-character
    category prefix as the ground truth (e.g. both 'I50' for heart failure
    subtypes), else 0. Finer-grained than diagnosis_category_accuracy's
    block-level match."""
    if not predicted_icd10_code or not ground_truth_icd10_code:
        return 0
    return 1 if icd_mod.same_category3(predicted_icd10_code, ground_truth_icd10_code) else 0


def information_collection_score(history_score: float) -> float:
    """Information collection is currently history-only.

    Test ordering and abnormal-result coverage are intentionally excluded from
    this score; they can be reintroduced later as separate metrics.
    """
    return round(history_score, 4)
