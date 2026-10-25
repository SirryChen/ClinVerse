"""Compute all evaluation scores from a completed transcript."""

from __future__ import annotations

from pipeline.patient_info_units import require_patient_information_units

from .metrics import (
    history_collection_score,
    test_order_coverage,
    abnormal_test_coverage,
    diagnosis_category_accuracy,
    diagnosis_subtype_accuracy,
    information_collection_score,
)

_DEFAULT_WEIGHTS = {
    "scoring": {
        "diagnosis_subweights": {"category": 0.5, "subtype": 0.5},
    }
}


def _patient_information_units(case: dict) -> list[dict]:
    return require_patient_information_units(case)


def _extend_disclosed_info_ids(target: list[str], meta: dict) -> None:
    target.extend(meta.get("disclosed_info_ids", []))


def score_transcript(transcript, case: dict, config: dict | None = None) -> dict:
    """Compute full evaluation scores from transcript + case.

    transcript: Transcript object or dict (via to_dict()).
    """
    cfg = (config or {}).get("scoring", _DEFAULT_WEIGHTS["scoring"])

    patient_info_units = _patient_information_units(case)
    evidence_inventory = case.get("evidence_inventory", [])

    if hasattr(transcript, "turns"):
        turns = transcript.turns
        final_answer = transcript.final_answer or {}
        collected_ids: list[str] = []
        returned_ids: list[str] = []
        returned_abnormal_ids: list[str] = []
        for t in turns:
            meta = t.hidden_metadata
            _extend_disclosed_info_ids(collected_ids, meta)
            returned_ids.extend(meta.get("returned_evidence_ids", []))
            returned_abnormal_ids.extend(meta.get("returned_abnormal_evidence_ids", []))
    else:
        # dict form
        final_answer = transcript.get("final_answer") or {}
        collected_ids = []
        returned_ids = []
        returned_abnormal_ids = []
        for t in transcript.get("turns", []):
            meta = t.get("hidden_metadata", {})
            _extend_disclosed_info_ids(collected_ids, meta)
            returned_ids.extend(meta.get("returned_evidence_ids", []))
            returned_abnormal_ids.extend(meta.get("returned_abnormal_evidence_ids", []))

    gt = case["simulation_views"]["moderator_view"]["ground_truth"]
    gt_title = gt.get("long_title", "")
    # Always ICD-10 regardless of source dataset — see core.schema.admission_from_parts
    # and core.icd.to_icd10() (converts MIMIC's ICD-9 ground truth too).
    gt_icd10 = gt.get("icd10_code", "")

    h_score = history_collection_score(collected_ids, patient_info_units)
    t_score = test_order_coverage(returned_ids, evidence_inventory)
    a_score = abnormal_test_coverage(returned_abnormal_ids, evidence_inventory)
    info_score = information_collection_score(h_score)

    predicted_icd10 = final_answer.get("icd10_code", "")
    cat_acc  = diagnosis_category_accuracy(predicted_icd10, gt_icd10)
    sub_acc  = diagnosis_subtype_accuracy(predicted_icd10, gt_icd10)
    if not cat_acc:
        # Block-level mismatch means the finer 3-char match can't count either.
        sub_acc = 0
    dsw = cfg.get("diagnosis_subweights", {"category": 0.5, "subtype": 0.5})
    diag_score = round(
        dsw.get("category", 0.5) * cat_acc + dsw.get("subtype", 0.5) * sub_acc, 4
    )

    return {
        "history_collection_score": h_score,
        "test_order_coverage": t_score,
        "abnormal_test_coverage": a_score,
        "information_collection_score": info_score,
        "diagnosis_category_accuracy": cat_acc,
        "diagnosis_subtype_accuracy": sub_acc,
        "diagnosis_score": diag_score,
        "ground_truth": gt_title,
        "ground_truth_icd10": gt_icd10,
        "predicted_icd10": predicted_icd10,
        "predicted_category": final_answer.get("diagnosis_category", ""),
        "predicted_subtype": final_answer.get("diagnosis_subtype", ""),
        "predicted_diagnosis": final_answer.get("final_diagnosis", ""),
    }
