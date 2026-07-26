from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path


def load_case(case_path: str | Path) -> dict:
    with open(case_path) as f:
        return json.load(f)


def transcript_tool_counts(transcript: dict) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for turn in transcript.get("turns", []):
        fn_name = turn.get("function") or ""
        if fn_name:
            counts[fn_name] += 1
    return dict(sorted(counts.items()))


def infer_rollout_tags(scores: dict, transcript: dict) -> list[str]:
    tags: list[str] = []
    if scores.get("diagnosis_score", 0) >= 1.0:
        tags.append("correct_diagnosis")
    elif scores.get("diagnosis_score", 0) > 0:
        tags.append("partial_diagnosis")
    else:
        tags.append("wrong_diagnosis")

    info_score = scores.get("information_collection_score", scores.get("history_collection_score", 0))
    if info_score < 0.25:
        tags.append("history_sparse")
    elif info_score >= 0.6:
        tags.append("history_good")

    if scores.get("abnormal_test_coverage", 0) < 0.25:
        tags.append("abnormal_tests_missed")

    if transcript.get("termination_reason") == "max_turns_reached":
        tags.append("max_turns_reached")

    tool_counts = transcript_tool_counts(transcript)
    if tool_counts.get("finalize_case", 0) == 0 and transcript.get("final_answer"):
        tags.append("forced_finalize")
    if tool_counts.get("order_tests", 0) == 0:
        tags.append("no_tests")
    if tool_counts.get("ask_patient", 0) == 0:
        tags.append("no_history_questions")
    return tags


def build_rollout_record(
    *,
    case_path: str | Path,
    result: dict,
    patient_provider: str,
    doctor_provider: str,
    lab_provider: str,
    split: str | None = None,
) -> dict:
    case = load_case(case_path)
    transcript = result["transcript"]
    scores = result["scores"]
    return {
        "case_id": result["case_id"],
        "case_path": str(Path(case_path).resolve()),
        "source": case.get("core_admission", {}).get("source", "unknown"),
        "split": split,
        "providers": {
            "patient_provider": patient_provider,
            "doctor_provider": doctor_provider,
            "lab_provider": lab_provider,
        },
        "ground_truth": {
            "title": scores.get("ground_truth", ""),
            "icd10_code": scores.get("ground_truth_icd10", ""),
        },
        "prediction": {
            "diagnosis": scores.get("predicted_diagnosis", ""),
            "icd10_code": scores.get("predicted_icd10", ""),
            "category": scores.get("predicted_category", ""),
            "subtype": scores.get("predicted_subtype", ""),
        },
        "scores": scores,
        "tool_counts": transcript_tool_counts(transcript),
        "tags": infer_rollout_tags(scores, transcript),
        "transcript": transcript,
    }


def summarize_rollouts(records: list[dict]) -> dict:
    if not records:
        return {"n": 0, "metrics": {}, "tags": {}, "sources": {}}

    metric_names = [
        "information_collection_score",
        "test_order_coverage",
        "abnormal_test_coverage",
        "diagnosis_score",
        "diagnosis_category_accuracy",
        "diagnosis_subtype_accuracy",
    ]
    metrics: dict[str, dict[str, float]] = {}
    for name in metric_names:
        values = [float(r["scores"].get(name, 0)) for r in records]
        metrics[name] = {
            "mean": round(statistics.fmean(values), 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
        }

    tags = Counter()
    sources = Counter()
    doctor_providers = Counter()
    for record in records:
        tags.update(record.get("tags", []))
        sources.update([record.get("source", "unknown")])
        doctor_providers.update([record["providers"]["doctor_provider"]])

    return {
        "n": len(records),
        "metrics": metrics,
        "tags": dict(tags.most_common()),
        "sources": dict(sources.most_common()),
        "doctor_providers": dict(doctor_providers.most_common()),
    }
