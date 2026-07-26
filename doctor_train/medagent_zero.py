from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from doctor_train.memrl import reward_from_scores


BANK_FORMAT = "hospital-medagent-zero-v1"
STAGES = ("examination", "diagnosis", "treatment")


def load_jsonl(path: str | Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_bank(path: str | Path | None) -> dict:
    if not path:
        return empty_bank()
    with open(path) as f:
        bank = json.load(f)
    if bank.get("format") != BANK_FORMAT:
        bank.setdefault("format", BANK_FORMAT)
    bank.setdefault("stages", {})
    for stage in STAGES:
        bank["stages"].setdefault(stage, {"record_library": [], "experience_base": []})
        bank["stages"][stage].setdefault("record_library", [])
        bank["stages"][stage].setdefault("experience_base", [])
    return bank


def save_bank(bank: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bank, ensure_ascii=False, indent=2), encoding="utf-8")


def empty_bank() -> dict:
    return {
        "format": BANK_FORMAT,
        "created_at": datetime.now().isoformat(),
        "description": (
            "MedAgent-Zero-style dual library: successful medical records and "
            "failure-derived clinical experience, split by clinical task stage."
        ),
        "stages": {
            stage: {"record_library": [], "experience_base": []}
            for stage in STAGES
        },
    }


def _tokenize(text: str) -> set[str]:
    text = str(text or "").lower()
    tokens = {tok for tok in re.findall(r"[\w.]+", text) if len(tok) > 1}
    cjk = "".join(re.findall(r"[\u3400-\u9fff]", text))
    tokens.update(cjk[idx : idx + 2] for idx in range(max(0, len(cjk) - 1)))
    return tokens


def _similarity(a: str, b: str) -> float:
    ta = _tokenize(a)
    tb = _tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _clip(value: Any, limit: int = 500) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + " ..."


def _initial_visible(record: dict) -> dict:
    return record.get("transcript", {}).get("initial_visible_info", {}) or {}


def query_from_record(record: dict, stage: str) -> str:
    visible = _initial_visible(record)
    parts = [
        f"age={visible.get('age', '')}",
        f"gender={visible.get('gender', '')}",
        f"admission={visible.get('admission_type', '')}",
        f"vitals={_clip(visible.get('vital_signs', ''), 240)}",
        f"chief_complaint={_clip(visible.get('chief_complaint', ''), 240)}",
    ]
    turns = record.get("transcript", {}).get("turns", [])
    if stage == "examination":
        parts.extend(_compact_turn(turn, 180) for turn in turns if turn.get("function") == "ask_patient")
    elif stage == "diagnosis":
        parts.extend(
            _compact_turn(turn, 220)
            for turn in turns
            if turn.get("function") in {"ask_patient", "request_physical_exam", "order_tests", "order_imaging"}
        )
    elif stage == "treatment":
        parts.extend(
            _compact_turn(turn, 220)
            for turn in turns
            if turn.get("function") in {"request_physical_exam", "order_tests", "order_imaging", "finalize_case"}
        )
    return "\n".join(part for part in parts if part and not part.endswith("="))


def query_from_case(case: dict, stage: str = "diagnosis") -> str:
    view = case.get("simulation_views", {}).get("patient_actor_view", {})
    init = view.get("initial_presentation", {})
    symptom_context = view.get("symptom_context", "")
    parts = [
        f"age={init.get('age', '')}",
        f"gender={init.get('gender', '')}",
        f"admission={init.get('admission_type', '')}",
        f"vitals={_clip(init.get('vital_signs', ''), 240)}",
        f"chief_complaint={_clip(_chief_complaint(symptom_context), 240)}",
        f"stage={stage}",
    ]
    return "\n".join(part for part in parts if part and not part.endswith("="))


def _chief_complaint(hpi: str) -> str:
    if not hpi:
        return ""
    for sep in (".", "\n"):
        idx = hpi.find(sep)
        if 10 < idx < 250:
            return hpi[: idx + 1].strip()
    return hpi[:200].strip()


def _compact_turn(turn: dict, max_response_chars: int = 220) -> str:
    fn_name = turn.get("function", "")
    args = {k: v for k, v in (turn.get("args", {}) or {}).items() if not str(k).startswith("_")}
    response = _clip(turn.get("response", ""), max_response_chars)
    return f"{fn_name} args={json.dumps(args, ensure_ascii=False)} -> {response}"


def _stage_success(record: dict, stage: str) -> bool:
    scores = record.get("scores", {})
    if stage == "examination":
        info = float(scores.get("information_collection_score", 0.0))
        tests = float(scores.get("test_order_coverage", 0.0))
        abnormal = float(scores.get("abnormal_test_coverage", 0.0))
        return info >= 0.7 and (tests >= 0.5 or abnormal >= 0.5)
    if stage == "diagnosis":
        return float(scores.get("diagnosis_score", 0.0)) >= 1.0
    if stage == "treatment":
        final_answer = record.get("transcript", {}).get("final_answer") or {}
        return bool(final_answer.get("treatment_plan") or final_answer.get("management_plan"))
    return False


def _record_answer(record: dict, stage: str) -> str:
    turns = record.get("transcript", {}).get("turns", [])
    if stage == "examination":
        useful = [
            turn for turn in turns
            if turn.get("function") in {"ask_patient", "request_physical_exam", "order_tests", "order_imaging"}
        ]
        return "\n".join(_compact_turn(turn, 260) for turn in useful[:10])
    if stage == "diagnosis":
        final_answer = record.get("transcript", {}).get("final_answer") or {}
        fields = [
            f"final_diagnosis={final_answer.get('final_diagnosis', '')}",
            f"icd10_code={final_answer.get('icd10_code', '')}",
            f"diagnosis_category={final_answer.get('diagnosis_category', '')}",
            f"diagnosis_subtype={final_answer.get('diagnosis_subtype', '')}",
            f"supporting_evidence={json.dumps(final_answer.get('supporting_evidence', []), ensure_ascii=False)}",
        ]
        return "\n".join(field for field in fields if not field.endswith("="))
    if stage == "treatment":
        final_answer = record.get("transcript", {}).get("final_answer") or {}
        return json.dumps(
            {
                "treatment_plan": final_answer.get("treatment_plan", ""),
                "management_plan": final_answer.get("management_plan", ""),
            },
            ensure_ascii=False,
        )
    return ""


def _failure_experience(record: dict, stage: str) -> str:
    scores = record.get("scores", {})
    tags = set(record.get("tags", []))
    lessons: list[str] = []
    if stage == "examination":
        info_score = float(scores.get("information_collection_score", scores.get("history_collection_score", 0.0)))
        if "history_sparse" in tags or info_score < 0.4:
            lessons.append("Collect a fuller symptom timeline, relevant comorbidities, medications, and risk factors before ordering broad tests.")
        if "abnormal_tests_missed" in tags or float(scores.get("abnormal_test_coverage", 0.0)) < 0.4:
            lessons.append("Order targeted tests that can expose likely abnormal findings for the leading differential, then refine after results.")
        if "no_tests" in tags:
            lessons.append("Do not finalize from history alone when objective labs, imaging, or focused exam findings are needed.")
    elif stage == "diagnosis":
        if "wrong_diagnosis" in tags:
            lessons.append("Before finalizing, reconcile the chief complaint, abnormal objective findings, and ICD-10 category; avoid anchoring on a single symptom.")
        elif "partial_diagnosis" in tags:
            lessons.append("The broad category may be close, but subtype and etiology require explicit supporting evidence and differential comparison.")
        if "forced_finalize" in tags or "max_turns_reached" in tags:
            lessons.append("Conserve turns for decisive evidence collection and commit only after the main differential has been tested.")
    elif stage == "treatment":
        lessons.append("Tie management recommendations to disease severity and the confirmed diagnosis; do not use treatment facts as diagnostic evidence.")

    if not lessons:
        lessons.append("Review the failed trajectory and extract only transferable diagnostic strategy, not patient-specific facts.")
    return "\n".join(f"- {lesson}" for lesson in lessons)


def _metadata(record: dict) -> dict:
    return {
        "case_id": record.get("case_id", ""),
        "source": record.get("source", ""),
        "split": record.get("split"),
        "reward": reward_from_scores(record.get("scores", {})),
        "tags": record.get("tags", []),
        "ground_truth_icd10": record.get("ground_truth", {}).get("icd10_code", ""),
        "predicted_icd10": record.get("prediction", {}).get("icd10_code", ""),
        "created_at": datetime.now().isoformat(),
    }


def entries_from_record(record: dict) -> dict[str, dict[str, list[dict]]]:
    entries: dict[str, dict[str, list[dict]]] = {
        stage: {"record_library": [], "experience_base": []}
        for stage in STAGES
    }
    for stage in STAGES:
        query = query_from_record(record, stage)
        if not query:
            continue
        base = {
            "id": f"{stage}_{record.get('case_id', 'unknown')}",
            "stage": stage,
            "query": query,
            "metadata": _metadata(record),
        }
        if _stage_success(record, stage):
            answer = _record_answer(record, stage)
            if answer:
                entries[stage]["record_library"].append({**base, "answer": answer})
        else:
            entries[stage]["experience_base"].append({
                **base,
                "experience": _failure_experience(record, stage),
                "validation": "not_retested",
            })
    return entries


def add_or_replace_record(bank: dict, record: dict) -> dict[str, int]:
    counts = {"record_library": 0, "experience_base": 0}
    entries = entries_from_record(record)
    for stage, libraries in entries.items():
        stage_bank = bank.setdefault("stages", {}).setdefault(stage, {"record_library": [], "experience_base": []})
        for library_name, new_items in libraries.items():
            items = stage_bank.setdefault(library_name, [])
            for new_item in new_items:
                replaced = False
                for idx, existing in enumerate(items):
                    if existing.get("id") == new_item["id"]:
                        items[idx] = new_item
                        replaced = True
                        break
                if not replaced:
                    items.append(new_item)
                counts[library_name] += 1
    return counts


def build_bank(records: list[dict]) -> dict:
    bank = empty_bank()
    seen: set[str] = set()
    for record in records:
        case_id = str(record.get("case_id", ""))
        if case_id in seen:
            continue
        seen.add(case_id)
        add_or_replace_record(bank, record)
    return bank


def retrieve(
    bank: dict,
    query: str,
    *,
    stage: str = "diagnosis",
    top_k_records: int = 3,
    top_k_experiences: int = 3,
    min_similarity: float = 0.0,
) -> dict[str, list[dict]]:
    stage_bank = bank.get("stages", {}).get(stage, {})
    output = {"record_library": [], "experience_base": []}
    for library_name, top_k in (("record_library", top_k_records), ("experience_base", top_k_experiences)):
        ranked = []
        for item in stage_bank.get(library_name, []):
            sim = _similarity(query, item.get("query", ""))
            if sim >= min_similarity:
                enriched = dict(item)
                enriched["retrieval"] = {"similarity": round(sim, 4), "stage": stage, "library": library_name}
                ranked.append(enriched)
        ranked.sort(key=lambda item: item["retrieval"]["similarity"], reverse=True)
        output[library_name] = ranked[: max(0, top_k)]
    return output


def format_context(retrieved: dict[str, list[dict]], *, language: str = "en") -> str:
    records = retrieved.get("record_library", [])
    experiences = retrieved.get("experience_base", [])
    if not records and not experiences:
        return ""
    if language == "zh":
        header = (
            "== MedAgent-Zero 既往病例与经验 ==\n"
            "以下内容来自相似训练病例。成功病例可作为诊疗思路参考；失败经验只提供可迁移原则。"
            "不要把既往病例的诊断、检查结果或患者事实当作当前病例事实。"
        )
        record_label = "成功病例记录"
        exp_label = "失败反思经验"
    else:
        header = (
            "== MedAgent-Zero Retrieved Records and Experience ==\n"
            "The following comes from similar training cases. Successful records are examples; "
            "failure-derived experience is transferable guidance only. Do not copy diagnoses, "
            "test results, or patient facts into the current case."
        )
        record_label = "Successful Medical Record"
        exp_label = "Failure-Derived Experience"

    blocks = [header]
    for idx, item in enumerate(records, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[{record_label} {idx}] sim={item.get('retrieval', {}).get('similarity', 0.0):.3f}",
                    f"Query:\n{_clip(item.get('query', ''), 900)}",
                    f"Answer/trajectory:\n{_clip(item.get('answer', ''), 1200)}",
                ]
            )
        )
    for idx, item in enumerate(experiences, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[{exp_label} {idx}] sim={item.get('retrieval', {}).get('similarity', 0.0):.3f}",
                    f"Trigger:\n{_clip(item.get('query', ''), 700)}",
                    f"Principles:\n{_clip(item.get('experience', ''), 900)}",
                ]
            )
        )
    return "\n\n".join(blocks)
