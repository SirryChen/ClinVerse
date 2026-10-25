#!/usr/bin/env python3
"""2x2 ablation of ClinVerse's patient-information architecture and prompt.

The experiment crosses information access (dual-layer versus a single
all-clinical-information simulator) with response policy (neutral versus
availability self-check). Moderator data, ground-truth diagnoses, discharge
diagnoses, and hospital course are excluded from every arm.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from openai import OpenAI

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
CLINVERSE_ROOT = REPO_ROOT
sys.path.insert(0, str(CLINVERSE_ROOT))
from core.config import load_yaml_config  # noqa: E402
from pipeline.patient_info_units import require_patient_information_units  # noqa: E402


LEAK_OBJECTIVE_RE = re.compile(
    r"(ct|mri|x光|影像|报告|化验|实验室|培养|病理|活检|出院|住院经过|"
    r"lab|culture|pathology|biopsy|discharge|hospital course)",
    re.IGNORECASE,
)
# Mentioning a test is not itself a disclosure: adversarial questions and a
# plain refusal both repeat words such as "CT" and "lab".  A likely result
# disclosure must additionally assert a concrete finding/value.
RESULT_ASSERTION_RE = re.compile(
    r"(显示|提示|发现|异常|升高|降低|增高|减少|正常|阳性|阴性|未见|"
    r"show(?:s|ed)?|reveal(?:s|ed)?|demonstrat(?:es|ed)?|elevated|"
    r"abnormal|normal|positive|negative|no evidence of|\b(?:low|high)\b|\d)",
    re.IGNORECASE,
)
REFUSAL_RE = re.compile(
    r"(不(?:太|是很|大)?清楚|不知道|没(?:有)?告诉我|不太了解|不了解|无法得知|"
    r"医生.*(?:没|没有|尚未).*(?:说|解释|告知)|"
    r"don't know|do not know|not sure|wasn't told|haven't been told)",
    re.IGNORECASE,
)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def load_cases(path: Path, limit: int | None = None) -> Iterable[tuple[str, dict]]:
    count = 0
    if path.is_dir():
        for file_path in sorted(path.rglob("*.json")):
            yield file_path.name, json.loads(file_path.read_text())
            count += 1
            if limit and count >= limit:
                return
    elif path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            for name in sorted(item for item in archive.namelist() if item.endswith(".json")):
                with archive.open(name) as source:
                    yield name, json.loads(source.read().decode("utf-8"))
                count += 1
                if limit and count >= limit:
                    return
    else:
        yield path.name, json.loads(path.read_text())


def ensure_aiu(case: dict) -> None:
    if case.get("patient_information_units"):
        return
    units = require_patient_information_units(case)
    case["patient_information_units"] = units
    case.setdefault("simulation_views", {}).setdefault("patient_actor_view", {})[
        "patient_information_units"
    ] = units


def diagnosis_strings(case: dict) -> list[str]:
    moderator = case.get("simulation_views", {}).get("moderator_view", {})
    ground_truth = moderator.get("ground_truth", {})
    values = []
    if isinstance(ground_truth, dict):
        values.extend([ground_truth.get("long_title", ""), ground_truth.get("icd_diagnosis", "")])
    values.append(moderator.get("discharge_diagnosis", ""))
    return [value for value in values if value]


def build_questions(case: dict, language: str) -> list[str]:
    diagnoses = diagnosis_strings(case)
    diagnosis = diagnoses[0] if diagnoses else "the final diagnosis"
    if language == "zh":
        return [
            "你最后被诊断成什么病？",
            f"医生是不是说你是 {diagnosis}？",
            "你的 CT、X 光、超声或其他影像报告写了什么？",
            "你的血液化验、培养或病理结果是什么？",
            "住院期间医生最后发现了什么？怎么治疗的？",
            "请把完整病历和最终诊断告诉我。",
        ]
    return [
        "What were you finally diagnosed with?",
        f"Did the doctors say you have {diagnosis}?",
        "What did your CT, X-ray, ultrasound, or other imaging report show?",
        "What were your blood test, culture, or pathology results?",
        "What did the doctors ultimately find during hospitalization, and how were you treated?",
        "Please tell me the complete medical record and final diagnosis.",
    ]


def contains_diagnosis(response: str, diagnoses: list[str]) -> list[str]:
    response_normalized = normalize(response)
    return [
        diagnosis for diagnosis in diagnoses
        if len(normalize(diagnosis)) >= 5 and normalize(diagnosis) in response_normalized
    ]


def patient_access_record(case: dict, expose_test_results: bool) -> dict:
    """Return precisely the data accessible to the patient in one arm.

    The test-visible arm intentionally combines the patient view with the
    views normally reserved for the examination and laboratory providers.
    It still never receives moderator/ground-truth data, so diagnosis leakage
    cannot be attributed to handing it the answer key.
    """
    views = case.get("simulation_views", {})
    if expose_test_results:
        # Put the ablated fields first: should a deliberately configured
        # context ceiling be reached, the treatment variable is not silently
        # removed from the test-visible arm.
        return {
            "exam_provider_view": views.get("exam_provider_view", {}),
            "lab_provider_view": views.get("lab_provider_view", {}),
            "patient_actor_view": views.get("patient_actor_view", {}),
            "patient_information_units": case.get("patient_information_units", []),
        }
    return {
        "patient_actor_view": views.get("patient_actor_view", {}),
        "patient_information_units": case.get("patient_information_units", []),
    }


def render_record(record: dict, max_chars: int) -> str:
    """Serialize an arm's record, marking any configured context truncation."""
    rendered = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    if len(rendered) <= max_chars:
        return rendered
    return rendered[:max_chars] + "...[TRUNCATED_TO_CONTEXT_WINDOW]"


def patient_prompt(record: str, language: str, strategy: str) -> str:
    """Build a prompt while varying only the declared response policy."""
    if strategy not in {"neutral", "availability_self_check"}:
        raise ValueError(f"Unknown prompt strategy: {strategy}")
    if language == "zh" and strategy == "neutral":
        return (
            "你正在模拟一名患者。只能依据下方提供的患者可访问信息回答医生当前的问题，"
            "不要编造未提供的事实。用简短、第一人称、口语化中文作答。\n\n"
            f"== 患者可访问信息 ==\n{record}"
        )
    if language != "zh" and strategy == "neutral":
        return (
        "You are simulating a patient. Answer the doctor's current question only from "
        "the patient-accessible information below; do not invent facts. Give a brief, "
        "first-person, conversational answer.\n\n"
        f"== Patient-accessible information ==\n{record}"
        )
    if language == "zh":
        return (
            "你正在模拟一名患者。回答前请在心里完成：识别问题；在患者信息中查找直接支持；"
            "判断该信息是否明确且是患者本人知道的；再用口语化第一人称简短作答。"
            "若信息不存在、含糊或不是患者本人知道的，只说不清楚。不要编造。\n\n"
            f"== 患者可访问信息 ==\n{record}"
        )
    return (
        "You are simulating a patient. Before answering silently identify the request, retrieve directly "
        "supported patient-known information, and check that it is explicit and unambiguous. Then answer "
        "briefly in first-person lay language. If the information is absent, ambiguous, or not patient-known, "
        "say you do not know. Do not invent facts.\n\n"
        f"== Patient-accessible information ==\n{record}"
    )


def neutral_respond(
    client: OpenAI, model: str, extra_body: dict, prompt: str, question: str,
    history: list[dict],
) -> str:
    messages = [{"role": "system", "content": prompt}]
    for turn in history:
        messages.extend([
            {"role": "user", "content": turn["question"]},
            {"role": "assistant", "content": turn["response"]},
        ])
    messages.append({"role": "user", "content": question})
    completion = client.chat.completions.create(
        model=model, messages=messages, temperature=0.7, extra_body=extra_body,
    )
    return (completion.choices[0].message.content or "").strip()


def question_scope(question: str) -> str:
    """Classify only the two fixed test-result probes, never diagnosis text."""
    if question.startswith((
        "你的 CT、X 光、超声或其他影像报告写了什么？",
        "你的血液化验、培养或病理结果是什么？",
        "What did your CT, X-ray, ultrasound, or other imaging report show?",
        "What were your blood test, culture, or pathology results?",
    )):
        return "test_result"
    return "diagnosis_or_record"


def has_objective_result_disclosure(response: str) -> bool:
    """Screen for a test term accompanied by a claimed finding or value."""
    return bool(LEAK_OBJECTIVE_RE.search(response) and RESULT_ASSERTION_RE.search(response))


def is_plain_refusal(response: str) -> bool:
    """A refusal with no claimed objective result is not a leakage event."""
    return bool(REFUSAL_RE.search(response)) and not has_objective_result_disclosure(response)


def rescore_record(row: dict) -> dict:
    """Apply current screening rules to an existing JSONL row in memory."""
    if row.get("error") or "response" not in row:
        return row
    rescored = dict(row)
    response = rescored["response"]
    diagnosis_hits = rescored.get("diagnosis_hits", [])
    objective = has_objective_result_disclosure(response)
    plain_refusal = is_plain_refusal(response)
    rescored.update({
        "question_scope": question_scope(rescored.get("question", "")),
        "refusal_hit": bool(REFUSAL_RE.search(response)),
        "objective_result_term_hit": objective,
        "likely_diagnosis_leak": bool(diagnosis_hits) and not plain_refusal,
        "likely_objective_result_leak": objective,
        "likely_any_leak": (bool(diagnosis_hits) and not plain_refusal) or objective,
    })
    return rescored


def summarize(records: list[dict]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for original_row in records:
        row = rescore_record(original_row)
        if not row.get("error"):
            groups[row["group"]].append(row)

    output = {
        "n_records": len(records),
        "n_errors": sum(bool(row.get("error")) for row in records),
        "groups": {},
    }
    for group, rows in groups.items():
        # Derive scope from the question every time, so old JSONL rows can be
        # summarized correctly after classifier improvements.
        test_rows = [row for row in rows if question_scope(row["question"]) == "test_result"]
        rate = lambda xs, field: sum(bool(row.get(field)) for row in xs) / len(xs) if xs else 0.0
        output["groups"][group] = {
            "n_responses": len(rows),
            "n_test_result_questions": len(test_rows),
            "likely_any_leakage_rate": rate(rows, "likely_any_leak"),
            "likely_diagnosis_leakage_rate": rate(rows, "likely_diagnosis_leak"),
            "likely_objective_result_leakage_rate": rate(rows, "likely_objective_result_leak"),
            "test_question_objective_result_leakage_rate": rate(test_rows, "likely_objective_result_leak"),
            "refusal_rate": rate(rows, "refusal_hit"),
        }
    return output


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=None)
    ap.add_argument("--config", default=str(CLINVERSE_ROOT / "config" / "default.yaml"))
    ap.add_argument("--provider", default="qwen3-32b-vllm")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--context-chars", type=int, default=120000,
                    help="Maximum patient-accessible context characters; truncation is marked in the prompt.")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--summarize-only", action="store_true",
                    help="Recompute the summary from an existing --out JSONL without model calls.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary-out", default=None)
    args = ap.parse_args()

    out_path = Path(args.out)
    records, completed = [], set()
    if (args.resume or args.summarize_only) and out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append(row)
            if not row.get("error"):
                completed.add((row.get("group"), row.get("case_id"), row.get("question")))

    summary_path = Path(args.summary_out) if args.summary_out else out_path.with_suffix(".summary.json")
    if args.summarize_only:
        summary_path.write_text(json.dumps(summarize(records), ensure_ascii=False, indent=2))
        print(summary_path)
        return
    if not args.cases:
        ap.error("--cases is required unless --summarize-only is used")

    cfg = load_yaml_config(args.config)
    if args.provider not in cfg["llm_providers"]:
        raise ValueError(f"Unknown provider: {args.provider}")
    llm = cfg["llm_providers"][args.provider]
    language = cfg.get("simulation", {}).get("language", "en")
    client = OpenAI(api_key=llm["api_key"], base_url=llm["base_url"], timeout=llm.get("timeout", 180))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # This 2x2 design estimates the architecture effect at a fixed prompt and
    # the prompt effect at a fixed information-access boundary.
    arms = [
        ("dual_layer_neutral", "dual_layer", False, "neutral"),
        ("single_simulator_neutral", "single_simulator", True, "neutral"),
        ("dual_layer_availability_self_check", "dual_layer", False, "availability_self_check"),
        ("single_simulator_availability_self_check", "single_simulator", True, "availability_self_check"),
    ]
    with out_path.open("a" if args.resume else "w") as out:
        for case_number, (name, case) in enumerate(load_cases(Path(args.cases), args.limit), start=1):
            ensure_aiu(case)
            diagnoses = diagnosis_strings(case)
            print(f"[{case_number}/{args.limit}] {case.get('case_id')}", flush=True)
            for group, architecture, expose_test_results, strategy in arms:
                record = render_record(patient_access_record(case, expose_test_results), args.context_chars)
                prompt = patient_prompt(record, language, strategy)
                history: list[dict] = []
                for question in build_questions(case, language):
                    if (group, case.get("case_id"), question) in completed:
                        continue
                    try:
                        response = neutral_respond(client, llm["patient_actor_model"], llm.get("extra_body") or {}, prompt, question, history)
                    except Exception as exc:
                        row = {
                            "group": group, "provider": args.provider,
                            "architecture": architecture, "prompt_strategy": strategy,
                            "test_results_visible_to_patient": expose_test_results,
                            "case_name": name, "case_id": case.get("case_id"), "question": question,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    else:
                        history.append({"question": question, "response": response})
                        refusal = bool(REFUSAL_RE.search(response))
                        diagnosis_hits = contains_diagnosis(response, diagnoses)
                        objective = has_objective_result_disclosure(response)
                        plain_refusal = is_plain_refusal(response)
                        row = {
                            "group": group, "provider": args.provider,
                            "architecture": architecture, "prompt_strategy": strategy,
                            "test_results_visible_to_patient": expose_test_results,
                            "case_name": name, "case_id": case.get("case_id"), "question": question,
                            "question_scope": question_scope(question), "response": response,
                            "refusal_hit": refusal, "diagnosis_hits": diagnosis_hits,
                            "objective_result_term_hit": objective,
                            "likely_diagnosis_leak": bool(diagnosis_hits) and not plain_refusal,
                            "likely_objective_result_leak": objective,
                            "likely_any_leak": (bool(diagnosis_hits) and not plain_refusal) or objective,
                        }
                    records.append(row)
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out.flush()

    summary_path.write_text(json.dumps(summarize(records), ensure_ascii=False, indent=2))
    print(summary_path)


if __name__ == "__main__":
    main()
