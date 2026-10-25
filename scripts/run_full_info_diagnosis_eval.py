#!/usr/bin/env python3
"""Direct diagnosis baselines without outcome-label leakage."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import sys
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import load_yaml_config
from core import icd as icd_mod
from core.registry import get_adapter
from env.task_stream import TaskStream
from evaluation.metrics import diagnosis_category_accuracy, diagnosis_subtype_accuracy


DIAGNOSIS_TASKS = ("primary-diagnosis", "subtype-given-category")


def system_prompt(evidence_mode: str, diagnosis_task: str = "primary-diagnosis") -> str:
    evidence_description = (
        "complete clinical evidence from a finished workup"
        if evidence_mode == "full-workup"
        else "only the patient's initial information: demographics, history, physical examination, "
        "vital signs, and recorded admission medications; no laboratory, microbiology, or imaging results"
    )
    if diagnosis_task == "subtype-given-category":
        return f"""You are performing constrained ICD-10-CM classification. You receive {evidence_description},
plus the known ICD-10-CM block of the patient's primary diagnosis.

The supplied block is authoritative and is a hard constraint. Your task is not to reconsider or
predict the block. You must select exactly one three-character ICD-10-CM category contained within
the supplied block. Even if the clinical evidence appears to favor a diagnosis outside that block,
choose the most plausible category inside the supplied block. Before answering, silently verify
that your three-character category falls within the stated block range. Never output the block
range itself or any category outside it.

Return JSON only, with no explanation or additional codes:
{{"icd10_code": "<one three-character ICD-10-CM category within the supplied block>"}}."""
    return f"""You are a clinical diagnostician. You receive {evidence_description},
but no final diagnosis, discharge diagnosis, billing codes, or outcome labels.
Infer the single most likely primary diagnosis.

Return JSON only: {{"icd10_code": "<your best ICD-10-CM code>"}}."""


def load_config(path: str) -> dict:
    return load_yaml_config(path)


def resolve_config_path(config_path: str, value: str) -> Path:
    root = Path(config_path).resolve().parent.parent
    path = Path(value)
    return path if path.is_absolute() else root / path


def as_text(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def render_labs(labs: list[dict], adapter) -> str:
    lines = []
    for lab in labs or []:
        item = adapter.lab_catalog.resolve_itemid(lab.get("itemid", ""))
        name = (item or {}).get("display_name") or (item or {}).get("canonical_name")
        name = name or f"itemid {lab.get('itemid', '')}"
        value = as_text(lab.get("valuestr"))
        if not value:
            continue
        low, high, flag = (as_text(lab.get(key)) for key in ("ref_range_lower", "ref_range_upper", "flag"))
        details = [value]
        if low or high:
            details.append(f"reference range: {low or '?'}–{high or '?'}")
        if flag:
            details.append(f"flag: {flag}")
        lines.append(f"- {name}: {'; '.join(details)}")
    return "\n".join(lines) or "- No laboratory results recorded."


def render_microbiology(rows: list[dict]) -> str:
    lines = []
    for row in rows or []:
        test = as_text(row.get("test_name")) or as_text(row.get("org_name")) or "Microbiology result"
        result = as_text(row.get("valuestr")) or as_text(row.get("interpretation"))
        specimen = as_text(row.get("spec_type_desc"))
        if result:
            lines.append(f"- {test}{f' ({specimen})' if specimen else ''}: {result}")
    return "\n".join(lines) or "- No microbiology results recorded."


def render_radiology(rows: list[dict]) -> str:
    lines = []
    for row in rows or []:
        title = " / ".join(part for part in (as_text(row.get("modality")), as_text(row.get("exam_name"))) if part)
        text = as_text(row.get("text"))
        if text:
            lines.append(f"- {title or 'Imaging study'}:\n{text}")
    return "\n".join(lines) or "- No imaging results recorded."


def build_prompt(
    case: dict,
    adapter,
    evidence_mode: str = "full-workup",
    diagnosis_category: str | None = None,
) -> str:
    """Render allowlisted core evidence only; never read moderator_view."""
    core = case.get("core_admission", {})
    demographics = core.get("demographics", {})
    # This explicit allowlist excludes discharge_location and hospital_expire_flag.
    demographics_text = "\n".join([
        f"- Age: {as_text(demographics.get('anchor_age')) or 'unknown'}",
        f"- Sex: {as_text(demographics.get('gender')) or 'unknown'}",
        f"- Race: {as_text(demographics.get('race')) or 'not recorded'}",
        f"- Admission type: {as_text(demographics.get('admission_type')) or 'not recorded'}",
        f"- Admission location: {as_text(demographics.get('admission_location')) or 'not recorded'}",
    ])
    medications = []
    for medication in core.get("medications", []) or []:
        name = as_text(medication.get("drug"))
        if name:
            suffix = " ".join(part for part in (as_text(medication.get("prod_strength")), as_text(medication.get("route"))) if part)
            medications.append(f"- {name}{f' ({suffix})' if suffix else ''}")
    sections = [
        "## Demographics\n" + demographics_text,
        "## History and presentation\n" + (as_text(core.get("hpi")) or "Not recorded."),
        "## Vital signs and physical examination\n" + (as_text(core.get("physical_exam")) or as_text(core.get("vital_signs")) or "Not recorded."),
        "## Medications\n" + ("\n".join(medications) or "- No medications recorded."),
    ]
    if evidence_mode == "full-workup":
        sections.extend([
            "## Laboratory results\n" + render_labs(core.get("labs", []), adapter),
            "## Microbiology results\n" + render_microbiology(core.get("microbiology", [])),
            "## Imaging results\n" + render_radiology(core.get("radiology", [])),
        ])
    if diagnosis_category:
        sections.append(
            "## Authoritative primary-diagnosis block (hard constraint)\n"
            f"- Allowed ICD-10-CM block: {diagnosis_category}\n"
            "- Output exactly one three-character ICD-10-CM category inside this block.\n"
            "- Do not output a category outside this range, even if other clinical evidence suggests one.\n"
            "- Silently check that the answer is inside the allowed block before returning the JSON."
        )
    return "\n\n".join(sections)


def parse_icd10(content: str) -> str:
    content = (content or "").strip()
    try:
        data = json.loads(content.removeprefix("```json").removeprefix("```").removesuffix("```").strip())
        if isinstance(data, dict):
            return as_text(data.get("icd10_code")).upper()
    except json.JSONDecodeError:
        pass
    match = re.search(r"\b([A-TV-Z][0-9][0-9A-Z](?:\.[0-9A-Z]{1,4})?)\b", content.upper())
    return match.group(1) if match else ""


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--cases-dir", required=True)
    parser.add_argument("--source", default="mimic-iv-ext")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--provider", required=True, help="llm_providers key; doctor_model is used.")
    parser.add_argument(
        "--evidence-mode", choices=("full-workup", "basic-info"), default="full-workup",
        help="full-workup includes test results; basic-info includes only initial patient information.",
    )
    parser.add_argument(
        "--diagnosis-task",
        choices=DIAGNOSIS_TASKS,
        default="primary-diagnosis",
        help="subtype-given-category reveals the ground-truth ICD block and evaluates the predicted three-character subtype.",
    )
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--sample-seed", type=int, default=None)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    provider = cfg["llm_providers"][args.provider]
    if "doctor_model" not in provider or "base_url" not in provider:
        raise ValueError(f"Provider {args.provider!r} must define doctor_model and base_url.")
    output_dir = Path(args.output_root) if args.output_root else resolve_config_path(args.config, cfg["paths"]["eval_results"]) / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path, errors_path = output_dir / "predictions.jsonl", output_dir / "errors.jsonl"
    predictions = load_jsonl(predictions_path) if args.resume else []
    errors = load_jsonl(errors_path) if args.resume else []
    completed_ids = {row["case_id"] for row in predictions}
    failed_ids = {row["case_id"] for row in errors}
    refs = list(TaskStream(args.cases_dir, source=args.source))
    if args.shuffle:
        random.Random(args.sample_seed).shuffle(refs)
    adapter = get_adapter(args.source, **cfg.get("datasets", {}).get(args.source, {}))
    client = OpenAI(api_key=provider["api_key"], base_url=provider["base_url"], timeout=provider.get("timeout", 180))
    pending = [ref for ref in refs if ref.case_id not in completed_ids and (args.retry_failed or ref.case_id not in failed_ids)]

    with predictions_path.open("a" if args.resume else "w", encoding="utf-8") as prediction_file, errors_path.open("a" if args.resume else "w", encoding="utf-8") as error_file:
        with tqdm(total=max(args.n - len(predictions), 0), desc=args.diagnosis_task, unit="case") as progress:
            for ref in pending:
                if len(predictions) >= args.n:
                    break
                try:
                    case = json.loads(ref.path.read_text(encoding="utf-8"))
                    ground_truth = case["simulation_views"]["moderator_view"]["ground_truth"].get("icd10_code", "")
                    category_hint = (
                        icd_mod.block(ground_truth)
                        if args.diagnosis_task == "subtype-given-category"
                        else None
                    )
                    if args.diagnosis_task == "subtype-given-category" and not category_hint:
                        raise ValueError(f"Cannot derive an ICD-10 block from ground truth {ground_truth!r}.")
                    response = client.chat.completions.create(
                        model=provider["doctor_model"],
                        messages=[
                            {"role": "system", "content": system_prompt(args.evidence_mode, args.diagnosis_task)},
                            {
                                "role": "user",
                                "content": build_prompt(
                                    case,
                                    adapter,
                                    args.evidence_mode,
                                    diagnosis_category=category_hint,
                                ),
                            },
                        ],
                        temperature=0,
                        max_tokens=128,
                        extra_body=provider.get("extra_body"),
                    )
                    predicted = parse_icd10(response.choices[0].message.content)
                    category = diagnosis_category_accuracy(predicted, ground_truth)
                    subtype = diagnosis_subtype_accuracy(predicted, ground_truth) if category else 0
                    row = {
                        "case_id": ref.case_id,
                        "diagnosis_task": args.diagnosis_task,
                        "diagnosis_category_hint": category_hint,
                        "predicted_icd10": predicted,
                        "ground_truth_icd10": ground_truth,
                        "diagnosis_category_accuracy": category,
                        "diagnosis_subtype_accuracy": subtype,
                    }
                    prediction_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                    prediction_file.flush()
                    predictions.append(row)
                    completed_ids.add(ref.case_id)
                    progress.update(1)
                except Exception as exc:
                    row = {"case_id": ref.case_id, "error": str(exc)}
                    errors.append(row)
                    failed_ids.add(ref.case_id)
                    error_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                    error_file.flush()

    def metric(key: str) -> dict:
        values = [row[key] for row in predictions]
        return {"mean": round(statistics.mean(values), 4) if values else 0.0, "std": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0}

    included_evidence = (
        "demographics, HPI, physical examination, admission medications, laboratory, microbiology, and imaging results"
        if args.evidence_mode == "full-workup"
        else "demographics, HPI, physical examination/vital signs, and admission medications only; all laboratory, microbiology, and imaging results are excluded"
    )
    label_policy = (
        "The prompt additionally reveals only the ground-truth-derived ICD-10 block as the "
        "known primary-diagnosis category; it does not reveal the three-character category or full code."
        if args.diagnosis_task == "subtype-given-category"
        else "The prompt does not reveal any ground-truth diagnosis label."
    )
    summary = {
        "n": len(predictions),
        "provider": args.provider,
        "source": args.source,
        "evidence_mode": args.evidence_mode,
        "diagnosis_task": args.diagnosis_task,
        "primary_metric": "diagnosis_subtype_accuracy" if args.diagnosis_task == "subtype-given-category" else None,
        "metrics": {
            key: metric(key)
            for key in ("diagnosis_category_accuracy", "diagnosis_subtype_accuracy")
        },
        "n_errors": len(errors),
        "leakage_policy": (
            f"Prompt allowlists core_admission {included_evidence}. {label_policy} "
            "It excludes discharge diagnosis, hospital course, billing ICDs, outcome fields, "
            "discharge location, and discharge medications."
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
