#!/usr/bin/env python3
"""Distill doctor rollouts into a SkillRL-compatible skill bank with an LLM."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI

from core.config import load_yaml_config
from doctor_train.memrl import reward_from_scores
from doctor_train.skillrl import _extract_task_buckets, build_skillbank


def load_jsonl(path: str | Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_config(path: str | Path) -> dict:
    return load_yaml_config(path)


def parse_json_array(text: str) -> list[dict]:
    """Extract the JSON array when a model includes a Markdown fence."""
    start = text.find("[")
    end = text.rfind("]") + 1
    if start < 0 or end <= start:
        raise ValueError("Distiller response did not contain a JSON array.")
    result = json.loads(text[start:end])
    if not isinstance(result, list) or not all(isinstance(item, dict) for item in result):
        raise ValueError("Distiller response JSON must be an array of objects.")
    return result


def compact_trajectory(record: dict) -> dict[str, Any]:
    """Keep strategy-relevant behavior while excluding case identity and labels."""
    turns = []
    for turn in record.get("transcript", {}).get("turns", [])[:12]:
        function = turn.get("function", "")
        args = turn.get("args", {}) or {}
        safe_args = {
            key: value
            for key, value in args.items()
            if key in {"question", "tests", "imaging_type", "exam_type"}
        }
        turns.append({"action": function, "arguments": safe_args})
    scores = record.get("scores", {})
    return {
        "outcome": "success" if reward_from_scores(scores) >= 0.75 else "failure",
        "scores": {
            key: round(float(scores.get(key, 0.0)), 3)
            for key in (
                "information_collection_score",
                "test_order_coverage",
                "abnormal_test_coverage",
                "diagnosis_score",
            )
        },
        "tags": record.get("tags", []),
        "actions": turns,
    }


def select_examples(records: list[dict], limit: int) -> tuple[list[dict], list[dict]]:
    ordered = sorted(records, key=lambda record: reward_from_scores(record.get("scores", {})), reverse=True)
    successes = [record for record in ordered if reward_from_scores(record.get("scores", {})) >= 0.75]
    failures = [record for record in reversed(ordered) if reward_from_scores(record.get("scores", {})) < 0.75]
    return [compact_trajectory(record) for record in successes[:limit]], [compact_trajectory(record) for record in failures[:limit]]


def request_skills(client: OpenAI, *, model: str, extra_body: dict, prompt: str, max_tokens: int) -> list[dict]:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=max_tokens,
        extra_body=extra_body or None,
    )
    return parse_json_array(response.choices[0].message.content or "")


def normalize_skills(skills: list[dict], prefix: str) -> list[dict]:
    normalized = []
    for index, skill in enumerate(skills, start=1):
        principle = str(skill.get("principle", "")).strip()
        title = str(skill.get("title", "")).strip()
        when = str(skill.get("when_to_apply", "")).strip()
        if not title or not principle or not when:
            continue
        normalized.append(
            {
                "skill_id": f"{prefix}_{index:03d}",
                "title": title,
                "principle": principle,
                "when_to_apply": when,
            }
        )
    if not normalized:
        raise ValueError(f"Distiller returned no valid {prefix} skills.")
    return normalized


def normalize_mistakes(mistakes: list[dict]) -> list[dict]:
    normalized = []
    for index, mistake in enumerate(mistakes, start=1):
        description = str(mistake.get("description", "")).strip()
        why = str(mistake.get("why_it_happens", "")).strip()
        avoid = str(mistake.get("how_to_avoid", "")).strip()
        if description and why and avoid:
            normalized.append(
                {
                    "mistake_id": f"err_{index:03d}",
                    "description": description,
                    "why_it_happens": why,
                    "how_to_avoid": avoid,
                }
            )
    if not normalized:
        raise ValueError("Distiller returned no valid common mistakes.")
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill doctor rollouts into a SkillRL skill bank using an LLM.")
    parser.add_argument("--input-jsonl", required=True, help="Path to rollout records.")
    parser.add_argument("--output", required=True, help="Output SkillRL-compatible JSON path.")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--distiller-provider", default="qwen3-32b-vllm", help="llm_providers key used for skill distillation.")
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N rollout records.")
    parser.add_argument("--examples-per-outcome", type=int, default=8, help="Successful and failed trajectories supplied per distillation call.")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--offline-template",
        action="store_true",
        help="Build the deterministic SkillRL-compatible template bank without an LLM distiller.",
    )
    args = parser.parse_args()

    if args.examples_per_outcome < 1:
        raise ValueError("--examples-per-outcome must be positive.")
    records = load_jsonl(args.input_jsonl)
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise ValueError("No rollout records were supplied.")

    if args.offline_template:
        skillbank = build_skillbank(records)
        skillbank["distillation"] = {
            "method": "deterministic_template",
            "input_records": len(records),
        }
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(skillbank, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps({
            "output": str(out_path),
            "method": "deterministic_template",
            "input_records": len(records),
            "general_skills": len(skillbank["general_skills"]),
            "task_skills": sum(
                len(skills)
                for skills in skillbank["task_specific_skills"].values()
            ),
            "common_mistakes": len(skillbank["common_mistakes"]),
        }, ensure_ascii=False, indent=2))
        return

    cfg = load_config(args.config)
    provider = cfg["llm_providers"][args.distiller_provider]
    model = provider.get("doctor_model")
    if not model or not provider.get("base_url"):
        raise ValueError(f"Distiller provider {args.distiller_provider!r} must define doctor_model and base_url.")
    client = OpenAI(
        api_key=provider.get("api_key", "EMPTY"),
        base_url=provider["base_url"],
        timeout=provider.get("timeout", 180),
    )
    extra_body = provider.get("extra_body", {})

    general_successes, general_failures = select_examples(records, args.examples_per_outcome)
    general_prompt = """You are distilling reusable clinical diagnostic-agent skills from rollout summaries.
Generate 8-12 GENERAL SKILLS that improve history taking, targeted investigation, evidence synthesis, and turn budgeting.
Use only transferable action strategies. Do not state, infer, or copy any patient-specific facts, diagnoses, ICD codes, or exact test results.
Each skill must be concise, actionable, and based on patterns that distinguish success from failure.
Return ONLY a JSON array of objects with title, principle, and when_to_apply.

SUCCESSFUL TRAJECTORIES:
{successes}

FAILED TRAJECTORIES:
{failures}
""".format(successes=json.dumps(general_successes, ensure_ascii=False), failures=json.dumps(general_failures, ensure_ascii=False))
    general_skills = normalize_skills(
        request_skills(client, model=model, extra_body=extra_body, prompt=general_prompt, max_tokens=args.max_tokens),
        "gen",
    )

    buckets: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        for bucket in _extract_task_buckets(record):
            buckets[bucket].append(record)
    task_specific_skills = {}
    for bucket, bucket_records in sorted(buckets.items()):
        successes, failures = select_examples(bucket_records, args.examples_per_outcome)
        prompt = """You are distilling reusable clinical diagnostic-agent skills for one episode stage.
Stage: {bucket}
Generate 4-6 stage-specific skills. Use only transferable decision rules; never copy diagnoses, ICD codes, patient facts, or exact test results.
Return ONLY a JSON array of objects with title, principle, and when_to_apply.

SUCCESSFUL TRAJECTORIES:
{successes}

FAILED TRAJECTORIES:
{failures}
""".format(
            bucket=bucket,
            successes=json.dumps(successes, ensure_ascii=False),
            failures=json.dumps(failures, ensure_ascii=False),
        )
        task_specific_skills[bucket] = normalize_skills(
            request_skills(client, model=model, extra_body=extra_body, prompt=prompt, max_tokens=args.max_tokens),
            f"{bucket[:4]}",
        )

    mistakes_prompt = """You are analyzing failed clinical diagnostic-agent trajectories.
Generate 6-10 common, avoidable mistakes. Focus on reusable process failures, not patient-specific clinical conclusions.
Do not state or infer diagnoses, ICD codes, patient facts, or exact test results.
Return ONLY a JSON array of objects with description, why_it_happens, and how_to_avoid.

FAILED TRAJECTORIES:
{failures}
""".format(failures=json.dumps(general_failures, ensure_ascii=False))
    common_mistakes = normalize_mistakes(
        request_skills(client, model=model, extra_body=extra_body, prompt=mistakes_prompt, max_tokens=args.max_tokens)
    )

    skillbank = {
        "schema_version": "skillrl-v1",
        "domain": "clinical_diagnosis",
        "distillation": {
            "method": "llm_trajectory_distillation",
            "provider": args.distiller_provider,
            "model": model,
            "input_records": len(records),
            "examples_per_outcome": args.examples_per_outcome,
        },
        "general_skills": general_skills,
        "task_specific_skills": task_specific_skills,
        "common_mistakes": common_mistakes,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(skillbank, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(out_path),
        "distiller_provider": args.distiller_provider,
        "distiller_model": model,
        "input_records": len(records),
        "general_skills": len(general_skills),
        "task_skills": sum(len(skills) for skills in task_specific_skills.values()),
        "common_mistakes": len(common_mistakes),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
