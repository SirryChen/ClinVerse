from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


def _extract_task_buckets(record: dict) -> list[str]:
    """Map a trajectory to transferable behavior stages.

    Disease labels are intentionally not used as SkillRL buckets: they are
    unavailable at inference time and make the bank brittle across hospitals.
    Tool usage and outcome tags are observable in every doctor benchmark.
    """
    buckets: list[str] = []
    tool_counts = record.get("tool_counts", {}) or {}
    if tool_counts.get("ask_patient", 0) or "history_sparse" in record.get("tags", []):
        buckets.append("history_collection")
    if tool_counts.get("request_physical_exam", 0) or tool_counts.get("order_tests", 0) or tool_counts.get("order_imaging", 0):
        buckets.append("targeted_investigation")
    if record.get("prediction") or record.get("scores", {}).get("diagnosis_score") is not None:
        buckets.append("diagnosis_synthesis")
    return buckets or [record.get("task_type", "clinical_diagnosis")]


class SkillRLTemplateMemory:
    """Self-contained template retrieval for ClinVerse SkillRL banks.

    This mirrors SkillRL's no-embedding template mode, while keeping runtime
    evaluation independent of the separately cloned SkillRL repository.
    """

    def __init__(self, skills_json_path: str | Path, *, task_specific_top_k: int | None = None):
        with open(skills_json_path, encoding="utf-8") as f:
            self.skills = json.load(f)
        self.task_specific_top_k = task_specific_top_k

    def _detect_task_type(self, task_description: str) -> str:
        task_specific = self.skills.get("task_specific_skills", {})
        text = task_description.lower()
        stage_keywords = {
            "history_collection": ("history", "symptom", "病史", "症状", "问诊"),
            "targeted_investigation": ("test", "exam", "lab", "imaging", "检查", "检验", "查体"),
            "diagnosis_synthesis": ("diagnosis", "differential", "finalize", "诊断", "鉴别", "结论"),
        }
        for stage, keywords in stage_keywords.items():
            if stage in task_specific and any(keyword in text for keyword in keywords):
                return stage
        return next(iter(task_specific), "unknown")

    def retrieve(self, task_description: str, *, top_k: int = 6) -> dict:
        task_type = self._detect_task_type(task_description)
        general = self.skills.get("general_skills", [])[:max(0, top_k)]
        task_skills = self.skills.get("task_specific_skills", {}).get(task_type, [])
        if self.task_specific_top_k is not None:
            task_skills = task_skills[:max(0, self.task_specific_top_k)]
        return {
            "task_type": task_type,
            "retrieval_mode": "template",
            "general_skills": general,
            "task_specific_skills": task_skills,
            "mistakes_to_avoid": self.skills.get("common_mistakes", [])[:5],
        }

    @staticmethod
    def format_for_prompt(retrieved: dict) -> str:
        sections = [
            "== Retrieved SkillRL Guidance ==",
            "Use these as transferable process guidance, not as facts about the current patient.",
        ]
        general = retrieved.get("general_skills", [])
        if general:
            sections.append("General principles:\n" + "\n".join(
                f"- {skill.get('title', '')}: {skill.get('principle', '')}" for skill in general
            ))
        task_skills = retrieved.get("task_specific_skills", [])
        if task_skills:
            sections.append("Stage-specific skills:\n" + "\n".join(
                f"- {skill.get('title', '')}: {skill.get('principle', '')} "
                f"(Apply when: {skill.get('when_to_apply', '')})"
                for skill in task_skills
            ))
        mistakes = retrieved.get("mistakes_to_avoid", [])
        if mistakes:
            sections.append("Mistakes to avoid:\n" + "\n".join(
                f"- {mistake.get('description', '')} Avoid by: {mistake.get('how_to_avoid', '')}"
                for mistake in mistakes
            ))
        return "\n\n".join(sections)


def build_skillbank(records: list[dict]) -> dict:
    general_skills = []
    task_specific = defaultdict(list)
    common_mistakes = []

    tag_counts = Counter()
    for record in records:
        tag_counts.update(record.get("tags", []))

    general_skills.append(
        {
            "skill_id": "gen_001",
            "title": "History Before Tests",
            "principle": "Collect chief complaint details plus PMH, medications, allergies, and social history before broadening the workup unless the patient is unstable.",
            "when_to_apply": "At the start of every non-crashing case.",
        }
    )
    general_skills.append(
        {
            "skill_id": "gen_002",
            "title": "Targeted Objective Workup",
            "principle": "Order tests and exams that separate the current leading diagnoses rather than vague blanket workups.",
            "when_to_apply": "After the first focused differential has formed.",
        }
    )
    general_skills.append(
        {
            "skill_id": "gen_003",
            "title": "Evidence Synthesis Before Finalize",
            "principle": "Before finalizing, verify that the top diagnosis is supported by the highest-value subjective and objective findings and map it to the closest ICD-10 code.",
            "when_to_apply": "On the final 1-2 turns of the episode.",
        }
    )

    if tag_counts.get("abnormal_tests_missed", 0):
        common_mistakes.append(
            {
                "mistake_id": "err_001",
                "description": "Missing abnormal objective evidence that was available in the environment.",
                "why_it_happens": "The agent orders low-yield tests or fails to revisit unexpected abnormal results.",
                "how_to_avoid": "Order discriminator tests first, then branch follow-up questions and confirmatory workup from the abnormal findings.",
            }
        )
    if tag_counts.get("history_sparse", 0):
        common_mistakes.append(
            {
                "mistake_id": "err_002",
                "description": "Finalizing after weak history collection.",
                "why_it_happens": "The agent jumps to tests or diagnosis before building clinical context.",
                "how_to_avoid": "Use an early-turn checklist that covers HPI, PMH, medications, allergies, and social history.",
            }
        )
    if tag_counts.get("max_turns_reached", 0):
        common_mistakes.append(
            {
                "mistake_id": "err_003",
                "description": "Running out of turns before convergence.",
                "why_it_happens": "The agent keeps exploring after one diagnosis already dominates.",
                "how_to_avoid": "Reserve the last turns for synthesis and commitment instead of new broad exploration.",
            }
        )

    bucket_counts = Counter()
    for record in records:
        bucket_counts.update(_extract_task_buckets(record))

    for idx, bucket in enumerate(sorted(bucket_counts), start=1):
        principles = {
            "history_collection": (
                "Use a structured HPI sequence, then explicitly cover past history, medications, "
                "allergies, and social history before ordering non-emergency tests."
            ),
            "targeted_investigation": (
                "Choose focused exams and tests that distinguish the leading differential; inspect "
                "each result before adding another broad workup."
            ),
            "diagnosis_synthesis": (
                "Synthesize subjective and objective evidence, revisit abnormal findings, and only "
                "then commit to one diagnosis and the closest supported code."
            ),
        }
        task_specific[bucket].append(
            {
                "skill_id": f"task_{idx:03d}",
                "title": bucket.replace("_", " ").title(),
                "principle": principles.get(
                    bucket,
                    "Apply the reusable strategy represented by this trajectory stage, without copying patient-specific facts.",
                ),
                "when_to_apply": f"When the episode enters the {bucket.replace('_', ' ')} stage.",
            }
        )

    return {
        "schema_version": "skillrl-v1",
        "domain": "clinical_diagnosis",
        "general_skills": general_skills,
        "task_specific_skills": dict(task_specific),
        "common_mistakes": common_mistakes,
    }
