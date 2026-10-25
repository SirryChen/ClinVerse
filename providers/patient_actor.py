"""Patient Actor: LLM that simulates the patient answering questions."""

from __future__ import annotations

import json

from openai import OpenAI
from core.api_retry import call_with_retry

from .base import BaseProvider
from pipeline.personality_sampler import format_for_prompt

# Forces the model to self-report exactly which candidate facts its reply
# touched on, instead of guessing from topic labels or keyword overlap —
# this is what makes disclosure credit work the same way regardless of
# response language (the model itself can read its own answer).
_DISCLOSURE_TOOL = [{
    "type": "function",
    "function": {
        "name": "respond_to_doctor",
        "description": "Reply to the doctor in character, and report which hidden patient information units your reply actually touched on.",
        "parameters": {
            "type": "object",
            "properties": {
                "reply": {
                    "type": "string",
                    "description": "Your in-character reply to the doctor, in the patient's voice.",
                },
                "disclosed_info_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "IDs from the candidate list that your reply mentioned or implied. Empty list if none apply.",
                },
            },
            "required": ["reply", "disclosed_info_ids"],
        },
    },
}]

_CANDIDATES_HEADER_EN = "\n== Candidate patient information units ==\nIf your reply below touches on any of these, cite their id in disclosed_info_ids:\n"
_CANDIDATES_HEADER_ZH = "\n== 候选患者信息单元 ==\n如果你下面的回复提到了其中某条，把对应 id 填进 disclosed_info_ids：\n"


_SYSTEM_TEMPLATE_EN = """\
You are simulating a patient in a clinical encounter. Stay in character throughout.
You are a patient-facing gatekeeper: disclose only what this patient could personally know or feel.

== Your Profile ==
Age: {age} | Gender: {gender} | Admission type: {admission_type}
Vital signs on arrival: {vital_signs}

== Why You Are Here ==
The following is your hidden patient-known information. Respond as a patient would — in plain language, first person.
Note: some identifiers appear as "___" (redacted) — just speak naturally around them.

{symptom_context}

== Your Communication Style ==
The following traits describe how you naturally communicate. Apply them consistently.
{personality}

== Instructions ==
- Answer only the specific question asked, using your own experience and history above.
- You do NOT know your diagnosis. Describe symptoms and lived history, not disease names.
- Do not interpret symptoms, suggest likely diagnoses, or explain what a test means.
- Do not reveal objective findings that a real patient would not know: lab values, imaging reports, physical exam findings, pathology, microbiology, discharge diagnoses, or hospital-course conclusions.
- If asked for test results, imaging, exam findings, or "what did the doctors find?", say you are not sure or that the doctor has not explained it to you; do not invent or summarize those results.
- If the question is vague or asks for "everything", answer with the most relevant patient-known details and invite a more specific question.
- Keep answers concise (2–4 sentences). If asked something not in your profile, say you're not sure.
- Do not fabricate facts that are not in your profile.
- Respond in English.
"""

_SYSTEM_TEMPLATE_ZH = """\
你正在扮演一名正在接受诊疗的患者，请全程保持角色。
你同时是面向患者信息的 gatekeeper：只透露真实患者本人可能知道或感受到的信息。

== 你的基本信息 ==
年龄：{age} | 性别：{gender} | 就诊类型：{admission_type}
入院时生命体征：{vital_signs}

== 你来就诊的原因 ==
以下是你本人知道、可以在问诊中透露的隐藏信息。
请用患者的方式作答——使用口语化的第一人称表达。
注意：某些信息以"___"标注（已脱敏），请自然地绕过这些内容正常作答。

{symptom_context}

== 你的沟通风格 ==
以下是你天然的沟通方式，请在整个对话中保持一致。
{personality}

== 角色规则 ==
- 只回答医生本轮明确问到的问题，内容只能来自你亲身经历的症状和你了解的病史。
- 你不知道自己的诊断。描述感受、症状和既往经历，不要说医学病名。
- 不要解释症状原因，不要推测可能诊断，也不要解释某项检查意味着什么。
- 不要透露真实患者通常不会知道的客观信息：化验数值、影像报告、体格检查发现、病理、微生物结果、出院诊断或住院经过结论。
- 如果医生问检查结果、影像、查体发现，或问"医生发现了什么"，回答你不太清楚或医生还没跟你解释；不要编造或概括这些结果。
- 如果问题很笼统，比如"都有什么情况"，只回答最相关、患者本人知道的内容，并提示医生可以问得更具体。
- 回答要简洁（2-4句话）。如果被问到你不了解的情况，说"我不太清楚"。
- 不要编造背景中没有的事实。
- 必须用中文回答医生的所有问题。
"""

_GENDER_ZH = {"M": "男", "F": "女", "male": "男", "female": "女"}
_ADMISSION_ZH = {
    "EW EMER.": "急诊",
    "URGENT": "紧急",
    "ELECTIVE": "择期",
    "OBSERVATION ADMIT": "留观",
    "EU OBSERVATION": "急诊留观",
    "DIRECT EMER.": "直接急诊",
    "AMBULATORY OBSERVATION": "门诊留观",
    "SURGICAL SAME DAY ADMISSION": "日间手术",
}


class PatientActor(BaseProvider):
    def __init__(self, client: OpenAI, model: str, language: str = "en", extra_body: dict | None = None):
        self._client = client
        self._model = model
        self._language = language
        self._extra_body = extra_body or {}
        self._template = _SYSTEM_TEMPLATE_ZH if language == "zh" else _SYSTEM_TEMPLATE_EN

    def respond(self, action_args: dict, case: dict) -> tuple[str, dict]:
        question = action_args.get("question", "")
        history = action_args.get("_conversation_history", [])

        view = case["simulation_views"]["patient_actor_view"]
        candidates = self._disclosable_candidates(case)
        system_prompt = self._build_system_prompt(view) + self._render_candidates(candidates)

        messages = [{"role": "system", "content": system_prompt}]
        for turn in history:
            messages.append({"role": "user", "content": turn["question"]})
            messages.append({"role": "assistant", "content": turn["response"]})
        messages.append({"role": "user", "content": question})

        completion = call_with_retry(
            self._client.chat.completions.create,
            model=self._model,
            messages=messages,
            temperature=0.7,
            tools=_DISCLOSURE_TOOL,
            tool_choice="required",
            extra_body=self._extra_body,
        )
        args = json.loads(completion.choices[0].message.tool_calls[0].function.arguments)
        response_text = (args.get("reply") or "").strip()

        candidate_ids = {c["info_id"] for c in candidates}
        raw_disclosed = args.get("disclosed_info_ids", [])
        disclosed = [eid for eid in raw_disclosed if eid in candidate_ids]

        return response_text, {
            "provider": "patient_actor",
            "disclosed_info_ids": disclosed,
        }

    def _disclosable_candidates(self, case: dict) -> list[dict]:
        units = case["patient_information_units"]
        if units:
            return [
                {
                    "info_id": item["info_id"],
                    "text": item.get("text", ""),
                    "category": item.get("category", ""),
                    "source": item.get("source", ""),
                    "importance": item.get("importance", "supporting"),
                }
                for item in units
                if item.get("patient_visible", True)
                and not item.get("initially_visible")
            ]

        raise ValueError("Patient simulation requires information units")

    def _render_candidates(self, candidates: list[dict]) -> str:
        if not candidates:
            return ""
        header = _CANDIDATES_HEADER_ZH if self._language == "zh" else _CANDIDATES_HEADER_EN
        lines = [f"- {c['info_id']}: [{c.get('category', 'info')}] {c.get('text', '')}" for c in candidates]
        return header + "\n".join(lines) + "\n"

    def _build_system_prompt(self, view: dict) -> str:
        init = view.get("initial_presentation", {})
        gender = str(init.get("gender", "unknown"))
        admission_type = str(init.get("admission_type", ""))
        if self._language == "zh":
            gender = _GENDER_ZH.get(gender, gender)
            admission_type = _ADMISSION_ZH.get(admission_type, admission_type)

        personality_data = view.get("personality", {})
        personality_text = (
            format_for_prompt(personality_data, language=self._language)
            if personality_data
            else ("Not specified." if self._language == "en" else "未指定。")
        )
        symptom_context = self._render_patient_known_context(
            view.get("patient_information_units", []),
        )

        return self._template.format(
            age=init.get("age", "unknown"),
            gender=gender,
            admission_type=admission_type,
            vital_signs=init.get("vital_signs", "not recorded" if self._language == "en" else "未记录"),
            symptom_context=symptom_context,
            personality=personality_text,
        )

    def _render_patient_known_context(self, units: list[dict]) -> str:
        visible = [
            u for u in units
            if u.get("patient_visible", True)
        ]
        if not visible:
            raise ValueError("Patient simulation requires patient-visible information units")
        grouped: dict[str, list[str]] = {}
        for u in visible:
            grouped.setdefault(u.get("category", "info"), []).append(u.get("text", ""))
        lines: list[str] = []
        for category in sorted(grouped):
            label = category if self._language != "zh" else _CATEGORY_ZH.get(category, category)
            lines.append(f"{label}:")
            for text in grouped[category]:
                if text:
                    lines.append(f"- {text}")
        return "\n".join(lines)


_CATEGORY_ZH = {
    "chief_complaint": "主诉",
    "symptom": "症状",
    "negative_symptom": "否认/阴性症状",
    "past_medical_history": "既往史",
    "surgical_history": "手术史",
    "medication": "用药史",
    "allergy": "过敏史",
    "immunization": "免疫接种史",
    "family_history": "家族史",
    "social_history": "社会史",
    "demographic_profile": "基本信息",
}
