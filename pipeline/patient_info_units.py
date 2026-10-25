"""Model-based patient information-unit extraction and schema validation."""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass
from typing import Protocol


INFO_CATEGORIES = {
    "chief_complaint",
    "symptom",
    "negative_symptom",
    "past_medical_history",
    "surgical_history",
    "medication",
    "allergy",
    "immunization",
    "family_history",
    "social_history",
    "demographic_profile",
}

class PatientInfoExtractor(Protocol):
    def extract(self, payload: dict) -> list[dict]:
        ...


@dataclass
class HttpPatientInfoExtractor:
    """HTTP information-unit extractor.

    Expected response format:
        {"items": [{"text": "...", "category": "symptom", ...}, ...]}
    or:
        [{"text": "...", "category": "symptom", ...}, ...]
    """

    endpoint: str
    model: str | None = None
    timeout: int = 120

    def extract(self, payload: dict) -> list[dict]:
        body = {"model": self.model, "payload": payload} if self.model else {"payload": payload}
        req = urllib.request.Request(
            self.endpoint,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if isinstance(data, dict):
            data = data.get("items") or data.get("patient_information_units") or []
        if not isinstance(data, list):
            raise ValueError("Patient info extractor response must be a list or contain an items list")
        return data


@dataclass
class OpenAIChatPatientInfoExtractor:
    """OpenAI-compatible chat extractor for AIU generation.

    This works with hosted OpenAI-compatible services and vLLM servers. The
    model is asked to return JSON only; the normalizer below still enforces the
    final schema used by prompts and scoring.
    """

    api_key: str
    base_url: str
    model: str
    timeout: int = 120
    extra_body: dict | None = None

    def extract(self, payload: dict) -> list[dict]:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        completion = client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            extra_body=self.extra_body or {},
        )
        content = completion.choices[0].message.content or ""
        data = _parse_json_payload(content)
        if isinstance(data, dict):
            data = data.get("items") or data.get("patient_information_units") or []
        if not isinstance(data, list):
            raise ValueError("Patient info chat extractor must return a JSON list or object with items")
        return data


_EXTRACTION_SYSTEM_PROMPT = """\
You extract patient-visible Atomic Information Units (AIUs) for a medical patient simulator.
Return JSON only, no markdown. The JSON must be either a list or {"items": [...]}.
Each item must contain:
- text: one minimal self-contained fact the patient can report
- category: one of the allowed_categories in the input
- source: "mimic_hpi", "mimic_demographics", or "synthea"
- importance: "key", "supporting", or "background"
- patient_visible: true unless the fact is objective clinician-only information
- initially_visible: true for chief_complaint and demographic_profile units; false for the remaining history
- Include a concise chief_complaint unit that states the main reason for presentation, without revealing the full HPI

Rules:
- Preserve negation, timing, quantities, and uncertainty.
- Do not infer facts.
- Do not include labs, imaging, pathology, microbiology, physical exam findings, procedures, hospital course conclusions, final diagnoses, or clinician interpretations.
- Split compound statements into the smallest useful facts.
"""


def _parse_json_payload(content: str):
    text = content.strip()
    if not text:
        raise ValueError("Patient info chat extractor returned empty content")
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(1))


def build_extraction_payload(admission: dict, trimmed_persona: dict) -> dict:
    """Return the information-unit extraction input."""
    demo = admission.get("demographics", {})
    return {
        "instructions": (
            "Extract patient-visible atomic information units only. Do not include lab values, "
            "imaging findings, pathology, microbiology, physical exam findings, hospital course "
            "conclusions, final diagnoses, or clinician interpretations. Do not infer facts."
        ),
        "allowed_categories": sorted(INFO_CATEGORIES),
        "mimic": {
            "demographics": {
                "age": demo.get("anchor_age"),
                "gender": demo.get("gender"),
                "marital_status": demo.get("marital_status"),
                "admission_type": demo.get("admission_type"),
            },
            "hpi": admission.get("hpi", ""),
        },
        "synthea_background": {
            "conditions": trimmed_persona.get("conditions", []),
            "medications": trimmed_persona.get("medications", []),
            "allergies": trimmed_persona.get("allergies", []),
            "immunizations": trimmed_persona.get("immunizations", []),
            "observations": trimmed_persona.get("observations", []),
        },
    }


def build_patient_information_units(
    admission: dict,
    trimmed_persona: dict,
    extractor: PatientInfoExtractor | None = None,
) -> list[dict]:
    if extractor is None:
        raise ValueError("Configure an information-unit extractor before building cases")
    raw = extractor.extract(build_extraction_payload(admission, trimmed_persona))
    if not isinstance(raw, list) or not raw:
        raise ValueError("The extractor must return a nonempty information-unit list")
    items = []
    for index, item in enumerate(raw, 1):
        if not isinstance(item, dict):
            raise ValueError("Each information unit must be an object")
        item = dict(item)
        item["info_id"] = f"info_{index:04d}"
        items.append(item)
    validate_information_units(items)
    return items


def validate_information_units(items: list[dict]) -> None:
    if not isinstance(items, list) or not items:
        raise ValueError("Cases require nonempty patient_information_units")
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each information unit must be an object")
        if not isinstance(item.get("info_id"), str) or not item["info_id"] or item["info_id"] in seen:
            raise ValueError("Information-unit IDs must be unique nonempty strings")
        seen.add(item["info_id"])
        if not isinstance(item.get("text"), str) or not item["text"].strip():
            raise ValueError("Information units require text")
        if item.get("category") not in INFO_CATEGORIES:
            raise ValueError("Information-unit category is not supported")
        if item.get("importance") not in {"key", "supporting", "background"}:
            raise ValueError("Information-unit importance is not supported")
        if not isinstance(item.get("source"), str) or not item["source"]:
            raise ValueError("Information units require a source")
        for name in ("patient_visible", "initially_visible"):
            if not isinstance(item.get(name), bool):
                raise ValueError(f"Information-unit {name} must be boolean")
        if item['initially_visible'] and not item['patient_visible']:
            raise ValueError('Initially visible units must be patient-visible')
        if item['category'] in {'chief_complaint', 'demographic_profile'} and item['patient_visible'] and not item['initially_visible']:
            raise ValueError('Presentation units must be initially visible')



def require_patient_information_units(case: dict) -> list[dict]:
    items = case.get("patient_information_units")
    validate_information_units(items)
    return items
