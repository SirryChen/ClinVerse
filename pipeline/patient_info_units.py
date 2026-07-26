"""Build patient-visible atomic information units for history-taking.

The target shape is intentionally small and model-agnostic so a future LLM
extractor can replace the heuristic fallback without changing PatientActor or
evaluation code.
"""

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

_OBJECTIVE_PATTERNS = re.compile(
    r"\b("
    r"lab|labs|cbc|wbc|rbc|hgb|hct|platelet|inr|ptt|sodium|potassium|"
    r"creatinine|troponin|lactate|urinalysis|ua|cxr|x[- ]?ray|ct|mri|"
    r"ultrasound|u/s|imaging|biopsy|pathology|culture|blood culture|"
    r"received|given|treated with|started on|ed report|vitals|vs were"
    r")\b",
    re.IGNORECASE,
)

_NON_CLINICAL_BACKGROUND = {
    "Medication review due (situation)",
    "Full-time employment (finding)",
    "Part-time employment (finding)",
    "Unemployed (finding)",
    "Limited social contact (finding)",
    "Social isolation (finding)",
    "Stress (finding)",
}


class PatientInfoExtractor(Protocol):
    def extract(self, payload: dict) -> list[dict]:
        ...


@dataclass
class HttpPatientInfoExtractor:
    """Thin HTTP interface for a future vLLM/OpenAI-compatible extractor.

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
- initially_visible: true only for age/gender/admission facts already shown at encounter start

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
    """Return the structured input a future LLM extractor should consume."""
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
    """Build normalized patient-visible information units.

    When `extractor` is supplied, it is responsible for atomizing the MIMIC HPI
    and fused background. The local heuristic fallback keeps the pipeline usable
    without network/model dependencies.
    """
    payload = build_extraction_payload(admission, trimmed_persona)
    raw_items = extractor.extract(payload) if extractor is not None else _heuristic_extract(admission, trimmed_persona)
    return _normalize_items(raw_items)


def build_patient_information_units_from_case(case: dict) -> list[dict]:
    """Build information units for legacy case JSONs that predate this field."""
    patient_view = case.get("simulation_views", {}).get("patient_actor_view", {})
    return build_patient_information_units(
        _admission_from_case(case),
        _persona_from_case_background(case, patient_view),
    )


def _heuristic_extract(admission: dict, trimmed_persona: dict) -> list[dict]:
    items: list[dict] = []
    demo = admission.get("demographics", {})

    if demo.get("anchor_age") is not None:
        items.append({
            "text": f"{demo.get('anchor_age')} years old",
            "category": "demographic_profile",
            "source": "mimic_demographics",
            "importance": "background",
            "patient_visible": True,
            "initially_visible": True,
        })
    if demo.get("gender"):
        items.append({
            "text": f"Gender: {demo.get('gender')}",
            "category": "demographic_profile",
            "source": "mimic_demographics",
            "importance": "background",
            "patient_visible": True,
            "initially_visible": True,
        })
    if demo.get("marital_status"):
        items.append({
            "text": f"Marital status: {demo.get('marital_status')}",
            "category": "social_history",
            "source": "mimic_demographics",
            "importance": "background",
            "patient_visible": True,
            "initially_visible": False,
        })

    for sent in _split_patient_visible_hpi(admission.get("hpi", "")):
        category = _classify_hpi_sentence(sent)
        items.append({
            "text": sent,
            "category": category,
            "source": "mimic_hpi",
            "importance": "key" if category in {"chief_complaint", "symptom", "negative_symptom"} else "supporting",
            "patient_visible": True,
            "initially_visible": False,
        })

    for cond in trimmed_persona.get("conditions", []):
        text = cond.get("display", "").strip()
        if not text or text in _NON_CLINICAL_BACKGROUND:
            continue
        items.append({
            "text": text,
            "category": "past_medical_history",
            "source": "synthea",
            "importance": "supporting",
            "patient_visible": True,
            "initially_visible": False,
        })

    for med in trimmed_persona.get("medications", []):
        text = med.get("display", "").strip()
        if text:
            items.append({
                "text": text,
                "category": "medication",
                "source": "synthea",
                "importance": "supporting",
                "patient_visible": True,
                "initially_visible": False,
            })

    for allergy in trimmed_persona.get("allergies", []):
        text = allergy.get("substance", "").strip()
        if text:
            items.append({
                "text": text,
                "category": "allergy",
                "source": "synthea",
                "importance": "supporting",
                "patient_visible": True,
                "initially_visible": False,
            })

    for imm in trimmed_persona.get("immunizations", []):
        text = imm.get("vaccine", "").strip()
        if text:
            items.append({
                "text": text,
                "category": "immunization",
                "source": "synthea",
                "importance": "background",
                "patient_visible": True,
                "initially_visible": False,
            })

    for obs in trimmed_persona.get("observations", []):
        display = obs.get("display", "").strip()
        if not display:
            continue
        if not any(kw in display.lower() for kw in ("smoking", "tobacco", "alcohol", "bmi", "body mass")):
            continue
        value = f"{obs.get('value', '')} {obs.get('unit', '')}".strip()
        text = f"{display}: {value}" if value else display
        items.append({
            "text": text,
            "category": "social_history",
            "source": "synthea",
            "importance": "supporting",
            "patient_visible": True,
            "initially_visible": False,
        })

    return items


def _patient_background_to_persona(background: dict) -> dict:
    social_history = background.get("social_history", {})
    return {
        "conditions": [{"display": x} for x in background.get("past_medical_history", [])],
        "medications": [{"display": x} for x in background.get("long_term_medications", [])],
        "allergies": [{"substance": x} for x in background.get("allergies", [])],
        "immunizations": [{"vaccine": x} for x in background.get("immunizations", [])],
        "observations": [
            {"display": key, "value": value, "unit": ""}
            for key, value in social_history.items()
        ],
    }


def _admission_from_case(case: dict) -> dict:
    if case.get("core_admission"):
        return case["core_admission"]
    return {
        "demographics": case.get("demographics", {}),
        "hpi": case.get("hpi", ""),
    }


def _persona_from_case_background(case: dict, patient_view: dict) -> dict:
    background = case.get("patient_background")
    if background:
        return _patient_background_to_persona(background)
    medical = patient_view.get("medical_background", {})
    social_history = medical.get("social_history", {})
    return {
        "conditions": [{"display": x} for x in medical.get("past_medical_history", [])],
        "medications": [{"display": x} for x in medical.get("medications", [])],
        "allergies": [{"substance": x} for x in medical.get("allergies", [])],
        "immunizations": [{"vaccine": x} for x in medical.get("immunizations", [])],
        "observations": [
            {"display": key, "value": value, "unit": ""}
            for key, value in social_history.items()
        ],
    }


def _split_patient_visible_hpi(text: str) -> list[str]:
    if not text:
        return []
    cleaned = re.sub(r"\s+", " ", str(text)).strip()
    parts = re.split(r"(?<=[.!?])\s+|;\s+", cleaned)
    out: list[str] = []
    for part in parts:
        part = part.strip(" -")
        if len(part) < 12:
            continue
        if _OBJECTIVE_PATTERNS.search(part):
            continue
        if part.lower().startswith(("past medical history:", "social history:", "family history:")):
            # These sections often contain multiple facts; keep for fallback but
            # split on coarse delimiters below when possible.
            section = re.sub(r"^(past medical history|social history|family history):\s*", "", part, flags=re.I)
            for sub in re.split(r"\s+#\s+|\s{2,}|,\s+", section):
                sub = sub.strip(" -")
                if len(sub) >= 8 and not _OBJECTIVE_PATTERNS.search(sub):
                    out.append(sub)
            continue
        out.append(part)
    return out[:40]


def _classify_hpi_sentence(text: str) -> str:
    low = text.lower()
    if low.startswith(("no ", "denies ", "without ")) or " denies " in low or " no " in low:
        return "negative_symptom"
    if "family history" in low or low.startswith(("mother", "father", "brother", "sister")):
        return "family_history"
    if "social history" in low or any(kw in low for kw in ("smok", "alcohol", "etoh", "ivdu", "drug use")):
        return "social_history"
    if any(kw in low for kw in ("history of", "h/o", "pmh", "status post", "s/p")):
        return "past_medical_history"
    return "symptom"


def _normalize_items(raw_items: list[dict]) -> list[dict]:
    normalized: list[dict] = []
    seen: set[tuple[str, str]] = set()
    counters: dict[str, int] = {}

    for raw in raw_items:
        text = str(raw.get("text") or raw.get("content") or "").strip()
        if not text:
            continue
        category = str(raw.get("category") or "history").strip()
        category = _canonical_category(category)
        if _looks_like_marital_status(text):
            category = "social_history"
        if category not in INFO_CATEGORIES:
            category = "past_medical_history"
        key = (category, re.sub(r"\s+", " ", text.lower()))
        if key in seen:
            continue
        seen.add(key)
        counters[category] = counters.get(category, 0) + 1
        info_id = raw.get("info_id") or raw.get("id") or f"info_{category}_{counters[category]:03d}"
        normalized.append({
            "info_id": str(info_id),
            "text": text,
            "category": category,
            "source": str(raw.get("source") or "unknown"),
            "importance": _canonical_importance(str(raw.get("importance") or "supporting")),
            "patient_visible": bool(raw.get("patient_visible", True)),
            "initially_visible": bool(raw.get("initially_visible", False))
            and not _looks_like_marital_status(text),
        })
    return normalized


def _canonical_category(category: str) -> str:
    c = category.lower().strip().replace("-", "_")
    aliases = {
        "profile": "demographic_profile",
        "demographics": "demographic_profile",
        "history": "past_medical_history",
        "pmh": "past_medical_history",
        "surgery": "surgical_history",
        "surgical": "surgical_history",
        "medicine": "medication",
        "medications": "medication",
        "negative": "negative_symptom",
    }
    return aliases.get(c, c)


def _canonical_importance(value: str) -> str:
    v = value.lower().strip()
    if v in {"key", "critical", "red_flag", "red-flag"}:
        return "key"
    if v in {"background", "low"}:
        return "background"
    return "supporting"


def _looks_like_marital_status(text: str) -> bool:
    low = text.lower()
    return any(word in low for word in ("marital", "widow", "married", "divorced", "single"))
