"""OpenAI function-call tool schemas for the doctor agent."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.ports import DatasetAdapter


def build_doctor_tools(adapter: "DatasetAdapter") -> list:
    """Return DOCTOR_TOOLS with descriptions fully enumerated from the
    adapter's catalogs, so the doctor agent cannot order outside the
    available scope for *this* hospital system.

    When adapter.has_radiology is False (e.g. sicdb-ext), order_imaging is
    dropped from the returned tool list entirely rather than kept around
    just to always answer "not available" — a tool that can never produce a
    result for this dataset shouldn't be offered as an option at all.
    """
    tests_desc = adapter.lab_catalog.describe_for_prompt()

    tools = copy.deepcopy(DOCTOR_TOOLS)
    if not adapter.has_radiology:
        tools = [t for t in tools if t["function"]["name"] != "order_imaging"]

    for tool in tools:
        fn = tool["function"]
        if fn["name"] == "order_tests":
            fn["description"] = tests_desc
            fn["parameters"]["properties"]["tests"]["description"] = (
                "List of test canonical names or panel names to order. "
                "Must be exact canonical names from the catalog enumerated in the tool description. "
                "Do not use vague requests such as 'blood work', 'routine labs', or 'full panel'."
            )
        elif fn["name"] == "order_imaging":
            fn["description"] = adapter.radiology_catalog.describe_for_prompt()
            fn["parameters"]["properties"]["studies"]["description"] = (
                "List of imaging study canonical names to order. "
                "Must be exact canonical names from the catalog enumerated in the tool description. "
                "Do not use vague requests such as 'imaging', 'scan', or 'full workup'."
            )
    return tools


DOCTOR_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "ask_patient",
            "description": (
                "Ask the patient a question about symptoms, history, lifestyle, "
                "medications, allergies, or other patient-known background. "
                "Do not use this for lab values, imaging findings, exam findings, pathology, "
                "or diagnostic interpretation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to ask the patient.",
                    },
                    "topics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Topics covered by this question, e.g. "
                            "'symptoms', 'onset', 'past_medical_history', "
                            "'medications', 'allergies', 'social_history', 'family_history'."
                        ),
                    },
                },
                "required": ["question", "topics"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "request_physical_exam",
            "description": (
                "Request specific components of the physical examination. "
                "Use this for objective exam findings; ask for focused components rather than a broad sweep."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "components": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "PE components to examine, e.g. 'general', 'vital_signs', "
                            "'cardiovascular', 'pulmonary', 'abdomen', 'neurological', "
                            "'extremities', 'heent'."
                        ),
                    },
                    "reason": {"type": "string"},
                },
                "required": ["components", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "order_tests",
            "description": (
                "Order laboratory tests or microbiology cultures. "
                "Use canonical panel names (CBC, BMP, CMP, LFT, coagulation, cardiac_markers, "
                "ABG, VBG, UA, lipid_panel, thyroid, iron_studies, renal_function, inflammatory) "
                "or individual test canonical names "
                "(e.g. creatinine, troponin_t, sodium, lipase, blood_culture). "
                "Use build_doctor_tools() to get the full enumerated list."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tests": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "List of exact test canonical names or panel names to order. "
                            "Avoid vague requests such as 'blood work', 'routine labs', or 'full panel'."
                        ),
                    },
                    "reason": {"type": "string"},
                },
                "required": ["tests", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "order_imaging",
            "description": (
                "Order imaging studies. Use canonical names: "
                "chest_xray, ct_head, ct_chest, ct_abdomen_pelvis, ct_spine, ct_neck, "
                "mri_head, mri_spine, mri_abdomen, "
                "ultrasound_abdomen, ultrasound_venous, ultrasound_extremity, echocardiogram."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "studies": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "List of exact imaging study canonical names to order. "
                            "Avoid vague requests such as 'imaging', 'scan', or 'full workup'."
                        ),
                    },
                    "reason": {"type": "string"},
                },
                "required": ["studies", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finalize_case",
            "description": "Submit your final diagnosis and close the case.",
            "parameters": {
                "type": "object",
                "properties": {
                    "final_diagnosis": {
                        "type": "string",
                        "description": "Your final clinical diagnosis (free text, any language).",
                    },
                    "icd10_code": {
                        "type": "string",
                        "description": (
                            "Your best-guess ICD-10 code for the final diagnosis, e.g. 'I50.12' or "
                            "'K70.30'. This is the field used for scoring — it is compared by code, "
                            "not by text, so it works the same way regardless of what language you "
                            "answered in. If you are not confident down to the decimal, a 3-character "
                            "category code is fine, e.g. 'I50' or 'K70'. Always provide your single "
                            "best estimate even if uncertain — never leave this empty."
                        ),
                    },
                    "diagnosis_category": {
                        "type": "string",
                        "description": (
                            "Broad disease category in English, e.g. 'pneumonia', 'heart_failure', "
                            "'sepsis', 'gi_bleeding', 'pancreatitis'. MUST be in English. "
                            "Human-readable only — not scored (icd10_code is)."
                        ),
                    },
                    "diagnosis_subtype": {
                        "type": "string",
                        "description": (
                            "Specific subtype or etiology in English, "
                            "e.g. 'community-acquired pneumonia'. MUST be in English. "
                            "Human-readable only — not scored (icd10_code is)."
                        ),
                    },
                    "differential_diagnoses": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Other diagnoses considered.",
                    },
                    "supporting_evidence": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Key findings that support your diagnosis.",
                    },
                },
                "required": [
                    "final_diagnosis",
                    "icd10_code",
                    "diagnosis_category",
                    "diagnosis_subtype",
                    "differential_diagnoses",
                    "supporting_evidence",
                ],
            },
        },
    },
]
