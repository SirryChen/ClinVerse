"""
Discharge note parsing: extract HPI, PE, Discharge Diagnosis, Discharge Procedures.
Logic mirrors paulhager/MIMIC-Clinical-Decision-Making-Dataset/dataset/discharge.py
but is generalised (no disease-specific sanitization at this stage).
"""
from __future__ import annotations

import re
from typing import Optional


# ── Section boundary patterns ──────────────────────────────────────────────────

_HPI_STARTS = re.compile(
    r"history of present(?:ing)? illness:",
    re.IGNORECASE,
)

_HPI_ENDS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"physical exam(?:ination)?:",
        r"physical ___:",
        r"pe:",
        r"pe ___:",
        r"(?:pertinent|___) results:",
        r"brief hospital course:",
        r"hospital course:",
    ]
]

_PE_STARTS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"physical exam(?:ination)?:",
        r"physical ___:",
        r"pe:",
        r"pe ___:",
    ]
]

_PE_ENDS = [
    re.compile(r"pertinent results:", re.IGNORECASE),
    re.compile(r"brief hospital course:", re.IGNORECASE),
    re.compile(r"hospital course:", re.IGNORECASE),
]

_DD_STARTS = [
    "discharge diagnosis:",
    "___ diagnosis:",
    "\n___:",
]

_DD_ENDS = [
    "discharge condition:",
    "___ condition:",
    "condition:",
    "procedures:",
    "invasive procedure on this admission:",
]

_DISCHARGE_PROC_START = re.compile(
    r"(?:major surgical or invasive procedure|procedures? performed|procedures? on this admission|discharge procedures?):",
    re.IGNORECASE,
)
_DISCHARGE_PROC_END = re.compile(
    r"(?:history of present(?:ing)? illness|attending|allergies|medications on admission|discharge medications|discharge condition|discharge instructions)",
    re.IGNORECASE,
)

# At-discharge PE removal patterns (keep only admission PE)
_DISCHARGE_PE_SUFFIXES = [
    re.compile(p, re.IGNORECASE) for p in [
        r"at discharge.*", r"upon discharge.*",
        r"on discharge.*",  r"discharge.*",
    ]
]


def _flatten(text: str) -> str:
    """Replace newlines with spaces for easier regex matching."""
    return text.replace("\n", " ")


def extract_hpi(text: str) -> str:
    """
    Extract the History of Present Illness section.
    Returns empty string if not found.
    """
    flat = _flatten(text)
    # Find start
    m_start = _HPI_STARTS.search(flat)
    if not m_start:
        return ""
    start_pos = m_start.end()

    # Find the earliest end boundary
    end_pos = len(flat)
    for pat in _HPI_ENDS:
        m = pat.search(flat, start_pos)
        if m and m.start() < end_pos:
            end_pos = m.start()

    hpi = flat[start_pos:end_pos].strip()
    return hpi


def extract_physical_examination(text: str) -> str:
    """
    Extract the Physical Examination section from admission (not discharge PE).
    Returns empty string if not found or too short.
    """
    flat = _flatten(text)
    pe_start = None
    for pat in _PE_STARTS:
        m = pat.search(flat)
        if m:
            pe_start = m.end()
            break
    if pe_start is None:
        return ""

    # Find end boundary
    end_pos = len(flat)
    for pat in _PE_ENDS:
        m = pat.search(flat, pe_start)
        if m and m.start() < end_pos:
            end_pos = m.start()

    pe = flat[pe_start:end_pos].strip()

    # Strip at-discharge PE suffix
    for pat in _DISCHARGE_PE_SUFFIXES:
        pe = pat.sub("", pe, count=1).strip()

    return pe


def extract_discharge_diagnosis(text: str) -> str:
    """
    Extract the Discharge Diagnosis free-text block.
    Returns empty string if section not found.
    """
    lower = text.lower()

    start = 0
    for hdr in _DD_STARTS:
        pos = lower.rfind(hdr)
        if pos != -1:
            start = max(start, pos + len(hdr))

    if not start:
        return ""

    end = len(text)
    for hdr in _DD_ENDS:
        pos = lower.rfind(hdr, start)
        if pos != -1 and pos < end:
            end = pos

    return text[start:end].strip()


def extract_discharge_procedures(text: str) -> list[str]:
    """
    Extract free-text discharge procedures listed in the discharge summary.
    Returns a list of non-empty procedure strings.
    """
    m_start = _DISCHARGE_PROC_START.search(text)
    if not m_start:
        return []
    start = m_start.end()

    # Find end
    m_end = _DISCHARGE_PROC_END.search(text, start)
    end = m_end.start() if m_end else len(text)

    block = text[start:end].strip()
    if not block:
        return []

    # Split on newlines / numbered items
    items = []
    for line in re.split(r"\n+", block):
        line = re.sub(r"^\s*[\d\.\-\*]+\s*", "", line).strip()
        if line:
            items.append(line)
    return items


def extract_hospital_course(text: str) -> str:
    """
    Extract the Brief Hospital Course section.
    This section describes in-hospital workup, treatment, and clinical evolution.
    Returns empty string if not found.
    """
    _COURSE_STARTS = re.compile(
        r"brief\s+hospital\s+course:",
        re.IGNORECASE,
    )
    _COURSE_ENDS = [
        re.compile(p, re.IGNORECASE) for p in [
            r"medications on admission:",
            r"discharge medications:",
            r"discharge diagnos(?:is|es):",
            r"___ diagnos(?:is|es):",
            r"discharge condition:",
            r"discharge instructions:",
            r"discharge disposition:",
        ]
    ]

    flat = _flatten(text)
    m_start = _COURSE_STARTS.search(flat)
    if not m_start:
        return ""
    start = m_start.end()

    end = len(flat)
    for pat in _COURSE_ENDS:
        m = pat.search(flat, start)
        if m and m.start() < end:
            end = m.start()

    return flat[start:end].strip()


def has_hpi_section(text: str) -> bool:
    """Quick check: does this discharge note contain an HPI section?"""
    return bool(_HPI_STARTS.search(text))


def primary_diagnosis_from_discharge(discharge_diagnosis: str) -> Optional[str]:
    """
    Return the primary (first non-empty, non-secondary) diagnosis line.
    """
    if not discharge_diagnosis:
        return None
    # Split off secondary diagnoses block
    sec_match = re.search(
        r"\n\s*secondary\s*(?:diagnos(?:is|es))?\s*:",
        discharge_diagnosis,
        re.IGNORECASE,
    )
    primary_block = discharge_diagnosis[:sec_match.start()].strip() if sec_match else discharge_diagnosis.strip()

    # Take first non-empty line
    for line in primary_block.split("\n"):
        line = re.sub(r"^\s*[\d\.\-\*]+\s*", "", line).strip()
        if line:
            return line.lower()
    return None
