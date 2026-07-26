"""ICD code normalization shared by every dataset adapter and by evaluation.

Built on the `icd-mappings` package (PyPI, MIT, ships its own offline GEM /
chapter data — no network calls at runtime). This is the one place that
knows how to compare diagnoses *by code* instead of by free-text keyword
matching, which is what makes scoring language-independent across hospital
systems (English MIMIC titles, German SICdb titles, ...).

    >>> to_icd10("5715", 9)
    'K7469'
    >>> block("I5012")
    'I30-I5A | Other forms of heart disease'
"""
from __future__ import annotations

import re
from functools import lru_cache

from icdmappings import Mapper

_mapper = Mapper()

# SICdb carries a small non-standard "numeric extension catalogue" for
# prosthesis/implant complications (e.g. "102.1") that isn't real ICD-10 and
# will never resolve through `icd-mappings`. Bucket it explicitly rather than
# silently returning None for these (see data/sicdb-ext/statistics/icd_utils.py
# for the same special-case, applied there for the chart-building use case).
NUMERIC_EXTENSION_CHAPTER = "Z00-Z00 | Supplementary prosthesis/implant complication codes (non-ICD)"


def _clean(code: str | None) -> str:
    """Upper-case, strip whitespace/dots — `icd-mappings` is dot-insensitive
    but normalizing here keeps category3()/equality checks consistent."""
    if not code:
        return ""
    return str(code).strip().upper().replace(".", "")


def _is_numeric_extension(code: str) -> bool:
    return bool(code) and code[0].isdigit()


@lru_cache(maxsize=4096)
def to_icd10(code: str | None, version: int) -> str | None:
    """Return an ICD-10 code for `code`, converting from ICD-9 if needed.

    version: 9 or 10 (anything else is treated as already-ICD-10 to fail open
    rather than silently drop a diagnosis).
    Returns None if the code is empty or has no known ICD-10 equivalent.
    """
    clean = _clean(code)
    if not clean:
        return None
    if version == 9:
        mapped = _mapper.map(clean, source="icd9", target="icd10")
        return _clean(mapped) if mapped else None
    return clean


@lru_cache(maxsize=4096)
def chapter(icd10_code: str | None) -> str | None:
    """Coarse grouping (e.g. 'I00-I99 | Diseases of the circulatory system').
    Used for the "category" axis of diagnosis scoring."""
    clean = _clean(icd10_code)
    if not clean:
        return None
    if _is_numeric_extension(clean):
        return NUMERIC_EXTENSION_CHAPTER
    return _mapper.map(clean, source="icd10", target="chapter")


@lru_cache(maxsize=4096)
def block(icd10_code: str | None) -> str | None:
    """Finer grouping than chapter (e.g. 'I30-I5A | Other forms of heart
    disease'). Used for the "subtype" axis of diagnosis scoring. Falls back
    to chapter() when a code has no block-level entry (this happens for some
    chapters, e.g. R96 in the search above)."""
    clean = _clean(icd10_code)
    if not clean:
        return None
    if _is_numeric_extension(clean):
        return NUMERIC_EXTENSION_CHAPTER
    b = _mapper.map(clean, source="icd10", target="block")
    return b if b else chapter(clean)


def category3(icd10_code: str | None) -> str:
    """Three-character category prefix (e.g. 'I50' for I50.12). Pure string
    slicing — no external data needed — used as a finer-grained tiebreaker
    than block() and reused by the dataset statistics scripts."""
    clean = _clean(icd10_code)
    if not clean:
        return ""
    digits_or_letters = re.match(r"^[A-Z0-9]{1,3}", clean)
    return digits_or_letters.group(0) if digits_or_letters else clean[:3]


def same_block(code_a: str | None, code_b: str | None) -> bool:
    ba, bb = block(code_a), block(code_b)
    return bool(ba) and ba == bb


def same_category3(code_a: str | None, code_b: str | None) -> bool:
    ca, cb = category3(code_a), category3(code_b)
    return bool(ca) and ca == cb
