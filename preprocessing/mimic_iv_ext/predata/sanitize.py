"""
Text sanitization utilities.
- Marks a patient's HPI as invalid if the primary diagnosis term appears in it
  (indicates a transfer where diagnosis was already known → not suitable for CDM task)
- Replaces remaining occurrences in PE / radiology text with '___'
"""
from __future__ import annotations

import re
from typing import Optional


def _make_pattern(terms: list[str]) -> Optional[re.Pattern]:
    if not terms:
        return None
    escaped = [re.escape(t) for t in terms if t]
    if not escaped:
        return None
    return re.compile("|".join(escaped), re.IGNORECASE)


def hpi_is_contaminated(hpi: str, terms: list[str]) -> bool:
    """Return True if the HPI explicitly names the diagnosis (transfer case)."""
    if not hpi or not terms:
        return False
    pat = _make_pattern(terms)
    return bool(pat and pat.search(hpi))


def sanitize_text(text: str, terms: list[str]) -> str:
    """Replace diagnosis terms in text with '___'."""
    if not text or not terms:
        return text
    pat = _make_pattern(terms)
    if not pat:
        return text
    return pat.sub("___", text)
