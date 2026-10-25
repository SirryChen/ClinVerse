"""
d_references lookup helpers for SICdb encoded fields.
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from preprocessing.sicdb_ext.predata.config import REFERENCES_FILE

log = logging.getLogger(__name__)

_REF_CACHE: dict[int, dict] | None = None


def load_references() -> dict[int, dict]:
    global _REF_CACHE
    if _REF_CACHE is not None:
        return _REF_CACHE
    log.info("Loading d_references…")
    df = pd.read_csv(REFERENCES_FILE)
    _REF_CACHE = {
        int(row["ReferenceGlobalID"]): {
            "value": str(row["ReferenceValue"]) if pd.notna(row["ReferenceValue"]) else "",
            "name": str(row["ReferenceName"]) if pd.notna(row["ReferenceName"]) else "",
            "unit": str(row["ReferenceUnit"]) if pd.notna(row["ReferenceUnit"]) else "",
            "loinc": str(row["LOINC_code"]) if pd.notna(row.get("LOINC_code")) else "",
        }
        for _, row in df.iterrows()
    }
    log.info("  %d reference entries loaded", len(_REF_CACHE))
    return _REF_CACHE


def ref_value(ref_id, refs: dict | None = None) -> str:
    if ref_id is None or (isinstance(ref_id, float) and pd.isna(ref_id)):
        return ""
    refs = refs or load_references()
    entry = refs.get(int(ref_id), {})
    return entry.get("value", "")


def ref_unit(ref_id, refs: dict | None = None) -> str:
    if ref_id is None or (isinstance(ref_id, float) and pd.isna(ref_id)):
        return ""
    refs = refs or load_references()
    entry = refs.get(int(ref_id), {})
    return entry.get("unit", "")


def ref_label(ref_id, refs: dict | None = None) -> str:
    """Human-readable label: prefer ReferenceValue, fallback to ReferenceName."""
    if ref_id is None or (isinstance(ref_id, float) and pd.isna(ref_id)):
        return ""
    refs = refs or load_references()
    entry = refs.get(int(ref_id), {})
    val = entry.get("value", "")
    name = entry.get("name", "")
    return val or name


def laboratory_labels(refs: dict | None = None) -> dict[int, str]:
    refs = refs or load_references()
    return {
        rid: info["value"]
        for rid, info in refs.items()
        if info.get("name") == "Laboratory" and info.get("value")
    }
