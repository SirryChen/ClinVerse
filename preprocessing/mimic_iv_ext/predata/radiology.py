"""
Radiology report parsing.
- Assigns modality and region from exam_name
- Strips IMPRESSION / CONCLUSION / INDICATION sections (prevent answer leakage)
- Returns only the Findings section content
"""
from __future__ import annotations

import logging
import re
from typing import Optional

import pandas as pd

from config import RAD_BAD_FIELDS, RAD_MODALITY_MAP, RAD_REGION_MAP

log = logging.getLogger(__name__)


# ── Modality / region detection ────────────────────────────────────────────────

def detect_modality(exam_name: str) -> Optional[str]:
    if not exam_name:
        return None
    tokens = re.split(r"[\s/,\-_]+", exam_name.upper())
    for tok in tokens:
        if tok in RAD_MODALITY_MAP:
            return RAD_MODALITY_MAP[tok]
    return None


def detect_region(exam_name: str) -> Optional[str]:
    if not exam_name:
        return None
    tokens = re.split(r"[\s/,\-_]+", exam_name.upper())
    for tok in tokens:
        if tok in RAD_REGION_MAP:
            return RAD_REGION_MAP[tok]
    # fallback: substring match
    upper = exam_name.upper()
    for key, val in RAD_REGION_MAP.items():
        if key in upper:
            return val
    return None


# ── Report section parsing ─────────────────────────────────────────────────────

def _parse_sections(text: str) -> dict[str, str]:
    """
    Split a radiology report into section-name → content dict.
    Section headers are ALL-CAPS lines followed by colon,
    or lines that are entirely upper-case.
    """
    lines = text.strip().split("\n")
    # Normalise: lines that are ALL-CAPS without colon → add colon
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.isupper() and stripped and not stripped.endswith(":"):
            lines[i] = stripped + ":"
    rejoined = "\n".join(lines)

    pattern = re.compile(
        r"(?m)^([A-Z][A-Z0-9 \t,._/-]+):((?:(?!^[A-Z][A-Z0-9 \t,._/-]+:).)*)",
        re.DOTALL,
    )
    sections = {}
    for m in pattern.finditer(rejoined):
        header  = m.group(1).strip().upper()
        content = m.group(2).strip()
        sections[header] = content
    return sections


def extract_findings(text: str) -> str:
    """
    Parse the report and return only non-bad-field sections joined together.
    If nothing usable is found, returns "".
    """
    if not text or not text.strip():
        return ""
    sections = _parse_sections(text)
    parts = []
    for header, content in sections.items():
        if not content:
            continue
        is_bad = any(header.startswith(bad) for bad in RAD_BAD_FIELDS)
        if not is_bad:
            parts.append(f"{header}:\n{content}")
    return "\n\n".join(parts).strip()


# ── Bulk radiology DataFrame processing ───────────────────────────────────────

def process_radiology(
    radiology_df: pd.DataFrame,
    radiology_detail_wide: pd.DataFrame,
    admissions_df: pd.DataFrame,
    valid_hadm_ids: set,
    hours_before: int = 24,
) -> pd.DataFrame:
    """
    Given:
        radiology_df          — all radiology notes (note_id, subject_id, hadm_id, charttime, text)
        radiology_detail_wide — wide pivot of radiology_detail (note_id, exam_name, parent_note_id)
        admissions_df         — (hadm_id, admittime)
        valid_hadm_ids        — admissions to include
        hours_before          — include reports up to this many hours before admittime

    Returns a DataFrame with columns:
        hadm_id, note_id, modality, region, exam_name, text
    where text = findings-only content.
    """
    log.info("Processing radiology reports…")

    rad = radiology_df[radiology_df["hadm_id"].isin(valid_hadm_ids)].copy()
    rad["hadm_id"] = rad["hadm_id"].astype("Int64")

    # Merge exam_name from detail table
    rad = rad.merge(radiology_detail_wide, on="note_id", how="left")

    # Resolve exam_name: some notes have parent_note_id → look up parent's exam_name
    parent_exam = (
        radiology_detail_wide[["note_id", "exam_name"]]
        .rename(columns={"note_id": "parent_note_id", "exam_name": "parent_exam_name"})
        .dropna(subset=["parent_note_id"])
    )
    rad = rad.merge(parent_exam, on="parent_note_id", how="left")
    mask_no_exam = rad["exam_name"].isna()
    rad.loc[mask_no_exam, "exam_name"] = rad.loc[mask_no_exam, "parent_exam_name"]
    rad.drop(columns=["parent_exam_name"], inplace=True, errors="ignore")

    # Time filter: keep reports in [admittime - hours_before, dischtime]
    admitmap  = admissions_df.set_index("hadm_id")["admittime"].to_dict()
    dischtmap = admissions_df.set_index("hadm_id")["dischtime"].to_dict()
    rad["admittime"] = rad["hadm_id"].map(admitmap)
    rad["dischtime"] = rad["hadm_id"].map(dischtmap)
    rad = rad.dropna(subset=["admittime", "charttime"])
    delta = pd.to_timedelta(hours_before, unit="h")
    mask_time = (
        (rad["charttime"] >= rad["admittime"] - delta) &
        (rad["dischtime"].isna() | (rad["charttime"] <= rad["dischtime"]))
    )
    rad = rad[mask_time].copy()
    rad.drop(columns=["admittime", "dischtime"], inplace=True, errors="ignore")

    # Detect modality and region
    rad["modality"] = rad["exam_name"].apply(
        lambda x: detect_modality(str(x)) if pd.notna(x) else None
    )
    rad["region"] = rad["exam_name"].apply(
        lambda x: detect_region(str(x)) if pd.notna(x) else None
    )

    # Require modality; allow region to be None/Unknown (use "Unknown" as fallback)
    rad = rad.dropna(subset=["modality"])
    rad["region"] = rad["region"].fillna("Unknown")

    # Extract findings-only text
    rad["text"] = rad["text"].apply(
        lambda t: extract_findings(t) if pd.notna(t) else ""
    )
    # Drop empty reports
    rad = rad[rad["text"].str.strip() != ""].copy()

    result = rad[["hadm_id", "note_id", "modality", "region", "exam_name", "text"]].copy()
    result = result.reset_index(drop=True)
    log.info("  %d radiology records across %d admissions",
             len(result), result["hadm_id"].nunique())
    return result
