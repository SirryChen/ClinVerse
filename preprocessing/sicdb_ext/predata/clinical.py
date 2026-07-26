"""
Clinical text assembly for SICdb.
No discharge-summary free text; HPI / PE / course are built from cases,
data_ref, data_range, and hourly vital signals (data_float_h).
"""
from __future__ import annotations

import re
from typing import Any

import pandas as pd

from config import PE_HOURS_AFTER_ICU, SEX_FEMALE_REF, SEX_MALE_REF
from references import load_references, ref_label, ref_unit, ref_value


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.lower().strip()
        if key and key not in seen:
            seen.add(key)
            out.append(item.strip())
    return out


def _s(val: Any) -> str:
    """NaN-safe stringify. `val or ""` is WRONG for floats: bool(float('nan')) is
    True in Python, so `nan or fallback` evaluates to nan itself, not fallback —
    that bug used to leak literal "nan" strings into HPI / discharge_diagnosis /
    icd_diagnosis whenever a cases.csv field (e.g. DiagnosisT2) was missing.
    """
    if val is None:
        return ""
    if isinstance(val, float) and pd.isna(val):
        return ""
    if isinstance(val, str) and val.strip().lower() == "nan":
        return ""
    return str(val).strip()


# Values that carry no clinical information and should never be surfaced as
# free text (SICdb encodes "no answer" / template placeholders this way).
_JUNK_VALUES = {"unknown", "no", "nein", "ja", "add a new one", ""}


def _meaningful(val: str) -> bool:
    return bool(val) and val.strip().lower() not in _JUNK_VALUES


def sex_to_gender(sex_ref) -> str | None:
    if pd.isna(sex_ref):
        return None
    sid = int(sex_ref)
    if sid == SEX_MALE_REF:
        return "M"
    if sid == SEX_FEMALE_REF:
        return "F"
    return ref_value(sex_ref)[:1].upper() or None


def icd_version_from_code(code: str) -> int:
    """SICdb (Salzburg, Austria) codes diagnoses exclusively in ICD-10-GM —
    including the U-chapter (e.g. U07.1 COVID-19) and the numeric supplementary
    catalogue (e.g. "102.1" prosthesis complications). There is no ICD-9 in this
    dataset, so a regex that special-cased "starts with a letter" (and therefore
    excluded 'U') would silently mis-tag ~190 real ICD-10 codes as ICD-9.
    """
    return 10


def terms_for_diagnosis(icd_code: str, long_title: str) -> list[str]:
    terms: list[str] = []
    if long_title:
        terms.append(long_title)
    if icd_code:
        terms.append(str(icd_code))
    out: list[str] = []
    seen: set[str] = set()
    for t in terms:
        k = t.lower()
        if k and k not in seen and len(t) >= 4:
            seen.add(k)
            out.append(t)
        for tok in re.split(r"[\s,;()/]+", t):
            tok = tok.strip(".")
            if len(tok) >= 7 and tok.lower() not in seen:
                seen.add(tok.lower())
                out.append(tok)
    return out


def primary_from_case(row: pd.Series) -> dict | None:
    code = _s(row.get("ICD10Main"))
    title = _s(row.get("ICD10MainText")) or _s(row.get("DiagnosisT2"))
    if not code and not title:
        return None
    return {
        "icd_code": code or None,
        "icd_version": icd_version_from_code(code) if code else 10,
        "long_title": title or code,
    }


def data_ref_map(data_ref_df: pd.DataFrame, case_id: int, refs: dict) -> dict[str, str]:
    if data_ref_df.empty:
        return {}
    sub = data_ref_df[data_ref_df["CaseID"] == case_id]
    out: dict[str, str] = {}
    for _, r in sub.iterrows():
        field = ref_value(r.get("FieldID"), refs)
        val = ref_value(r.get("RefID"), refs)
        if field and _meaningful(val):
            out[field] = val
    return out


def build_hpi(row: pd.Series, data_ref_df: pd.DataFrame, refs: dict) -> str:
    parts: list[str] = []

    urgency = ref_label(row.get("AdmissionUrgency"), refs)
    surg_type = ref_label(row.get("SurgicalAdmissionType"), refs)
    referring = ref_label(row.get("ReferringUnit"), refs)
    sepsis = ref_label(row.get("AdmissionFormHasSepsis"), refs)

    # "Unknown" / "No" carry no information for these fields ~30-40% of the
    # time (SICdb leaves them unanswered) — drop the filler instead of
    # padding the HPI with uninformative sentences.
    if _meaningful(urgency):
        parts.append(f"Admission urgency: {urgency}.")
    if _meaningful(surg_type):
        parts.append(f"Surgical admission type: {surg_type}.")
    if _meaningful(referring):
        parts.append(f"Referring unit: {referring}.")
    if _meaningful(sepsis):
        parts.append(f"Sepsis on admission form: {sepsis}.")

    # NOTE: deliberately do NOT state the primary diagnosis (ICD10MainText /
    # ICD10Main) here. Unlike MIMIC's free-text HPI — written by a clinician
    # before the diagnosis is settled, so it almost never names the answer —
    # this HPI is *assembled from the coded diagnosis itself*. Printing
    # "Primary diagnosis: X (ICD-10: Y)" would hand the clinical-decision-
    # making task's ground truth straight to the doctor agent / patient actor
    # on turn one. Only the secondary free-text diagnosis (DiagnosisT2) is
    # surfaced, and only when it differs from the primary — same rule as
    # build_discharge_diagnosis(). run.py also sanitizes the final HPI against
    # the primary-diagnosis terms as defense in depth.
    dx2 = _s(row.get("DiagnosisT2"))
    icd_text = _s(row.get("ICD10MainText"))
    if dx2 and dx2 != icd_text:
        parts.append(f"Known diagnosis: {dx2}.")

    preconds = data_ref_map(data_ref_df, int(row["hadm_id"]), refs)
    pmh_keys = [
        "PreconditionDiabetes", "PreconditionArtHypertension", "PreconditionLungDisease",
        "PreconditionRenalDysfunction", "PremedicationBetablocker", "PremedicationACEInhibitor",
        "PremedicationStatin", "PremedicationDiuretic", "NYHA", "PostCovid", "SarsCoV2Pneumonia",
    ]
    pmh = [f"{k}: {preconds[k]}" for k in pmh_keys if k in preconds]
    if pmh:
        parts.append("Past medical history / preconditions: " + "; ".join(_dedupe(pmh)) + ".")

    offset_first = row.get("OffsetAfterFirstAdmission")
    if pd.notna(offset_first) and int(offset_first) > 0:
        days = int(offset_first) // 86400
        parts.append(f"Readmission; {days} day(s) after first ICU admission.")

    saps = row.get("saps3")
    if pd.notna(saps):
        parts.append(f"SAPS III score: {saps}.")

    return " ".join(parts).strip()


def build_physical_examination(
    case_id: int,
    vitals: dict[int, list[float]],
    refs: dict,
) -> str:
    """Aggregate ICU-admission-window vitals into a PE string."""
    if not vitals:
        return ""

    lines: list[str] = []
    for data_id, values in sorted(vitals.items()):
        if not values:
            continue
        label = ref_value(data_id, refs) or ref_label(data_id, refs)
        unit = ref_unit(data_id, refs)
        if not label:
            continue

        # SICdb labels cumulative-per-hour quantities (e.g. "Urine (c)") with a
        # "(c)" suffix; data_float_h.Val for those is the hourly total, so the
        # PE-window value should be the SUM across hours, not the mean of
        # hourly totals. Continuous signals (HR, BP, SpO2, …) are averaged.
        if label.rstrip().endswith("(c)"):
            label = label.rsplit("(c)", 1)[0].strip() + " (24h total)"
            agg = sum(values)
        else:
            agg = sum(values) / len(values)

        val_str = f"{agg:.1f}" if abs(agg - round(agg)) > 0.05 else str(int(round(agg)))
        if unit:
            lines.append(f"{label}: {val_str} {unit}")
        else:
            lines.append(f"{label}: {val_str}")

    return "; ".join(_dedupe(lines))


def build_hospital_course(
    row: pd.Series,
    data_range_df: pd.DataFrame,
    data_ref_df: pd.DataFrame,
    refs: dict,
) -> str:
    parts: list[str] = []
    case_id = int(row["hadm_id"])

    if not data_range_df.empty:
        sub = data_range_df[data_range_df["CaseID"] == case_id].copy()
        if not sub.empty:
            devices: list[str] = []
            for _, r in sub.iterrows():
                name = ref_label(r.get("DataID"), refs)
                if name:
                    devices.append(name)
            devices = _dedupe(devices)
            if devices:
                parts.append("Devices / lines: " + "; ".join(devices[:40]) + ".")

    preconds = data_ref_map(data_ref_df, case_id, refs)
    interventions = [
        k for k in preconds
        if k.startswith("Intervention") or k.startswith("HeartSurgery")
    ]
    if interventions:
        iv_lines = [f"{k}: {preconds[k]}" for k in interventions]
        parts.append("Interventions: " + "; ".join(iv_lines) + ".")

    cpb = row.get("HeartSurgeryCPBTime")
    if pd.notna(cpb) and int(cpb) > 0:
        parts.append(f"Cardiopulmonary bypass time: {int(cpb)} min.")

    crrt = row.get("HoursOfCRRT")
    if pd.notna(crrt) and int(crrt) > 0:
        parts.append(f"CRRT hours: {int(crrt)}.")

    stay_days = row.get("HospitalStayDays")
    if pd.notna(stay_days):
        parts.append(f"Hospital stay: {int(stay_days)} day(s).")

    return " ".join(parts).strip()


def build_discharge_diagnosis(row: pd.Series) -> str:
    """
    Only the secondary diagnosis (DiagnosisT2, when it differs from the coded
    primary) is included. The primary diagnosis itself is deliberately left
    out: run.py sanitizes this field against the very same icd_code/long_title
    terms used to build it, so a "[Primary] {title} (ICD-10: {code})" line
    would always be fully masked to "[Primary] ___ (ICD-10: ___)" — pure noise.
    Unlike MIMIC's free-text discharge summary (which only occasionally repeats
    the coded diagnosis verbatim), SICdb has no free text, so that line can
    never survive sanitization.
    """
    title = _s(row.get("ICD10MainText"))
    dx2 = _s(row.get("DiagnosisT2"))
    if dx2 and dx2 != title:
        return f"[Secondary] {dx2}"
    return ""


def extract_discharge_procedures(
    case_id: int,
    data_range_df: pd.DataFrame,
    refs: dict,
) -> list[str]:
    if data_range_df.empty:
        return []
    sub = data_range_df[data_range_df["CaseID"] == case_id]
    procs = _dedupe(ref_label(r.get("DataID"), refs) for _, r in sub.iterrows())
    return [p for p in procs if p]
