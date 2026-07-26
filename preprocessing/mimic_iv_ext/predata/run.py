"""
MIMIC-IV-Ext Full Dataset Builder
==================================
Generalised version of paulhager/MIMIC-Clinical-Decision-Making-Dataset,
extended to ALL primary diagnoses in MIMIC-IV 3.1 (not just 4 abdominal pathologies).

Output directory: dataset/mimic-iv-ext/
  history_of_present_illness.csv
  physical_examination.csv
  discharge_diagnosis.csv
  discharge_procedures.csv
  laboratory_tests.csv
  microbiology.csv
  radiology_reports.csv
  icd_diagnosis.csv
  icd_procedures.csv
  lab_test_mapping.csv
  primary_diagnosis_index.json   -- {hadm_id: {icd_code, icd_version, long_title}}
  stats.json

Usage:
    python preprocessing/mimic_iv_ext/predata/run.py [--limit N] [--min-pe-length 40]
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict

import pandas as pd

# Add the predata directory to sys.path so sibling modules import cleanly,
# regardless of how run.py is invoked (python run.py  OR  python -m ...).
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import config    # noqa: E402
import loader    # noqa: E402
import discharge # noqa: E402
import labs      # noqa: E402
import radiology # noqa: E402
import sanitize  # noqa: E402
import export    # noqa: E402

from config import (  # noqa: E402
    OUT_HPI, OUT_PE, OUT_DISCHARGE_DX, OUT_DISCHARGE_PR,
    OUT_LABS, OUT_MICRO, OUT_RADIOLOGY,
    OUT_ICD_DX, OUT_ICD_PROC, OUT_LAB_MAPPING,
    OUT_DEMOGRAPHICS, OUT_MEDICATIONS, OUT_HOSP_COURSE,
    OUT_DX_INDEX, OUT_STATS,
    MIN_PE_LENGTH,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_primary_diagnosis(diag_df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per hadm_id: the seq_num==1 (primary) diagnosis."""
    primary = diag_df[diag_df["seq_num"] == 1].copy()
    # Keep only first primary per admission (should be unique, but guard)
    primary = primary.sort_values("seq_num").drop_duplicates(subset="hadm_id", keep="first")
    return primary.set_index("hadm_id")


def _terms_for_icd(long_title: str) -> list[str]:
    """
    Return a list of distinctive multi-word phrases to use for HPI contamination
    detection and text sanitization.

    Strategy:
    - Use the full long_title as one phrase (catches exact mentions).
    - Additionally extract distinctive tokens (>=6 chars, not stopwords) so that
      partial disease names are also caught.
    - Common English words are excluded to avoid false positives.
    """
    if not long_title or pd.isna(long_title):
        return []

    _STOPWORDS = {
        "with", "without", "other", "unspecified", "type", "due",
        "acute", "chronic", "disease", "disorder", "syndrome",
        "failure", "including", "specified", "primary", "secondary",
        "multiple", "single", "bilateral", "unilateral", "initial",
        "subsequent", "sequela", "encounter", "complication", "history",
        "patient", "following", "resulting", "associated",
    }

    title = str(long_title).strip()
    # Phrase-level check: full long_title
    terms = [title]

    # Token-level: very specific tokens (≥7 chars, not a stopword)
    for tok in re.split(r"[\s,;()/]+", title):
        tok = tok.strip(".()")
        if len(tok) >= 7 and tok.lower() not in _STOPWORDS:
            terms.append(tok)

    return terms


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run(limit: int = 0, min_pe_length: int = MIN_PE_LENGTH):
    t0 = time.time()
    export.ensure_output_dir()

    # ── 1. Load structured tables (memory-careful order) ──────────────────────
    log.info("=== Step 1: Load structured tables ===")

    # Open output writers FIRST so we can write-and-free large tables immediately
    export.ensure_output_dir()
    handles = export.open_writers()

    admissions_df  = loader.load_admissions()
    patients_df    = loader.load_patients()
    d_labitems     = loader.load_d_labitems()

    # Load ONLY primary diagnoses (seq_num=1, ~545K rows) — avoid holding 6.3M rows
    primary_only_df = loader.load_primary_diagnoses()
    log.info("Primary diagnoses: %d admissions with seq_num=1", len(primary_only_df))

    # Optional: limit to N admissions for testing
    all_hadm_ids: list[int] = primary_only_df["hadm_id"].unique().tolist()
    if limit and limit > 0:
        log.warning("LIMIT mode: processing only first %d admissions", limit)
        all_hadm_ids = all_hadm_ids[:limit]
    hadm_id_set = set(all_hadm_ids)

    # Build lightweight pandas Series for lookups (much less memory than Python dicts)
    p_sub = primary_only_df[primary_only_df["hadm_id"].isin(hadm_id_set)].copy()
    p_sub["hadm_id"] = p_sub["hadm_id"].astype("int64")
    p_sub = p_sub.set_index("hadm_id")
    # Deduplicate: some hadm_ids may have >1 seq_num=1 row due to data issues
    p_sub = p_sub[~p_sub.index.duplicated(keep="first")]

    # Three separate Series with efficient dtypes instead of one fat Python dict-of-dicts
    primary_icd_code    = p_sub["icd_code"].astype("category")        # ~5MB
    primary_icd_version = p_sub["icd_version"].astype("Int16")        # ~1MB
    primary_long_title  = p_sub["long_title"].fillna("").astype(pd.StringDtype())  # ~30MB
    del p_sub, primary_only_df
    gc.collect()
    log.info("  primary lookup Series built (%d entries)", len(primary_long_title))

    # Write demographics FIRST (needs admissions + patients), then aggressively free
    log.info("Building and writing patient_demographics…")
    demo_df = (
        admissions_df[admissions_df["hadm_id"].isin(hadm_id_set)]
        .merge(patients_df, on="subject_id", how="left")
    )[["hadm_id", "subject_id", "gender", "anchor_age",
       "race", "marital_status", "insurance",
       "admission_type", "admission_location", "discharge_location",
       "hospital_expire_flag"]]
    export.write_dataframe(handles, OUT_DEMOGRAPHICS, demo_df)
    del demo_df, patients_df
    gc.collect()

    # Slim admissions_df to only 3 columns needed later for radiology time filter
    admissions_slim = admissions_df[["hadm_id", "admittime", "dischtime"]].copy()
    del admissions_df
    gc.collect()
    log.info("  demographics written; admissions slimmed; patients freed")

    # Now stream ICD tables (lower memory footprint)
    log.info("Streaming and writing icd_diagnosis…")
    for chunk in loader.stream_diagnoses_icd(hadm_id_set):
        rows = (
            chunk[["hadm_id", "seq_num", "icd_code", "icd_version", "long_title"]]
            .rename(columns={"long_title": "icd_diagnosis"})
            .dropna(subset=["icd_diagnosis"])
        )
        export.write_dataframe(handles, OUT_ICD_DX, rows)
    gc.collect()
    log.info("  icd_diagnosis written")

    log.info("Streaming and writing icd_procedures…")
    for chunk in loader.stream_procedures_icd(hadm_id_set):
        rows = (
            chunk[["hadm_id", "icd_code", "long_title", "icd_version"]]
            .rename(columns={"long_title": "icd_title"})
            .dropna(subset=["icd_title"])
        )
        export.write_dataframe(handles, OUT_ICD_PROC, rows)
    gc.collect()
    log.info("  icd_procedures written")

    # ── 2. Build lab test mapping ──────────────────────────────────────────────
    log.info("=== Step 2: Build lab_test_mapping ===")
    lab_mapping_df = labs.build_lab_test_mapping(d_labitems)
    del d_labitems
    gc.collect()

    # ── 3. Stream lab events → write first-instance rows on-the-fly ──────────
    # Uses two-pass approach: Pass 1 builds compact min-charttime index (~280MB);
    # Pass 2 streams again and writes matching rows directly — never holds 20M rows.
    log.info("=== Step 3: Stream lab events (two-pass write) ===")
    hadm_has_labs: set = set()

    def _write_lab_batch(df):
        export.write_dataframe(handles, OUT_LABS, df)
        hadm_has_labs.update(df["hadm_id"].dropna().astype("int64").tolist())

    try:
        labs.build_lab_records_streaming(
            lambda: loader.stream_labevents(hadm_id_set),
            hadm_id_set,
            write_chunk_callback=_write_lab_batch,
        )
    except Exception as exc:
        import traceback
        log.error("EXCEPTION in lab streaming: %s\n%s", exc, traceback.format_exc())
        raise
    gc.collect()
    log.info("  lab events written; %d admissions have labs", len(hadm_has_labs))

    # ── 4. Stream microbiology events ─────────────────────────────────────────
    log.info("=== Step 4: Stream microbiology events ===")
    micro_df = labs.build_micro_records_streaming(
        loader.stream_microbiology(hadm_id_set),
        hadm_id_set,
    )

    # ── 5. Write lab_test_mapping + micro now (before loading 426MB radiology) ─
    log.info("Writing lab_test_mapping…")
    export.save_lab_mapping(lab_mapping_df, handles)
    del lab_mapping_df; gc.collect()

    log.info("Writing microbiology…")
    export.write_dataframe(handles, OUT_MICRO, micro_df)
    del micro_df; gc.collect()

    # ── 6. Load radiology detail (426MB) — deferred until just before use ─────
    log.info("=== Step 5: Load radiology detail ===")
    radiology_detail_wide = loader.load_radiology_detail()

    # ── 7. Load and process radiology reports ─────────────────────────────────
    log.info("=== Step 6: Load and process radiology reports ===")
    radiology_raw = loader.load_radiology(hadm_ids=hadm_id_set)
    radiology_df  = radiology.process_radiology(
        radiology_raw,
        radiology_detail_wide,
        admissions_slim,
        hadm_id_set,
        hours_before=config.RAD_HOURS_BEFORE_ADMIT,
    )
    del radiology_raw, admissions_slim, radiology_detail_wide
    gc.collect()
    log.info("  radiology done; detail/raw/admissions freed")

    log.info("Writing radiology_reports…")
    export.write_dataframe(handles, OUT_RADIOLOGY, radiology_df)
    hadm_has_rad = set(radiology_df["hadm_id"].unique()) if not radiology_df.empty else set()
    del radiology_df; gc.collect()

    # ── 8. Stream prescriptions → discharge medications ───────────────────────
    log.info("=== Step 7: Stream prescriptions ===")
    med_chunks = []
    for chunk in loader.stream_prescriptions(hadm_id_set):
        med_chunks.append(chunk)
    meds_df = pd.concat(med_chunks, ignore_index=True) if med_chunks else pd.DataFrame()

    log.info("Writing discharge_medications…")
    export.write_dataframe(handles, OUT_MEDICATIONS, meds_df)
    del meds_df; gc.collect()

    # ── 11. Stream discharge notes → HPI / PE / Dx / Proc ─────────────────────
    log.info("=== Step 11: Stream discharge notes (HPI / PE extraction) ===")

    stats = defaultdict(int)
    primary_dx_index = {}

    # ── Collect best discharge note per hadm_id (latest note_seq) ────────────
    # Multiple notes per admission may exist (original + addenda).
    # We keep only the one with the highest note_seq (most complete).
    log.info("Selecting best discharge note per hadm_id…")
    best_note: dict[int, tuple] = {}   # hadm_id → (note_seq, text)
    for chunk in pd.read_csv(
        config.DISCHARGE_FILE,
        usecols=["note_id", "subject_id", "hadm_id", "note_seq", "charttime", "text"],
        chunksize=config.CHUNK_SIZE,
    ):
        chunk = chunk[chunk["hadm_id"].isin(hadm_id_set)].copy()
        if chunk.empty:
            continue
        chunk["note_seq"] = pd.to_numeric(chunk["note_seq"], errors="coerce").fillna(0)
        for _, row in chunk.iterrows():
            hadm_id  = int(row["hadm_id"])
            note_seq = int(row["note_seq"])
            if hadm_id not in best_note or note_seq > best_note[hadm_id][0]:
                best_note[hadm_id] = (note_seq, str(row["text"]) if pd.notna(row["text"]) else "")

    log.info("  %d unique admissions with discharge notes", len(best_note))

    log.info("Streaming discharge notes (HPI / PE / Hospital Course extraction)…")
    for hadm_id, (_, text) in best_note.items():
        stats["total_notes"] += 1

        # Must have an HPI section
        if not discharge.has_hpi_section(text):
            stats["skip_no_hpi_section"] += 1
            continue

        hpi    = discharge.extract_hpi(text)
        pe     = discharge.extract_physical_examination(text)
        dd     = discharge.extract_discharge_diagnosis(text)
        dp     = discharge.extract_discharge_procedures(text)
        course = discharge.extract_hospital_course(text)

        if not hpi:
            stats["skip_empty_hpi"] += 1
            continue
        if not pe or len(pe) < min_pe_length:
            stats["skip_short_pe"] += 1
            continue
        if hadm_id not in hadm_has_labs:
            stats["skip_no_labs"] += 1
            continue
        if hadm_id not in hadm_has_rad:
            stats["skip_no_rad"] += 1
            continue

        # Look up primary diagnosis for this admission
        if hadm_id not in primary_long_title.index:
            stats["skip_no_primary_dx"] += 1
            continue
        _raw_lt = primary_long_title.at[hadm_id]
        long_title = str(_raw_lt.iloc[0] if hasattr(_raw_lt, "iloc") else _raw_lt)

        # Sanitize: if HPI mentions the primary diagnosis → transfer case → skip
        terms = _terms_for_icd(long_title)
        if terms and sanitize.hpi_is_contaminated(hpi, terms):
            stats["skip_contaminated_hpi"] += 1
            continue

        # Sanitize PE, hospital course, and discharge diagnosis text
        pe     = sanitize.sanitize_text(pe, terms)
        dd     = sanitize.sanitize_text(dd, terms)
        course = sanitize.sanitize_text(course, terms)

        # Write to CSVs
        export.write_row(handles, OUT_HPI,         {"hadm_id": hadm_id, "hpi": hpi})
        export.write_row(handles, OUT_PE,           {"hadm_id": hadm_id, "pe": pe})
        export.write_row(handles, OUT_HOSP_COURSE,  {"hadm_id": hadm_id, "hospital_course": course})
        export.write_row(handles, OUT_DISCHARGE_DX, {"hadm_id": hadm_id, "discharge_diagnosis": dd})
        for proc in dp:
            export.write_row(handles, OUT_DISCHARGE_PR, {
                "hadm_id": hadm_id,
                "discharge_procedure": proc,
            })

        # Record in primary_dx_index
        _raw_code = primary_icd_code.at[hadm_id]
        icd_code  = str(_raw_code.iloc[0] if hasattr(_raw_code, "iloc") else _raw_code)
        _raw_ver  = primary_icd_version.at[hadm_id]
        _ver_val  = _raw_ver.iloc[0] if hasattr(_raw_ver, "iloc") else _raw_ver
        icd_version = int(_ver_val) if pd.notna(_ver_val) else None
        primary_dx_index[str(hadm_id)] = {
            "icd_code":    icd_code,
            "icd_version": icd_version,
            "long_title":  long_title,
        }
        stats["accepted"] += 1

    # ── 12. Close writers ──────────────────────────────────────────────────────
    export.close_writers(handles)

    # ── 13. Save JSON indexes ──────────────────────────────────────────────────
    log.info("=== Step 13: Save JSON outputs ===")
    export.save_json(primary_dx_index, OUT_DX_INDEX)

    elapsed = time.time() - t0
    stats["elapsed_seconds"] = round(elapsed, 1)
    stats["total_in_primary_dx_index"] = len(primary_dx_index)
    export.save_json(dict(stats), OUT_STATS)

    log.info("=== DONE in %.1f s ===", elapsed)
    log.info("Accepted admissions : %d", stats["accepted"])
    log.info("Stats: %s", json.dumps(dict(stats), indent=2))
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build MIMIC-IV-Ext full dataset")
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Process only first N admissions (0 = all; useful for smoke-testing)",
    )
    parser.add_argument(
        "--min-pe-length", type=int, default=MIN_PE_LENGTH,
        help="Minimum physical examination text length to be considered valid",
    )
    args = parser.parse_args()
    run(limit=args.limit, min_pe_length=args.min_pe_length)
