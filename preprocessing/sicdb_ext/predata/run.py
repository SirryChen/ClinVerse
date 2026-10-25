"""
SICdb-Ext Full Dataset Builder
==============================
Builds a MIMIC-IV-Ext-compatible dataset from SICdb v1.0.8 CSV tables.

SICdb has no discharge-summary free text or radiology; clinical narratives are
assembled from cases, data_ref, data_range, and hourly vital signals.

Output: dataset/sicdb-ext/

Usage:
    python preprocessing/sicdb_ext/predata/run.py [--limit N] [--min-pe-length 40]
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from preprocessing.sicdb_ext.predata import clinical
from preprocessing.sicdb_ext.predata import config
from preprocessing.sicdb_ext.predata import export
from preprocessing.sicdb_ext.predata import labs
from preprocessing.sicdb_ext.predata import loader
from preprocessing.sicdb_ext.predata import references
from preprocessing.sicdb_ext.predata import sanitize

from preprocessing.sicdb_ext.predata.config import (  # noqa: E402
    OUT_HPI, OUT_PE, OUT_DISCHARGE_DX, OUT_DISCHARGE_PR,
    OUT_LABS, OUT_MICRO, OUT_ICD_DX, OUT_ICD_PROC, OUT_LAB_MAPPING,
    OUT_DEMOGRAPHICS, OUT_MEDICATIONS, OUT_HOSP_COURSE, OUT_COMORBIDITY,
    OUT_DX_INDEX, OUT_STATS,
    MIN_HPI_LENGTH, MIN_PE_LENGTH, CLINICAL_BATCH_SIZE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def run(limit: int = 0, min_pe_length: int = MIN_PE_LENGTH):
    t0 = time.time()
    export.ensure_output_dir()
    handles = export.open_writers()
    refs = references.load_references()

    log.info("=== Step 1: Load cases ===")
    cases = loader.load_cases(limit=limit)
    case_ids = set(cases["hadm_id"].astype(int).tolist())

    log.info("Writing patient_demographics…")
    export.write_dataframe(handles, OUT_DEMOGRAPHICS, loader.build_demographics(cases, refs))

    log.info("=== Step 2: lab_test_mapping ===")
    lab_mapping = labs.build_lab_test_mapping(refs)
    export.save_lab_mapping(lab_mapping, handles)

    log.info("=== Step 3: Stream laboratory_tests ===")
    cases_with_labs = labs.build_lab_records_streaming(
        loader.stream_laboratory,
        case_ids,
        refs,
        write_chunk_callback=lambda df: export.write_dataframe(handles, OUT_LABS, df),
    )
    log.info("  %d cases with labs", len(cases_with_labs))

    log.info("=== Step 4: Microbiology (none in SICdb) — skip ===")

    log.info("=== Step 5: icd_diagnosis / icd_procedures / medications ===")
    loader.write_icd_diagnoses(cases, lambda df: export.write_dataframe(handles, OUT_ICD_DX, df))
    loader.write_icd_procedures(case_ids, refs, lambda df: export.write_dataframe(handles, OUT_ICD_PROC, df))
    loader.stream_medications(case_ids, refs, lambda df: export.write_dataframe(handles, OUT_MEDICATIONS, df))

    log.info("=== Step 6: Vitals index for PE ===")
    icu_offsets = {
        int(r.hadm_id): int(r.ICUOffset) if pd.notna(r.ICUOffset) else 0
        for r in cases.itertuples()
    }
    vitals_target = case_ids & cases_with_labs if cases_with_labs else case_ids
    vitals_index = loader.build_vitals_index(vitals_target, icu_offsets)

    log.info("=== Step 7: Load data_ref / data_range ===")
    data_ref = loader.load_data_ref(case_ids)
    data_range = loader.load_data_range(case_ids)

    log.info("=== Step 8: Clinical text extraction ===")
    stats: dict = defaultdict(int)
    primary_dx_index: dict = {}

    for _, row in cases.iterrows():
        stats["total_cases"] += 1
        hadm_id = int(row["hadm_id"])
        subject_id = int(row["subject_id"])

        if hadm_id not in cases_with_labs:
            stats["skip_no_labs"] += 1
            continue

        primary = clinical.primary_from_case(row)
        if not primary:
            stats["skip_no_primary_dx"] += 1
            continue

        hpi = clinical.build_hpi(row, data_ref, refs)
        pe = clinical.build_physical_examination(
            hadm_id, vitals_index.get(hadm_id, {}), refs,
        )
        course = clinical.build_hospital_course(row, data_range, data_ref, refs)
        dd = clinical.build_discharge_diagnosis(row)
        procs = clinical.extract_discharge_procedures(hadm_id, data_range, refs)

        # Sanitize BEFORE the length checks (and before HPI specifically):
        # build_hpi() no longer states the primary diagnosis, but it is still
        # assembled from coded fields (data_ref preconditions, DiagnosisT2)
        # rather than free clinician text, so it gets the same defense-in-
        # depth masking as pe/dd/course — unlike MIMIC, where only pe/dd/
        # course need it because the HPI is genuinely free text that was
        # already screened for contamination upstream.
        terms = clinical.terms_for_diagnosis(
            primary.get("icd_code") or "", primary.get("long_title") or "",
        )
        hpi = sanitize.sanitize_text(hpi, terms)
        pe = sanitize.sanitize_text(pe, terms)
        dd = sanitize.sanitize_text(dd, terms)
        course = sanitize.sanitize_text(course, terms)

        if not hpi or len(hpi) < MIN_HPI_LENGTH:
            stats["skip_short_hpi"] += 1
            continue
        if not pe or len(pe) < min_pe_length:
            stats["skip_short_pe"] += 1
            continue

        export.write_row(handles, OUT_HPI, {"hadm_id": hadm_id, "hpi": hpi})
        export.write_row(handles, OUT_PE, {"hadm_id": hadm_id, "pe": pe})
        export.write_row(handles, OUT_HOSP_COURSE, {"hadm_id": hadm_id, "hospital_course": course})
        export.write_row(handles, OUT_DISCHARGE_DX, {"hadm_id": hadm_id, "discharge_diagnosis": dd})
        for proc in procs:
            export.write_row(handles, OUT_DISCHARGE_PR, {"hadm_id": hadm_id, "discharge_procedure": proc})

        # Structured comorbidity/precondition flags (data_ref Precondition*/
        # Premedication*/NYHA/...), the same dict already baked into hpi/
        # hospital_course as free text — exported separately so downstream
        # consumers (e.g. ClinVerse's Synthea condition-prior extraction)
        # don't have to regex-parse those sentences back apart.
        preconds = clinical.data_ref_map(data_ref, hadm_id, refs)
        for field, value in preconds.items():
            export.write_row(handles, OUT_COMORBIDITY, {
                "hadm_id": hadm_id, "field": field, "value": value,
            })

        primary_dx_index[str(hadm_id)] = {
            "hadm_id": hadm_id,
            "subject_id": subject_id,
            "icd_code": primary["icd_code"],
            "icd_version": primary["icd_version"],
            "long_title": primary["long_title"],
        }
        stats["accepted"] += 1

    export.close_writers(handles)

    log.info("=== Step 9: Save JSON ===")
    export.save_json(primary_dx_index, OUT_DX_INDEX)
    elapsed = time.time() - t0
    stats["elapsed_seconds"] = round(elapsed, 1)
    stats["total_in_primary_dx_index"] = len(primary_dx_index)
    export.save_json(dict(stats), OUT_STATS)

    log.info("=== DONE in %.1f s ===", elapsed)
    log.info("Accepted cases: %d", stats.get("accepted", 0))
    log.info("Stats: %s", json.dumps(dict(stats), indent=2))
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build SICdb-Ext dataset")
    parser.add_argument("--limit", type=int, default=0, help="Process first N cases only")
    parser.add_argument("--min-pe-length", type=int, default=MIN_PE_LENGTH)
    args = parser.parse_args()
    run(limit=args.limit, min_pe_length=args.min_pe_length)
