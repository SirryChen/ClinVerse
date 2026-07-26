"""
CSV export helpers.
Write incremental rows to output files using csv.DictWriter so that
large datasets are never fully held in memory at once.
"""
from __future__ import annotations

import csv
import json
import logging
import os
from typing import Any, Iterator

from config import (
    BASE_OUT,
    OUT_HPI, OUT_PE, OUT_DISCHARGE_DX, OUT_DISCHARGE_PR,
    OUT_LABS, OUT_MICRO, OUT_RADIOLOGY,
    OUT_ICD_DX, OUT_ICD_PROC, OUT_LAB_MAPPING,
    OUT_DEMOGRAPHICS, OUT_MEDICATIONS, OUT_HOSP_COURSE,
    OUT_DX_INDEX, OUT_STATS,
)

log = logging.getLogger(__name__)

_SCHEMAS: dict[str, list[str]] = {
    OUT_HPI:          ["hadm_id", "hpi"],
    OUT_PE:           ["hadm_id", "pe"],
    OUT_HOSP_COURSE:  ["hadm_id", "hospital_course"],
    OUT_DISCHARGE_DX: ["hadm_id", "discharge_diagnosis"],
    OUT_DISCHARGE_PR: ["hadm_id", "discharge_procedure"],
    # labs: added flag column
    OUT_LABS:         ["hadm_id", "itemid", "valuestr", "flag",
                       "ref_range_lower", "ref_range_upper"],
    # micro: added test_name, spec_type_desc, antibiogram
    OUT_MICRO:        ["hadm_id", "test_itemid", "test_name",
                       "spec_itemid", "spec_type_desc", "valuestr", "antibiogram"],
    OUT_RADIOLOGY:    ["hadm_id", "note_id", "modality", "region", "exam_name", "text"],
    # icd_diagnosis: added seq_num, icd_code, icd_version
    OUT_ICD_DX:       ["hadm_id", "seq_num", "icd_code", "icd_version", "icd_diagnosis"],
    OUT_ICD_PROC:     ["hadm_id", "icd_code", "icd_title", "icd_version"],
    OUT_LAB_MAPPING:  ["itemid", "label", "fluid", "category", "count", "corresponding_ids"],
    # new tables
    OUT_DEMOGRAPHICS: ["hadm_id", "subject_id", "gender", "anchor_age",
                       "race", "marital_status", "insurance",
                       "admission_type", "admission_location", "discharge_location",
                       "hospital_expire_flag"],
    OUT_MEDICATIONS:  ["hadm_id", "drug", "prod_strength",
                       "dose_val_rx", "dose_unit_rx", "route", "doses_per_24_hrs"],
}


def ensure_output_dir():
    os.makedirs(BASE_OUT, exist_ok=True)
    log.info("Output directory: %s", BASE_OUT)


def open_writers() -> dict[str, tuple]:
    """Open all output CSV files and return {path: (file_handle, writer)}."""
    handles = {}
    for path, cols in _SCHEMAS.items():
        fh = open(path, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        handles[path] = (fh, writer)
    return handles


def close_writers(handles: dict):
    for path, (fh, _) in handles.items():
        fh.close()
    log.info("All CSV files closed.")


def write_row(handles: dict, path: str, row: dict):
    _, writer = handles[path]
    writer.writerow(row)


def write_rows(handles: dict, path: str, rows: list[dict]):
    _, writer = handles[path]
    writer.writerows(rows)


def write_dataframe(handles: dict, path: str, df, chunk_size: int = 10_000):
    """Write a pandas DataFrame to an already-open CSV writer, chunked to avoid large
    temporary Python dicts from to_dict('records') on very large DataFrames."""
    import pandas as pd
    if df is None or (hasattr(df, "empty") and df.empty):
        return
    cols = _SCHEMAS[path]
    for col in cols:
        if col not in df.columns:
            df[col] = None
    _, writer = handles[path]
    sub = df[cols]
    for start in range(0, len(sub), chunk_size):
        writer.writerows(sub.iloc[start : start + chunk_size].to_dict("records"))


def save_json(obj: Any, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    log.info("Saved JSON: %s", path)


def save_lab_mapping(df, handles: dict):
    """Special-case: corresponding_ids is a list — stringify it."""
    import pandas as pd
    if df is None or df.empty:
        return
    _, writer = handles[OUT_LAB_MAPPING]
    cols = _SCHEMAS[OUT_LAB_MAPPING]
    for _, row in df.iterrows():
        d = {c: row.get(c) for c in cols}
        ids = d.get("corresponding_ids", [])
        if isinstance(ids, (list, set)):
            d["corresponding_ids"] = str(sorted(int(i) for i in ids if pd.notna(i)))
        writer.writerow(d)
