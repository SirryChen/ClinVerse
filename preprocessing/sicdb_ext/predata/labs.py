"""
Laboratory processing for SICdb.
First instance per (CaseID, LaboratoryID) by Offset; itemid = LaboratoryID.
"""
from __future__ import annotations

import logging

import pandas as pd

from references import load_references, ref_label, ref_unit

log = logging.getLogger(__name__)


def _make_valuestr(row: pd.Series, refs: dict) -> str | None:
    val = row.get("LaboratoryValue")
    if pd.isna(val):
        return None
    unit = ref_unit(row.get("LaboratoryID"), refs)
    s = str(val).strip()
    if unit:
        return f"{s} {unit}"
    return s


def build_lab_test_mapping(refs: dict | None = None) -> pd.DataFrame:
    refs = refs or load_references()
    rows = []
    for rid, info in refs.items():
        if info.get("name") != "Laboratory" or not info.get("value"):
            continue
        rows.append({
            "itemid": rid,
            "label": info["value"],
            "fluid": None,
            "category": "Laboratory",
            "count": 0,
            "corresponding_ids": [rid],
        })
    if not rows:
        return pd.DataFrame(columns=[
            "itemid", "label", "fluid", "category", "count", "corresponding_ids",
        ])
    df = pd.DataFrame(rows)
    label_to_ids = df.groupby("label")["itemid"].apply(list).to_dict()
    df["corresponding_ids"] = df["label"].map(label_to_ids)
    return df.drop_duplicates(subset=["itemid"], keep="first")


def build_lab_records_streaming(
    lab_chunks_factory,
    valid_case_ids: set,
    refs: dict,
    write_chunk_callback=None,
) -> set[int]:
    """Two-pass streaming; returns set of CaseIDs with at least one lab."""
    import gc as _gc

    log.info("Building lab records (two-pass streaming)…")
    collapsed: pd.DataFrame | None = None
    batch: list[pd.DataFrame] = []
    COLLAPSE_EVERY = 20

    def _collapse(frames: list[pd.DataFrame], prev: pd.DataFrame | None) -> pd.DataFrame:
        if prev is not None:
            frames = [prev] + frames
        combined = pd.concat(frames, ignore_index=True)
        result = (
            combined.groupby(["CaseID", "LaboratoryID"], sort=False)["Offset"]
            .min().reset_index()
            .astype({"CaseID": "int32", "LaboratoryID": "int32"})
        )
        del combined
        _gc.collect()
        return result

    log.info("  Pass 1: minimum Offset per (CaseID, LaboratoryID)…")
    chunk_count = 0
    for chunk in lab_chunks_factory():
        chunk = chunk[chunk["CaseID"].isin(valid_case_ids)].copy()
        if chunk.empty:
            continue
        chunk = chunk.dropna(subset=["Offset", "LaboratoryID", "CaseID"])
        chunk["Offset"] = pd.to_numeric(chunk["Offset"], errors="coerce")
        chunk = chunk.dropna(subset=["Offset"])
        mini = (
            chunk.groupby(["CaseID", "LaboratoryID"], sort=False)["Offset"]
            .min().reset_index()
            .astype({"CaseID": "int32", "LaboratoryID": "int32"})
        )
        batch.append(mini)
        del chunk
        chunk_count += 1
        if len(batch) >= COLLAPSE_EVERY:
            collapsed = _collapse(batch, collapsed)
            batch = []

    if batch:
        collapsed = _collapse(batch, collapsed)

    if collapsed is None:
        return set()

    best = collapsed.set_index(["CaseID", "LaboratoryID"])
    del collapsed
    _gc.collect()
    log.info("  Pass 1 done: %d unique (CaseID, LaboratoryID) pairs", len(best))

    log.info("  Pass 2: writing first-instance rows…")
    cases_with_labs: set[int] = set()
    for chunk in lab_chunks_factory():
        chunk = chunk[chunk["CaseID"].isin(valid_case_ids)].copy()
        if chunk.empty:
            continue
        chunk["Offset"] = pd.to_numeric(chunk["Offset"], errors="coerce")
        chunk = chunk.dropna(subset=["Offset", "LaboratoryID", "CaseID"])
        chunk["case_i32"] = chunk["CaseID"].astype("int32")
        chunk["lab_i32"] = chunk["LaboratoryID"].astype("int32")
        chunk = chunk.join(
            best.rename(columns={"Offset": "min_offset"}),
            on=["case_i32", "lab_i32"],
            how="inner",
        )
        chunk = chunk[chunk["Offset"] == chunk["min_offset"]]
        if chunk.empty:
            continue

        chunk["valuestr"] = chunk.apply(lambda r: _make_valuestr(r, refs), axis=1)
        chunk = chunk.dropna(subset=["valuestr"])
        if chunk.empty:
            continue

        first = chunk.drop_duplicates(
            subset=["CaseID", "LaboratoryID"], keep="first",
        ).copy()
        first["hadm_id"] = first["CaseID"].astype(int)
        first["itemid"] = first["LaboratoryID"].astype(int)
        first["flag"] = None
        first["ref_range_lower"] = None
        first["ref_range_upper"] = None
        out = first[["hadm_id", "itemid", "valuestr", "flag", "ref_range_lower", "ref_range_upper"]]

        cases_with_labs.update(out["hadm_id"].tolist())
        if write_chunk_callback:
            write_chunk_callback(out)

    log.info("  %d cases with labs", len(cases_with_labs))
    return cases_with_labs
