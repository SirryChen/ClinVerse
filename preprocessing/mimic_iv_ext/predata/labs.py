"""
Laboratory and microbiology event processing.
Mirrors paulhager CDM pipeline:
  - First instance per (hadm_id, itemid) sorted by charttime
  - valuestr = valuenum + unit, or value + unit, or flag, or comments
  - Lab-test synonym mapping for semantic matching
"""
from __future__ import annotations

import collections
import logging
import re
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)


# ── valuestr construction ──────────────────────────────────────────────────────

def _make_valuestr_lab(row: pd.Series) -> Optional[str]:
    """Create a human-readable value string from a labevents row."""
    valuenum = row.get("valuenum")
    value    = row.get("value")
    uom      = row.get("valueuom")
    flag     = row.get("flag")
    comments = row.get("comments")

    def _with_uom(v):
        if pd.notna(uom) and str(uom).strip():
            return f"{v} {str(uom).strip()}"
        return str(v)

    if pd.notna(valuenum) and str(valuenum) != "___":
        return _with_uom(valuenum)
    if pd.notna(value) and str(value) not in ("___", ""):
        return _with_uom(value)
    if pd.notna(flag) and str(flag).strip():
        return str(flag).strip()
    if pd.notna(comments) and str(comments).strip():
        return str(comments).strip()
    return None


def _make_valuestr_micro(row: pd.Series) -> Optional[str]:
    org_name = row.get("org_name")
    comments = row.get("comments")
    if pd.notna(org_name) and str(org_name).strip():
        return str(org_name).strip()
    if pd.notna(comments) and str(comments).strip():
        return str(comments).strip()
    return None


# ── Per-admission first-instance extraction ────────────────────────────────────

def extract_first_labs(
    chunks_iter,
    hadm_id: int,
) -> tuple[dict, dict, dict]:
    """
    Given an iterator of lab-events DataFrames (pre-filtered to this hadm_id),
    return (le_dict, ref_low_dict, ref_up_dict) keyed by itemid.
    Only the first (earliest charttime) instance per itemid is kept.
    """
    rows = []
    for chunk in chunks_iter:
        sub = chunk[chunk["hadm_id"] == hadm_id].copy()
        if not sub.empty:
            rows.append(sub)
    if not rows:
        return {}, {}, {}

    df = pd.concat(rows, ignore_index=True)
    df = df.sort_values("charttime", ascending=True)
    df["valuestr"] = df.apply(_make_valuestr_lab, axis=1)
    df = df.dropna(subset=["valuestr"])
    df = df[df["valuestr"] != "___"]
    df = df.drop_duplicates(subset="itemid", keep="first")

    le     = df.set_index("itemid")["valuestr"].to_dict()
    ref_lo = df.set_index("itemid")["ref_range_lower"].to_dict()
    ref_hi = df.set_index("itemid")["ref_range_upper"].to_dict()
    return le, ref_lo, ref_hi


def extract_first_microbiology(
    chunks_iter,
    hadm_id: int,
) -> dict:
    """
    Return {test_itemid: valuestr} for first instance of each microbiology test.
    Multiple organisms for the same test_itemid are merged as comma-separated.
    """
    rows = []
    for chunk in chunks_iter:
        sub = chunk[chunk["hadm_id"] == hadm_id].copy()
        if not sub.empty:
            rows.append(sub)
    if not rows:
        return {}

    df = pd.concat(rows, ignore_index=True)
    df = df.sort_values("charttime", ascending=True)

    result = {}
    for (test_itemid, _), grp in df.groupby(["test_itemid", "charttime"], sort=False):
        if test_itemid in result:
            continue  # already have first instance
        first = grp.iloc[0]
        if pd.isna(first["org_itemid"]):
            val = _make_valuestr_micro(first)
        else:
            unique_orgs = grp.dropna(subset=["org_name"])["org_name"].unique()
            val = ", ".join(str(o) for o in unique_orgs) if len(unique_orgs) else _make_valuestr_micro(first)
        if val:
            result[test_itemid] = val
    return result


# ── Bulk streaming extraction (memory-efficient for all 545K admissions) ────────

def build_lab_records_streaming(
    lab_chunks_factory,
    valid_hadm_ids: set,
    write_chunk_callback=None,
) -> pd.DataFrame | None:
    """
    Memory-efficient two-pass approach for 20M+ lab rows.

    Pass 1: Stream labevents to find minimum charttime per (hadm_id, itemid).
            Compact storage: (hadm_id int32, itemid int32, ct_ns int64) → ~280 MB.
    Pass 2: Stream again; yield first-instance rows in small batches.
            If write_chunk_callback is provided, call it for each batch
            (zero accumulation in memory). Otherwise return full DataFrame.

    write_chunk_callback(df): called with each small result DataFrame.
    """
    import gc as _gc

    log.info("Building lab records (two-pass streaming)…")

    # ── Pass 1: find minimum charttime per (hadm_id, itemid) ──────────────────
    # Batch collapse every COLLAPSE_EVERY chunks (free each raw chunk immediately).
    # Between collapses only COLLAPSE_EVERY small minis are in memory (~<100MB).
    COLLAPSE_EVERY = 20
    log.info("  Pass 1: collecting minimum charttimes (batch=%d)…", COLLAPSE_EVERY)
    batch: list[pd.DataFrame] = []
    collapsed: pd.DataFrame | None = None
    chunk_count = 0

    def _collapse(frames: list[pd.DataFrame], prev: pd.DataFrame | None) -> pd.DataFrame:
        if prev is not None:
            frames = [prev] + frames
        combined = pd.concat(frames, ignore_index=True)
        result = (
            combined.groupby(["hadm_id", "itemid"], sort=False)["ct_ns"]
            .min().reset_index()
            .astype({"hadm_id": "int32", "itemid": "int32"})
        )
        del combined; _gc.collect()
        return result

    for chunk in lab_chunks_factory():
        chunk = chunk[chunk["hadm_id"].isin(valid_hadm_ids)].copy()
        if not chunk.empty:
            chunk = chunk.dropna(subset=["charttime", "itemid", "hadm_id"])
            chunk["ct_ns"] = pd.to_datetime(chunk["charttime"]).astype("int64")
            mini = (
                chunk.groupby(["hadm_id", "itemid"], sort=False)["ct_ns"]
                .min().reset_index()
                .astype({"hadm_id": "int32", "itemid": "int32"})
            )
            batch.append(mini)
        del chunk; _gc.collect()  # free raw chunk immediately

        chunk_count += 1
        if len(batch) >= COLLAPSE_EVERY:
            collapsed = _collapse(batch, collapsed)
            batch = []
            _gc.collect()
            if chunk_count % 100 == 0:
                log.info("  Pass 1 chunk %d, unique pairs: %d", chunk_count, len(collapsed))

    # Final collapse of remaining batch
    if batch:
        collapsed = _collapse(batch, collapsed)
        batch = []
        _gc.collect()

    if collapsed is None:
        if write_chunk_callback is None:
            return pd.DataFrame(columns=["hadm_id", "itemid", "valuestr",
                                         "flag", "ref_range_lower", "ref_range_upper"])
        return None

    best = collapsed.set_index(["hadm_id", "itemid"])
    del collapsed; _gc.collect()
    log.info("  Pass 1 done: %d unique (hadm_id, itemid) pairs, "
             "tracking DataFrame %.0f MB",
             len(best), best.memory_usage(deep=True).sum() / 1e6)

    # ── Pass 2: stream again, emit first-instance rows in small batches ────────
    log.info("  Pass 2: writing first-instance rows…")
    total_rows = 0
    hadm_seen: set = set()
    # Accumulate result if no callback given (small datasets / tests only)
    result_parts: list[pd.DataFrame] = []

    for chunk in lab_chunks_factory():
        chunk = chunk[chunk["hadm_id"].isin(valid_hadm_ids)].copy()
        if chunk.empty:
            continue
        chunk = chunk.dropna(subset=["charttime", "itemid", "hadm_id"])
        chunk["charttime"] = pd.to_datetime(chunk["charttime"])
        chunk["ct_ns"] = chunk["charttime"].view("int64")

        # Join against best-time index (only keep exact-first-time rows)
        chunk["hadm_id_i32"] = chunk["hadm_id"].astype("int32")
        chunk["itemid_i32"]  = chunk["itemid"].astype("int32")
        chunk = chunk.join(
            best.rename(columns={"ct_ns": "min_ct_ns"}),
            on=["hadm_id_i32", "itemid_i32"],
            how="inner",
        )
        chunk = chunk[chunk["ct_ns"] == chunk["min_ct_ns"]]
        if chunk.empty:
            continue

        chunk["valuestr"] = chunk.apply(_make_valuestr_lab, axis=1)
        chunk = chunk.dropna(subset=["valuestr"])
        chunk = chunk[chunk["valuestr"] != "___"]
        if chunk.empty:
            continue

        first = chunk.drop_duplicates(
            subset=["hadm_id", "itemid"], keep="first"
        )[["hadm_id", "itemid", "valuestr", "flag",
           "ref_range_lower", "ref_range_upper"]].copy()
        first["hadm_id"] = first["hadm_id"].astype("Int64")
        first["itemid"]  = first["itemid"].astype("Int64")

        if first.empty:
            continue

        if write_chunk_callback is not None:
            write_chunk_callback(first)
        else:
            result_parts.append(first)
        total_rows += len(first)
        hadm_seen.update(first["hadm_id"].dropna().tolist())

    log.info("  %d lab records across %d admissions", total_rows, len(hadm_seen))

    if write_chunk_callback is not None:
        return None

    if not result_parts:
        return pd.DataFrame(columns=["hadm_id", "itemid", "valuestr",
                                     "flag", "ref_range_lower", "ref_range_upper"])

    result = pd.concat(result_parts, ignore_index=True)
    result = result.drop_duplicates(subset=["hadm_id", "itemid"], keep="first")
    return result


def build_micro_records_streaming(
    micro_chunks,
    valid_hadm_ids: set,
) -> pd.DataFrame:
    """
    Stream through microbiologyevents, keep first instance per (hadm_id, test_itemid).
    Returns DataFrame:
        hadm_id, test_itemid, test_name, spec_itemid, spec_type_desc, valuestr,
        antibiogram   (JSON-like string: "{drug: S/R/I, …}")
    """
    log.info("Building microbiology records (streaming)…")
    # {(hadm_id, test_itemid): (min_charttime, valuestr, spec_itemid, spec_type_desc, test_name)}
    best: dict[tuple, tuple] = {}
    # Collect all antibiogram rows: {(hadm_id, test_itemid): {ab_name: interpretation}}
    antibiogram: dict[tuple, dict] = {}

    for chunk in micro_chunks:
        chunk = chunk[chunk["hadm_id"].isin(valid_hadm_ids)].copy()
        if chunk.empty:
            continue
        chunk = chunk.sort_values("charttime")

        for (hadm_id, test_id, ctime), grp in chunk.groupby(
            ["hadm_id", "test_itemid", "charttime"], sort=False
        ):
            key = (int(hadm_id), int(test_id))
            if key not in best:
                first = grp.iloc[0]
                if pd.isna(first["org_itemid"]):
                    val = _make_valuestr_micro(first)
                else:
                    unique_orgs = grp.dropna(subset=["org_name"])["org_name"].unique()
                    val = ", ".join(str(o) for o in unique_orgs) if len(unique_orgs) else _make_valuestr_micro(first)
                if val:
                    spec_id   = int(first["spec_itemid"]) if pd.notna(first["spec_itemid"]) else None
                    spec_desc = str(first["spec_type_desc"]) if pd.notna(first["spec_type_desc"]) else None
                    test_name = str(first["test_name"]) if pd.notna(first["test_name"]) else None
                    best[key] = (ctime, val, spec_id, spec_desc, test_name, bool(grp["org_itemid"].notna().any()))

            # Collect antibiogram regardless of whether it's the first test instance
            ab_rows = grp.dropna(subset=["ab_name", "interpretation"])
            for _, ab_row in ab_rows.iterrows():
                ab = str(ab_row["ab_name"]).strip()
                interp = str(ab_row["interpretation"]).strip()
                if ab and interp:
                    if key not in antibiogram:
                        antibiogram[key] = {}
                    antibiogram[key][ab] = interp

    records = []
    for k, (_, vs, spec_id, spec_desc, test_name, is_abnormal) in best.items():
        ab_str = "; ".join(
            f"{ab}={interp}" for ab, interp in sorted(antibiogram.get(k, {}).items())
        )
        records.append({
            "hadm_id":       k[0],
            "test_itemid":   k[1],
            "test_name":     test_name,
            "spec_itemid":   spec_id,
            "spec_type_desc": spec_desc,
            "valuestr":      vs,
            "antibiogram":   ab_str or None,
            "is_abnormal": is_abnormal,
        })

    df = pd.DataFrame(records)
    log.info("  %d micro records across %d admissions", len(df),
             df["hadm_id"].nunique() if not df.empty else 0)
    return df


# ── Lab test synonym mapping ───────────────────────────────────────────────────

def build_lab_test_mapping(d_labitems: pd.DataFrame) -> pd.DataFrame:
    """
    Build a lab_test_mapping DataFrame:
        itemid, label, fluid, category, count, corresponding_ids
    corresponding_ids groups medically identical tests so that
    a request for "WBC" matches all equivalent itemids.
    """
    log.info("Building lab_test_mapping…")
    df = d_labitems.copy()
    df["itemid"] = df["itemid"].astype("Int64")

    # Group duplicate labels → corresponding_ids
    label_to_ids = df.groupby("label")["itemid"].apply(list).to_dict()
    df["corresponding_ids"] = df["label"].map(label_to_ids)

    # Add abbreviation variants (short/long name splits)
    extras = _expand_abbreviations(df)
    df = pd.concat([df, extras], ignore_index=True).drop_duplicates(
        subset=["itemid", "label"], keep="first"
    )

    # Add Total-prefix variants
    df = _add_total_prefix(df)

    # Propagate corresponding_ids transitively
    df = _propagate_corresponding_ids(df)

    df["count"] = df.get("count", 0)
    return df[["itemid", "label", "fluid", "category", "count", "corresponding_ids"]]


def _expand_abbreviations(df: pd.DataFrame) -> pd.DataFrame:
    """
    For labels like "Lymphocytes (%)", add rows for "Lymphocytes" and "%".
    """
    pattern = re.compile(r"^(.*?)\s*\(([^)]+)\)\s*$")
    extras = []
    for _, row in df.iterrows():
        m = pattern.match(str(row["label"]))
        if m:
            long_name  = m.group(1).strip()
            short_name = m.group(2).strip()
            for name in [long_name, short_name]:
                if name and name != row["label"]:
                    extras.append({
                        "itemid": row["itemid"],
                        "label":  name,
                        "fluid":  row.get("fluid"),
                        "category": row.get("category"),
                        "corresponding_ids": row["corresponding_ids"],
                    })
    return pd.DataFrame(extras)


def _add_total_prefix(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'Total X' variants for labels containing ', Total' or ' Total'."""
    total_mask = df["label"].str.contains("Total", na=False)
    total_entries = df[total_mask & ~df["label"].str.startswith("Total").fillna(False)].copy()
    if total_entries.empty:
        return df
    total_entries["label"] = (
        "Total " + total_entries["label"]
        .str.replace(", Total", "", regex=False)
        .str.replace(" Total", "", regex=False)
    )
    return pd.concat([df, total_entries], ignore_index=True)


def _propagate_corresponding_ids(df: pd.DataFrame) -> pd.DataFrame:
    """BFS to union all transitively linked itemid groups."""
    item_to_ids: dict = {}
    for _, row in df.iterrows():
        if pd.notna(row["itemid"]) and isinstance(row["corresponding_ids"], (list, set)):
            item_to_ids[row["itemid"]] = list(row["corresponding_ids"])

    for root in list(item_to_ids.keys()):
        visited = set()
        queue = collections.deque([root])
        while queue:
            cur = queue.popleft()
            if cur in visited:
                continue
            visited.add(cur)
            for nxt in item_to_ids.get(cur, []):
                if nxt not in visited:
                    queue.append(nxt)
        item_to_ids[root] = list(visited)

    df["corresponding_ids"] = df["itemid"].map(
        lambda x: item_to_ids.get(x, [x]) if pd.notna(x) else []
    )
    return df
