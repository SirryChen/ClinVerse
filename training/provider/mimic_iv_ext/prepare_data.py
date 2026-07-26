"""
Build JSONL training data from MIMIC-IV-Ext CSVs.

Trains a lab-provider agent from an omniscient perspective:
  - label  : numeric / qualitative test result only (no unit, no [ABNORMAL] tag)
  - prompt : full clinical context + target test reference range & unit
  - train  : all abnormal labs + a configurable fraction of normal labs

Usage:
    python prepare_data.py [--config config.yaml] [--max-patients N]
                           [--train-ratio R] [--test-ratio R]

Output:
    output/train.jsonl
    output/test.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_NUMERIC_VALUE_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$")
_VALUE_UNIT_RE = re.compile(r"^([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+(.+)$")
_ABNORMAL_FLAGS = frozenset({
    "abnormal", "critical", "delta", "high", "low", "h", "l", "hh", "ll",
})

SYSTEM_PROMPT = (
    "You are a clinical laboratory assistant with complete access to a patient's medical record. "
    "Given the patient's full clinical information and the specified laboratory test, "
    "predict the final test result. "
    "Respond with ONLY the numeric result (number only). Do not include units, labels, or explanations."
)


# ── Formatting helpers ─────────────────────────────────────────────────────────

def _is_missing(val) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    return str(val).strip() in ("", "nan", "NaN", "NaT", "None")


def _fmt_flag(flag_val) -> str:
    if _is_missing(flag_val):
        return ""
    return f" [{str(flag_val).upper()}]"


def _is_abnormal(flag_val) -> bool:
    if _is_missing(flag_val):
        return False
    return str(flag_val).strip().lower() in _ABNORMAL_FLAGS


def _extract_result_and_unit(valuestr: str) -> tuple[str, str]:
    """Split '12.5 mg/dL' → ('12.5', 'mg/dL'); categorical values keep unit empty."""
    text = str(valuestr).strip()
    match = _VALUE_UNIT_RE.match(text)
    if match:
        return match.group(1), match.group(2).strip()
    return text, ""


def _is_numeric_value(value: str) -> bool:
    return bool(_NUMERIC_VALUE_RE.match(str(value).strip()))


def _fmt_ref_range(lower, upper, unit: str = "") -> str:
    suffix = f" {unit}" if unit else ""
    has_lo = not _is_missing(lower)
    has_hi = not _is_missing(upper)
    if has_lo and has_hi:
        return f"{lower} - {upper}{suffix}"
    if has_lo:
        return f">= {lower}{suffix}"
    if has_hi:
        return f"<= {upper}{suffix}"
    return "Not available"


def _build_item_unit_map(labs_df: pd.DataFrame, valid_itemids: set[str]) -> dict[str, str]:
    """Most common unit per itemid, inferred from valuestr patterns like '12.5 mg/dL'."""
    from collections import Counter

    units: dict[str, str] = {}
    subset = labs_df[labs_df["itemid"].isin(valid_itemids)]
    for itemid, grp in subset.groupby("itemid"):
        counter: Counter[str] = Counter()
        for valuestr in grp["valuestr"].dropna():
            _, unit = _extract_result_and_unit(str(valuestr))
            if unit:
                counter[unit] += 1
        if counter:
            units[str(itemid)] = counter.most_common(1)[0][0]
    return units


def _select_train_targets(
    hadm_ids: list[str],
    labs_grp,
    id_map: dict[str, str],
    min_ctx: int,
    demo_idx: pd.DataFrame,
    normal_sample_ratio: float,
    seed: int,
) -> set[tuple[str, str]]:
    """Keep all abnormal (hadm_id, itemid) pairs; sample normal pairs at given ratio."""
    abnormal_keys: set[tuple[str, str]] = set()
    normal_pool: list[tuple[str, str]] = []

    for hadm_id in hadm_ids:
        if hadm_id not in demo_idx.index:
            continue
        try:
            patient_labs = labs_grp.get_group(hadm_id).to_dict("records")
        except KeyError:
            continue

        mapped = [
            r for r in patient_labs
            if r["itemid"] in id_map and not _is_missing(r.get("valuestr"))
        ]
        if len(mapped) < min_ctx + 1:
            continue

        for r in mapped:
            key = (hadm_id, r["itemid"])
            if _is_abnormal(r.get("flag")):
                abnormal_keys.add(key)
            else:
                normal_pool.append(key)

    n_normal_keep = round(len(normal_pool) * normal_sample_ratio)
    rng = random.Random(seed)
    sampled_normals = (
        set(rng.sample(normal_pool, min(n_normal_keep, len(normal_pool))))
        if normal_pool and n_normal_keep > 0 else set()
    )
    selected = abnormal_keys | sampled_normals

    log.info(
        "Train label filter — abnormal: %d | normal kept: %d / %d (%.1f%%) | total: %d",
        len(abnormal_keys),
        len(sampled_normals),
        len(normal_pool),
        100 * normal_sample_ratio,
        len(selected),
    )
    return selected


def _truncate(text: str, max_chars: int) -> str:
    if _is_missing(text):
        return ""
    text = str(text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " [...]"


# ── Prompt builder ─────────────────────────────────────────────────────────────

def _build_user_prompt(
    demo: dict,
    hpi: str,
    pe: str,
    diagnoses: list[dict],
    medications: list[dict],
    microbiology: list[dict],
    hospital_course: str,
    discharge_diagnosis: str,
    context_labs: list[dict],
    target_label: str,
    target_category: str,
    ref_range_lower,
    ref_range_upper,
    unit: str,
    cfg_data: dict,
) -> str:
    lines: list[str] = []

    # ── 1. Target（置顶）─────────────────────────────────────────────────────
    lines.append("## Lab Test to Predict")
    lines.append(f"Name: {target_label}")
    if not _is_missing(target_category):
        lines.append(f"Category: {target_category}")
    lines.append(f"Reference range: {_fmt_ref_range(ref_range_lower, ref_range_upper, unit)}")
    if unit:
        lines.append(f"Unit: {unit}")
    lines.append(
        "Output requirement: respond with ONLY the final numeric test result "
        "(number only; no unit, no extra text)."
    )

    # ── 2. Demographics ───────────────────────────────────────────────────────
    lines.append("\n## Patient Profile")
    lines.append(f"- Age: {demo.get('anchor_age', '?')} | Sex: {demo.get('gender', '?')} | Race: {demo.get('race', '?')}")
    lines.append(f"- Admission type: {demo.get('admission_type', '?')} | Insurance: {demo.get('insurance', '?')}")
    adm_loc = demo.get("admission_location", "")
    dis_loc = demo.get("discharge_location", "")
    if not _is_missing(adm_loc):
        lines.append(f"- Admitted from: {adm_loc}")
    if not _is_missing(dis_loc):
        lines.append(f"- Discharge destination: {dis_loc}")
    if demo.get("hospital_expire_flag") == 1:
        lines.append("- Hospital expiry: YES")

    # ── 3. Discharge Diagnosis ────────────────────────────────────────────────
    dd_text = _truncate(discharge_diagnosis, 500)
    if dd_text:
        lines.append("\n## Discharge Diagnosis")
        lines.append(dd_text)

    # ── 4. ICD Diagnoses ──────────────────────────────────────────────────────
    valid_dx = [d for d in diagnoses if not _is_missing(d.get("icd_diagnosis"))]
    if valid_dx:
        lines.append("\n## Diagnoses / Comorbidities")
        for dx in valid_dx:
            seq    = dx.get("seq_num", "")
            title  = dx["icd_diagnosis"]
            code   = dx.get("icd_code", "")
            ver    = dx.get("icd_version", "")
            prefix = f"[#{seq}] " if not _is_missing(seq) else ""
            suffix = f" (ICD-{ver}: {code})" if not _is_missing(code) else ""
            lines.append(f"- {prefix}{title}{suffix}")

    # ── 5. History of Present Illness ─────────────────────────────────────────
    # 限制放大到 p90（3500 chars）；剩余截断由 tokenizer max_length 兜底
    hpi_text = _truncate(hpi, cfg_data.get("hpi_max_chars", 3500))
    if hpi_text:
        lines.append("\n## History of Present Illness")
        lines.append(hpi_text)

    # ── 6. Physical Examination ───────────────────────────────────────────────
    pe_text = _truncate(pe, cfg_data.get("pe_max_chars", 1300))
    if pe_text:
        lines.append("\n## Physical Examination")
        lines.append(pe_text)

    # ── 7. Hospital Course ────────────────────────────────────────────────────
    hc_text = _truncate(hospital_course, cfg_data.get("hospital_course_max_chars", 5000))
    if hc_text:
        lines.append("\n## Hospital Course")
        lines.append(hc_text)

    # ── 8. Medications ────────────────────────────────────────────────────────
    valid_meds = [m for m in medications if not _is_missing(m.get("drug"))]
    if valid_meds:
        lines.append("\n## Medications")
        seen: set[str] = set()
        for med in valid_meds:
            drug     = med["drug"]
            strength = med.get("prod_strength", "")
            route    = med.get("route", "")
            key = f"{drug}|{strength}|{route}"
            if key in seen:
                continue
            seen.add(key)
            parts = [drug]
            if not _is_missing(strength):
                parts.append(str(strength))
            if not _is_missing(route):
                parts.append(f"({route})")
            lines.append("- " + " ".join(parts))

    # ── 9. Microbiology ───────────────────────────────────────────────────────
    valid_micro = [m for m in microbiology if not _is_missing(m.get("valuestr"))]
    if valid_micro:
        lines.append("\n## Microbiology")
        for mic in valid_micro:
            test      = mic.get("test_name", "")
            spec      = mic.get("spec_type_desc", "")
            result    = mic.get("valuestr", "")
            abx       = mic.get("antibiogram", "")
            label_str = f"{test} ({spec})" if not _is_missing(spec) else test
            line = f"- {label_str}: {result}"
            if not _is_missing(abx):
                line += f" | Antibiogram: {abx}"
            lines.append(line)

    # ── 10. Available Laboratory Results（最后）───────────────────────────────
    # 置于末尾：若序列超长被截断，丢失的是末尾的部分 lab context，不影响目标
    if context_labs:
        lines.append("\n## Available Laboratory Results")
        for lab in context_labs:
            flag = _fmt_flag(lab.get("flag"))
            lines.append(f"- {lab['label']}: {lab['valuestr']}{flag}")

    return "\n".join(lines)


def _build_label(valuestr: str) -> str:
    """Label is the result value only (numeric when possible), without unit or flag."""
    result, _ = _extract_result_and_unit(valuestr)
    return result


# ── Data loading ───────────────────────────────────────────────────────────────

def load_tables(cfg: dict) -> dict:
    d    = cfg["data"]
    base = Path(d["mimic_data_dir"])

    def _read(key: str, filename: str, **kwargs) -> pd.DataFrame:
        path = Path(d.get(key, base / filename))
        log.info("Loading %s …", path.name)
        dtype = {"hadm_id": str}
        if "dtype" in kwargs:
            dtype = {**dtype, **kwargs.pop("dtype")}
        return pd.read_csv(path, dtype=dtype, **kwargs)

    demo  = _read("demographics_path",      "patient_demographics.csv")
    hpi   = _read("hpi_path",               "history_of_present_illness.csv").set_index("hadm_id")
    pe    = _read("pe_path",                "physical_examination.csv").set_index("hadm_id")
    hc    = _read("hospital_course_path",   "hospital_course.csv").set_index("hadm_id")
    dd    = _read("discharge_diagnosis_path","discharge_diagnosis.csv").set_index("hadm_id")
    labs  = _read("labs_path",              "laboratory_tests.csv",
                  dtype={"hadm_id": str, "itemid": str})
    micro = _read("microbiology_path",      "microbiology.csv")
    meds  = _read("medications_path",       "discharge_medications.csv")
    icd   = _read("icd_diagnosis_path",     "icd_diagnosis.csv")

    log.info("Labs: %d rows / %d admissions", len(labs), labs["hadm_id"].nunique())

    return {
        "demo":  demo,
        "hpi":   hpi,
        "pe":    pe,
        "hc":    hc,
        "dd":    dd,
        "labs":  labs,
        "micro": micro.groupby("hadm_id"),
        "meds":  meds.groupby("hadm_id"),
        "icd":   icd.groupby("hadm_id"),
    }


def _get_group(grouped, hadm_id: str) -> list[dict]:
    try:
        return grouped.get_group(hadm_id).to_dict("records")
    except KeyError:
        return []


def _get_text(df: pd.DataFrame, hadm_id: str, col: str) -> str:
    if hadm_id in df.index:
        val = df.loc[hadm_id, col]
        return "" if _is_missing(val) else str(val)
    return ""


# ── Example builder ────────────────────────────────────────────────────────────

def build_examples(
    cfg: dict,
    tables: dict,
    id_map: dict[str, str],
    item_category: dict[str, str],
    item_unit_map: dict[str, str],
    hadm_ids: list[str],
    desc: str = "Building examples",
    apply_train_filter: bool = False,
) -> list[dict]:
    """Build training examples for the given list of hadm_ids."""
    cfg_data = cfg["data"]
    min_ctx  = cfg_data.get("min_context_labs", 2)

    labs_df  = tables["labs"]
    demo_idx = tables["demo"].set_index("hadm_id")

    labs_grp = labs_df[labs_df["hadm_id"].isin(set(hadm_ids))].groupby("hadm_id")
    examples: list[dict] = []

    selected_keys: set[tuple[str, str]] | None = None
    if apply_train_filter:
        selected_keys = _select_train_targets(
            hadm_ids=hadm_ids,
            labs_grp=labs_grp,
            id_map=id_map,
            min_ctx=min_ctx,
            demo_idx=demo_idx,
            normal_sample_ratio=cfg_data.get("normal_sample_ratio", 0.10),
            seed=cfg_data.get("seed", 42),
        )

    for hadm_id in tqdm(hadm_ids, desc=desc):
        try:
            patient_labs = labs_grp.get_group(hadm_id).to_dict("records")
        except KeyError:
            continue

        mapped = [
            r for r in patient_labs
            if r["itemid"] in id_map and not _is_missing(r.get("valuestr"))
        ]
        if len(mapped) < min_ctx + 1:
            continue

        if hadm_id not in demo_idx.index:
            continue

        # ── Static context (shared across all examples for this admission) ──
        demo_row = demo_idx.loc[hadm_id].to_dict()
        hpi_text = _get_text(tables["hpi"], hadm_id, "hpi")
        pe_text  = _get_text(tables["pe"],  hadm_id, "pe")
        hc_text  = _get_text(tables["hc"],  hadm_id, "hospital_course")
        dd_text  = _get_text(tables["dd"],  hadm_id, "discharge_diagnosis")

        diagnoses    = _get_group(tables["icd"],   hadm_id)
        medications  = _get_group(tables["meds"],  hadm_id)
        microbiology = _get_group(tables["micro"], hadm_id)

        # ── One example per lab (leave-one-out) ────────────────────────────
        for i, target_lab in enumerate(mapped):
            if selected_keys is not None:
                if (hadm_id, target_lab["itemid"]) not in selected_keys:
                    continue

            context_labs = [
                {
                    "label":    id_map[r["itemid"]],
                    "valuestr": str(r["valuestr"]),
                    "flag":     r.get("flag"),
                }
                for j, r in enumerate(mapped) if j != i
            ]

            valuestr = str(target_lab["valuestr"])
            _, unit_from_value = _extract_result_and_unit(valuestr)
            unit = unit_from_value or item_unit_map.get(target_lab["itemid"], "")

            user_prompt = _build_user_prompt(
                demo=demo_row,
                hpi=hpi_text,
                pe=pe_text,
                diagnoses=diagnoses,
                medications=medications,
                microbiology=microbiology,
                hospital_course=hc_text,
                discharge_diagnosis=dd_text,
                context_labs=context_labs,
                target_label=id_map[target_lab["itemid"]],
                target_category=item_category.get(target_lab["itemid"], ""),
                ref_range_lower=target_lab.get("ref_range_lower"),
                ref_range_upper=target_lab.get("ref_range_upper"),
                unit=unit,
                cfg_data=cfg_data,
            )

            examples.append({
                "messages": [
                    {"role": "system",    "content": SYSTEM_PROMPT},
                    {"role": "user",      "content": user_prompt},
                    {"role": "assistant", "content": _build_label(valuestr)},
                ],
                # metadata for evaluation (not used in training)
                "_meta": {
                    "hadm_id":        hadm_id,
                    "itemid":         target_lab["itemid"],
                    "test_name":      id_map[target_lab["itemid"]],
                    "category":       item_category.get(target_lab["itemid"], ""),
                    "valuestr":       valuestr,
                    "label":          _build_label(valuestr),
                    "unit":           unit,
                    "is_abnormal":    _is_abnormal(target_lab.get("flag")),
                    "flag":           "" if _is_missing(target_lab.get("flag")) else str(target_lab["flag"]),
                    "ref_range_lower": target_lab.get("ref_range_lower", ""),
                    "ref_range_upper": target_lab.get("ref_range_upper", ""),
                    "is_numeric":     _is_numeric_value(_build_label(valuestr)),
                },
            })

    return examples


# ── Hadm-level split ───────────────────────────────────────────────────────────

def split_hadm_ids(
    labs_df: pd.DataFrame,
    train_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """
    Split admissions into train / test at the hadm_id level.

    Both ratios are fractions of the full dataset (0–1).  The two splits are
    disjoint; any remaining admissions are discarded when
    train_ratio + test_ratio < 1.

    Splitting by hadm_id ensures no admission appears in both splits,
    preventing context leakage across examples.
    """
    if train_ratio < 0 or test_ratio < 0:
        raise ValueError(f"ratios must be non-negative, got train={train_ratio}, test={test_ratio}")
    if train_ratio + test_ratio > 1.0:
        raise ValueError(
            f"train_ratio + test_ratio must be <= 1.0, got {train_ratio + test_ratio:.4f}"
        )
    if train_ratio == 0 and test_ratio == 0:
        raise ValueError("at least one of train_ratio / test_ratio must be > 0")

    all_hadm = labs_df["hadm_id"].unique().tolist()
    rng = random.Random(seed)
    rng.shuffle(all_hadm)

    n_total = len(all_hadm)
    n_test  = round(n_total * test_ratio)  if test_ratio  > 0 else 0
    n_train = round(n_total * train_ratio) if train_ratio > 0 else 0

    if n_test + n_train > n_total:
        n_train = n_total - n_test

    test_ids  = all_hadm[:n_test]
    train_ids = all_hadm[n_test:n_test + n_train]
    n_unused  = n_total - n_test - n_train

    log.info(
        "hadm_id split — train: %d (%.1f%%) | test: %d (%.1f%%) | unused: %d (%.1f%%) "
        "| total: %d | config train_ratio=%.2f test_ratio=%.2f",
        len(train_ids), 100 * len(train_ids) / n_total if n_total else 0,
        len(test_ids),  100 * len(test_ids)  / n_total if n_total else 0,
        n_unused,       100 * n_unused       / n_total if n_total else 0,
        n_total, train_ratio, test_ratio,
    )
    return train_ids, test_ids


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build MIMIC-IV-Ext lab-provider training data")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=None,
                        help="训练集占全量 hadm_id 的比例 (0–1)，覆盖 config 中的 train_ratio")
    parser.add_argument("--test-ratio", type=float, default=None,
                        help="测试集占全量 hadm_id 的比例 (0–1)，覆盖 config 中的 test_ratio")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.max_patients is not None:
        cfg["data"]["max_patients"] = args.max_patients
    if args.train_ratio is not None:
        cfg["data"]["train_ratio"] = args.train_ratio
    if args.test_ratio is not None:
        cfg["data"]["test_ratio"] = args.test_ratio

    Path(cfg["data"]["output_dir"]).mkdir(parents=True, exist_ok=True)

    mapping_df    = pd.read_csv(cfg["data"]["lab_mapping_path"], dtype={"itemid": str})
    id_map        = dict(zip(mapping_df["itemid"].astype(str), mapping_df["label"]))
    item_category = dict(zip(mapping_df["itemid"].astype(str), mapping_df["category"]))
    log.info("Loaded %d itemid mappings", len(id_map))

    tables = load_tables(cfg)
    d      = cfg["data"]

    item_unit_map = _build_item_unit_map(
        tables["labs"],
        set(id_map.keys()),
    )
    log.info("Inferred units for %d / %d itemids", len(item_unit_map), len(id_map))

    # ── Split by hadm_id (disjoint train/test, no admission leaks) ─────────
    test_ratio  = d.get("test_ratio", 0.10)
    train_ratio = d.get("train_ratio", 1.0 - test_ratio)

    train_ids, test_ids = split_hadm_ids(
        labs_df     = tables["labs"],
        train_ratio = train_ratio,
        test_ratio  = test_ratio,
        seed        = d.get("seed", 42),
    )

    def _build_and_write(
        hadm_ids: list[str],
        path: str,
        desc: str,
        apply_train_filter: bool,
    ) -> None:
        examples = build_examples(
            cfg, tables, id_map, item_category, item_unit_map, hadm_ids,
            desc=desc,
            apply_train_filter=apply_train_filter,
        )
        with open(path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        log.info("Wrote %d examples → %s", len(examples), path)

    _build_and_write(train_ids, d["train_file"], "Train", apply_train_filter=True)
    _build_and_write(test_ids,  d["test_file"],  "Test",  apply_train_filter=False)


if __name__ == "__main__":
    main()
