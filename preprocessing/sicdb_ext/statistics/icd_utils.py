"""ICD-10(-GM) chapter (major) and category (minor) helpers for SICdb-Ext.

SICdb (Salzburg, Austria) codes every diagnosis in ICD-10-GM — there is no
ICD-9 in this dataset (see predata/clinical.py:icd_version_from_code) — so,
unlike mimic-iv-ext/statistics/icd_utils.py, no ICD-9 chapter table is needed.
There is also no external "d_icd_diagnoses" dictionary to join against: the
(German) long_title for every code already lives in primary_diagnosis_index.json,
so the minor-category title lookup is built directly from that file.
"""
from __future__ import annotations

from functools import lru_cache

from config import load_primary_diagnosis_df

# ICD-10-CM/GM chapter prefixes (same chapter boundaries as ICD-10-CM; SICdb's
# ICD-10-GM follows the same A-Z letter/range layout for chapter purposes)
ICD10_CHAPTERS: list[tuple[str, str, str]] = [
    ("A", "B", "某些传染病和寄生虫病"),
    ("C", "D49", "肿瘤"),
    ("D50", "D89", "血液及造血器官疾病和某些涉及免疫机制的疾患"),
    ("E", "E89", "内分泌、营养和代谢疾病"),
    ("F", "F99", "精神和行为障碍"),
    ("G", "G99", "神经系统疾病"),
    ("H", "H59", "眼和附器疾病"),
    ("H60", "H95", "耳和乳突疾病"),
    ("I", "I99", "循环系统疾病"),
    ("J", "J99", "呼吸系统疾病"),
    ("K", "K95", "消化系统疾病"),
    ("L", "L99", "皮肤和皮下组织疾病"),
    ("M", "M99", "肌肉骨骼系统和结缔组织疾病"),
    ("N", "N99", "泌尿生殖系统疾病"),
    ("O", "O9A", "妊娠、分娩和产褥期"),
    ("P", "P96", "围生期起源的某些情况"),
    ("Q", "Q99", "先天性畸形、变形和染色体异常"),
    ("R", "R99", "症状、体征和异常临床与实验室所见"),
    ("S", "T98", "损伤、中毒和外因的某些其他后果"),
    ("U", "U99", "新发疾病临时分类(如 COVID-19)"),
    ("V", "Y99", "疾病和死亡的外因"),
    ("Z", "Z99", "影响健康状态和与保健机构接触的因素"),
]

# SICdb also carries a small numeric "extension catalogue" for prosthesis /
# implant complications (e.g. "102.1") that doesn't fit the letter-prefixed
# ICD-10 ranges above — bucket it separately rather than mis-sorting it.
NUMERIC_EXTENSION_CHAPTER = "假体/植入物并发症补充编码"


def _icd10_sort_key(code: str) -> tuple:
    """Sortable key for ICD-10 range comparison."""
    parts: list[tuple[int, str]] = []
    for ch in code:
        if ch.isdigit():
            parts.append((0, ch))
        else:
            parts.append((1, ch))
    return tuple(parts)


def icd10_chapter(code: str) -> str:
    code = code.strip().upper().replace(".", "")
    if not code:
        return "未知"
    if code[0].isdigit():
        return NUMERIC_EXTENSION_CHAPTER
    for start, end, name in ICD10_CHAPTERS:
        s_key = _icd10_sort_key(start)
        e_key = _icd10_sort_key(end)
        c_key = _icd10_sort_key(code)
        if s_key <= c_key <= e_key:
            return name
    return "未知"


def icd_major_chapter(code: str, version: int = 10) -> str:
    return icd10_chapter(code)


def icd_minor_key(code: str, version: int = 10) -> str:
    """Three-character category key for grouping minor classes."""
    code = code.strip().upper().replace(".", "")
    return code[:3]


@lru_cache(maxsize=1)
def load_icd_title_lookup() -> dict[tuple[int, str], str]:
    """(icd_version, icd_code) -> German long_title, sourced from the
    dataset's own primary_diagnosis_index.json (no external ICD dictionary
    exists for SICdb)."""
    df = load_primary_diagnosis_df()
    lookup: dict[tuple[int, str], str] = {}
    for _, row in df.iterrows():
        key = (int(row["icd_version"]), str(row["icd_code"]).strip().upper())
        lookup.setdefault(key, row["long_title"])
    return lookup


@lru_cache(maxsize=1)
def load_icd_minor_prefix_titles() -> dict[tuple[int, str], str]:
    """Map (version, 3-char prefix) → representative long_title."""
    lookup = load_icd_title_lookup()
    prefix_map: dict[tuple[int, str], str] = {}
    for (ver, code), title in lookup.items():
        pfx = icd_minor_key(code, ver)
        prefix_map.setdefault((ver, pfx), title)
    return prefix_map


def icd_minor_label(code: str, version: int = 10, lookup: dict | None = None) -> str:
    """Human-readable minor category: prefix + representative German title."""
    prefix = icd_minor_key(code, version)
    ver = int(version)
    prefix_titles = load_icd_minor_prefix_titles()

    exact = None
    if lookup is not None:
        exact = lookup.get((ver, code.strip().upper().replace(".", "")))
    if exact:
        return f"[{prefix}] {exact}"

    title = prefix_titles.get((ver, prefix))
    if title:
        return f"[{prefix}] {title}"
    return f"[{prefix}] (未命名)"
