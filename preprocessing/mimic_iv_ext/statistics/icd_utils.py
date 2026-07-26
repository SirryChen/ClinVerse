"""ICD-9 / ICD-10 chapter (major) and category (minor) helpers."""
from __future__ import annotations

import gzip
from functools import lru_cache

import pandas as pd

from config import D_ICD_DX

# ICD-9-CM chapter ranges (inclusive numeric prefix)
ICD9_CHAPTERS: list[tuple[int, int, str]] = [
    (1, 139, "感染性和寄生虫病"),
    (140, 239, "肿瘤"),
    (240, 279, "内分泌、营养和代谢疾病"),
    (280, 289, "血液及造血器官疾病"),
    (290, 319, "精神疾患"),
    (320, 389, "神经系统和感觉器官疾病"),
    (390, 459, "循环系统疾病"),
    (460, 519, "呼吸系统疾病"),
    (520, 579, "消化系统疾病"),
    (580, 629, "泌尿生殖系统疾病"),
    (630, 679, "妊娠、分娩和产褥期"),
    (680, 709, "皮肤及皮下组织疾病"),
    (710, 739, "肌肉骨骼系统和结缔组织疾病"),
    (740, 759, "先天性畸形"),
    (760, 779, "围生期起源的某些情况"),
    (780, 799, "症状、体征和异常临床所见"),
    (800, 999, "损伤和中毒"),
    (0, 0, "补充编码(V/E)"),  # handled separately
]

ICD9_V_RANGE = (range(1, 100), "影响健康状态和与保健机构接触的因素(V码)")
ICD9_E_RANGE = (range(800, 1000), "外因(E码)")

# ICD-10-CM chapter prefixes
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
    ("S", "T88", "损伤、中毒和外因的某些其他后果"),
    ("V", "Y99", "疾病和死亡的外因"),
    ("Z", "Z99", "影响健康状态和与保健机构接触的因素"),
]


def _icd9_numeric(code: str) -> int | None:
    digits = "".join(c for c in code if c.isdigit())
    if not digits:
        return None
    return int(digits[:3]) if len(digits) >= 3 else int(digits)


def icd9_chapter(code: str) -> str:
    code = code.strip().upper()
    if code.startswith("V"):
        return ICD9_V_RANGE[1]
    if code.startswith("E"):
        return ICD9_E_RANGE[1]
    num = _icd9_numeric(code)
    if num is None:
        return "未知"
    for lo, hi, name in ICD9_CHAPTERS[:-1]:
        if lo <= num <= hi:
            return name
    return "补充编码(V/E)"


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
    for start, end, name in ICD10_CHAPTERS:
        s_key = _icd10_sort_key(start)
        e_key = _icd10_sort_key(end)
        c_key = _icd10_sort_key(code)
        if s_key <= c_key <= e_key:
            return name
    return "未知"


def icd_major_chapter(code: str, version: int) -> str:
    if version == 9:
        return icd9_chapter(code)
    return icd10_chapter(code)


def icd_minor_key(code: str, version: int) -> str:
    """Three-character category key for grouping minor classes."""
    code = code.strip().upper().replace(".", "")
    if version == 9:
        digits = "".join(c for c in code if c.isdigit())
        return digits[:3].ljust(3, "0") if digits else code[:3]
    return code[:3]


@lru_cache(maxsize=1)
def load_icd_title_lookup() -> dict[tuple[int, str], str]:
    lookup: dict[tuple[int, str], str] = {}
    with gzip.open(D_ICD_DX, "rt") as f:
        df = pd.read_csv(f, dtype={"icd_code": str, "icd_version": int, "long_title": str})
    for _, row in df.iterrows():
        key = (int(row["icd_version"]), str(row["icd_code"]).strip().upper())
        lookup[key] = row["long_title"]
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


def icd_minor_label(code: str, version: int, lookup: dict | None = None) -> str:
    """Human-readable minor category: prefix + representative title."""
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
