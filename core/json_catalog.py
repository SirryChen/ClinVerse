"""Generic JSON-backed lab/radiology catalogs.

MIMIC-IV-Ext's and SICdb-Ext's catalog JSON files use the identical shape
(panels: [...], items/studies: [...] with canonical_name/aliases) — only the
*content* differs (MIMIC itemid space vs SICdb LaboratoryID space, English
vs German aliases). So there is exactly one lookup implementation here,
shared by both adapters; each adapter just points it at its own JSON file.
If a future dataset's catalog needs a genuinely different lookup strategy
(e.g. a database-backed catalog instead of a static JSON file), it can
implement core.ports.LabCatalogPort directly without touching this class.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Optional

from .ports import CatalogItem, RadiologyStudy


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


# Chinese display names for common lab tests (canonical_name -> 中文名).
# Keyed by canonical_name, which both MIMIC-IV-Ext's and SICdb-Ext's
# lab_provider_catalog.json share verbatim (e.g. "white_blood_cells") even
# though the underlying item-id spaces and source languages differ — so one
# table works for every dataset's catalog-resolved items. Items only found
# via the lab_test_mapping.csv label lookup (no canonical_name) have no entry
# here and just render in their source-language label.
LAB_NAME_ZH: dict[str, str] = {
    "white_blood_cells": "白细胞计数",
    "red_blood_cells": "红细胞计数",
    "hemoglobin": "血红蛋白",
    "hematocrit": "红细胞压积",
    "platelets": "血小板计数",
    "platelet_count": "血小板计数",
    "mean_corpuscular_volume": "平均红细胞体积(MCV)",
    "sodium": "血钠",
    "potassium": "血钾",
    "chloride": "血氯",
    "bicarbonate": "碳酸氢根",
    "bun": "血尿素氮(BUN)",
    "creatinine": "肌酐",
    "glucose": "血糖",
    "calcium": "血钙",
    "magnesium": "血镁",
    "phosphate": "血磷",
    "albumin": "白蛋白",
    "total_protein": "总蛋白",
    "total_bilirubin": "总胆红素",
    "direct_bilirubin": "直接胆红素",
    "ast": "谷草转氨酶(AST)",
    "alt": "谷丙转氨酶(ALT)",
    "alkaline_phosphatase": "碱性磷酸酶(ALP)",
    "ggt": "γ-谷氨酰转肽酶(GGT)",
    "pt": "凝血酶原时间(PT)",
    "inr": "国际标准化比值(INR)",
    "ptt": "活化部分凝血活酶时间(APTT)",
    "fibrinogen": "纤维蛋白原",
    "d_dimer": "D-二聚体",
    "troponin_t": "肌钙蛋白T",
    "troponin_i": "肌钙蛋白I",
    "ck_mb": "肌酸激酶同工酶(CK-MB)",
    "bnp": "脑钠肽(BNP)",
    "procalcitonin": "降钙素原(PCT)",
    "crp": "C反应蛋白(CRP)",
    "lactate": "乳酸",
    "lipase": "脂肪酶",
    "amylase": "淀粉酶",
    "tsh": "促甲状腺激素(TSH)",
    "free_t4": "游离甲状腺素(FT4)",
    "free_t3": "游离三碘甲状腺原氨酸(FT3)",
    "urine_glucose": "尿糖",
    "urine_protein": "尿蛋白",
    "urine_blood": "尿潜血",
    "urine_wbc": "尿白细胞",
    "ph": "动脉血pH",
    "pao2": "动脉血氧分压(PaO₂)",
    "paco2": "动脉血二氧化碳分压(PaCO₂)",
    "sao2": "动脉血氧饱和度(SaO₂)",
    "base_excess": "碱剩余(BE)",
    "hba1c": "糖化血红蛋白(HbA1c)",
}


class JsonLabCatalog:
    def __init__(self, catalog_path: str | Path, mapping_csv_path: str | Path | None = None):
        with open(catalog_path, encoding="utf-8") as f:
            self._catalog = json.load(f)

        self._lookup: dict[str, CatalogItem] = {}
        self._itemid_lookup: dict[str, CatalogItem] = {}
        for item in self._catalog["items"]:
            self._lookup[item["canonical_name"]] = item
            for alias in item.get("aliases", []):
                self._lookup[_normalize(alias)] = item
            for iid in item.get("itemids", []):
                self._itemid_lookup[str(iid)] = item

        self._panel_lookup: dict[str, list[str]] = {}
        for panel in self._catalog["panels"]:
            self._panel_lookup[panel["panel_name"].lower()] = panel["items"]
            for alias in panel.get("aliases", []):
                self._panel_lookup[_normalize(alias)] = panel["items"]

        # Fallback itemid -> raw label, for itemids that exist in the source
        # hospital's lab dictionary but weren't curated into the orderable
        # catalog above (e.g. ascites/rare panels). Covers the long tail of
        # itemids that show up in real admissions but have no CatalogItem.
        self._itemid_labels: dict[str, str] = {}
        if mapping_csv_path is not None:
            mapping_csv_path = Path(mapping_csv_path)
            if mapping_csv_path.exists():
                with open(mapping_csv_path, encoding="utf-8", newline="") as f:
                    for row in csv.DictReader(f):
                        self._itemid_labels[str(row["itemid"])] = row["label"]

    def resolve(self, test_name: str) -> Optional[CatalogItem]:
        return self._lookup.get(_normalize(test_name))

    def resolve_itemid(self, itemid) -> Optional[CatalogItem]:
        """Look up a dataset-local lab item id directly (no name/alias
        matching) — used to render raw lab records (keyed by itemid) as
        human-readable evidence. Falls back to lab_test_mapping.csv's raw
        label for itemids outside the curated catalog."""
        key = str(itemid)
        item = self._itemid_lookup.get(key)
        if item is not None:
            return item
        label = self._itemid_labels.get(key)
        if label is not None:
            return {"canonical_name": "", "display_name": label, "itemids": [key]}
        return None

    def resolve_panel(self, panel_name: str) -> Optional[list[CatalogItem]]:
        names = self._panel_lookup.get(_normalize(panel_name))
        if names is None:
            return None
        items = [self._lookup[n] for n in names if n in self._lookup]
        return items or None

    def describe_for_prompt(self) -> str:
        panels = self._catalog["panels"]
        panel_lines = [
            f'  {p["panel_name"]} ({p["display_name"]}): {", ".join(p["items"])}'
            for p in panels
        ]

        by_panel: dict[str, list[str]] = {p["panel_name"]: [] for p in panels}
        seen: set[str] = set()
        for p in panels:
            for name in p["items"]:
                if name not in seen:
                    by_panel[p["panel_name"]].append(name)
                    seen.add(name)

        standalone = [
            i["canonical_name"] for i in self._catalog["items"]
            if i["canonical_name"] not in seen and i.get("source_table") != "microbiology"
        ]
        microbiology = [
            i["canonical_name"] for i in self._catalog["items"]
            if i.get("source_table") == "microbiology"
        ]

        item_lines = [
            f'  [{p["panel_name"]}] {", ".join(by_panel[p["panel_name"]])}'
            for p in panels if by_panel[p["panel_name"]]
        ]
        if standalone:
            item_lines.append(f'  [standalone] {", ".join(standalone)}')

        text = (
            "Order laboratory tests or microbiology cultures. "
            "ONLY the canonical names listed below are accepted — "
            "anything else returns 'not available in this case'.\n\n"
            "PANELS (ordering a panel name returns all its component tests):\n"
            + "\n".join(panel_lines)
            + "\n\nINDIVIDUAL TESTS (order by exact canonical_name; "
            "grouped below by their primary panel for reference):\n"
            + "\n".join(item_lines)
        )
        if microbiology:
            text += "\n\nMICROBIOLOGY: " + ", ".join(microbiology)
        return text


class JsonRadiologyCatalog:
    def __init__(self, catalog_path: str | Path):
        with open(catalog_path, encoding="utf-8") as f:
            self._catalog = json.load(f)

        self._lookup: dict[str, RadiologyStudy] = {}
        for study in self._catalog["studies"]:
            self._lookup[study["canonical_name"]] = study
            for alias in study.get("aliases", []):
                self._lookup[_normalize(alias)] = study

    def resolve(self, study_name: str) -> Optional[RadiologyStudy]:
        return self._lookup.get(_normalize(study_name))

    def describe_for_prompt(self) -> str:
        study_lines = [
            f'  {s["canonical_name"]} — {s["display_name"]}'
            for s in self._catalog["studies"]
        ]
        return (
            "Order imaging studies. "
            "ONLY the canonical names listed below are accepted — "
            "anything else returns 'not recognized'.\n\n"
            "AVAILABLE STUDIES:\n" + "\n".join(study_lines)
        )
