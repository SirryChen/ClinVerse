"""
Lab Provider Catalog Builder for SICdb-Ext
============================================
Reads dataset/SICdb/d_references.csv.gz (ReferenceName == "Laboratory") and
produces:

  dataset/sicdb-ext/lab_provider_catalog.json

Mirrors the structure of data/mimic-iv-ext/predata/build_catalog.py's lab
catalog (same JSON schema, consumed the same way by
adapters/*/lab_catalog.py), but `itemid` here is SICdb's `LaboratoryID`
(== d_references.ReferenceGlobalID for Laboratory entries) instead of MIMIC's
d_labitems.itemid — a completely different, non-overlapping ID space, which
is exactly why this needs its own catalog rather than reusing MIMIC's.

No radiology_catalog.json: SICdb has no radiology data at all (see
predata/README.md), so adapters/sicdb_ext declares has_radiology = False
instead of shipping an catalog that would always return "study not found".

Curated items below cover the high-yield labs MIMIC's catalog covers (CBC,
BMP, LFT, coagulation, cardiac markers, ABG, inflammatory, lipids, thyroid).
SICdb has ~539 distinct "Laboratory" reference entries in total; items not
curated here are simply not orderable by canonical name yet — extend
ITEM_DEFS as needed, the build is idempotent.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


import json
import os

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
REFERENCES_FILE = os.path.join(os.environ.get("CLINVERSE_SICDB_DIR", os.path.join(_REPO_ROOT, "dataset", "SICdb")), "d_references.csv.gz")
DATASET_DIR = os.environ.get("CLINVERSE_SICDB_NORMALIZED_DIR", os.path.join(_REPO_ROOT, "data", "normalized", "sicdb-ext"))
OUT_LAB = os.path.join(DATASET_DIR, "lab_provider_catalog.json")


# ---------------------------------------------------------------------------
# Item definitions: canonical_name -> (display_name, [ReferenceGlobalIDs], extra_aliases)
# ReferenceGlobalID list lets several SICdb variants (e.g. blood-gas vs. lab
# chemistry sodium) resolve to the same canonical test, same pattern MIMIC's
# catalog uses for itemids.
# ---------------------------------------------------------------------------
ITEM_DEFS: dict[str, tuple[str, list[int], list[str]]] = {
    # ── CBC ──────────────────────────────────────────────────────────────
    "white_blood_cells": ("White Blood Cells", [301], ["WBC", "Leukozyten", "Leukocyten", "leukocyte count"]),
    "red_blood_cells":   ("Red Blood Cells", [599], ["RBC", "Erythrozyten", "Erythrocyten", "red cells"]),
    "hemoglobin":        ("Hemoglobin", [289, 288, 658], ["Hb", "Hämoglobin", "Haemoglobin"]),
    "hematocrit":        ("Hematocrit", [217, 183, 682], ["Hct", "Hämatokrit", "Haematocrit"]),
    "platelets":         ("Platelets", [314, 315], ["PLT", "Thrombozyten", "Thrombocyten", "platelet count"]),
    "hba1c":             ("Hemoglobin A1c", [214, 474], ["HbA1c", "glycated hemoglobin"]),

    # ── Electrolytes / BMP ───────────────────────────────────────────────
    "sodium":      ("Sodium", [469, 455, 686], ["Na", "Natrium"]),
    "potassium":   ("Potassium", [463, 453, 685], ["K", "Kalium"]),
    "chloride":    ("Chloride", [450, 683], ["Cl", "Chlorid"]),
    "bicarbonate": ("Bicarbonate", [451], ["HCO3", "Bicarbonat", "Akt. Bicarbonat"]),
    "calcium":     ("Calcium", [457, 655], ["Ca"]),
    "magnesium":   ("Magnesium", [468], ["Mg"]),
    "glucose":     ("Glucose", [348, 656], ["Glukose", "blood sugar"]),
    "bun":         ("Blood Urea Nitrogen", [355], ["BUN", "Harnstoff", "Urea"]),
    "creatinine":  ("Creatinine", [367, 368], ["Cr", "Kreatinin"]),

    # ── Liver function ───────────────────────────────────────────────────
    "ast":                  ("AST", [616], ["GOT", "SGOT", "Aspartat-Aminotransferase"]),
    "alt":                  ("ALT", [617], ["GPT", "ALAT", "SGPT", "Alanin-Aminotransferase"]),
    "alkaline_phosphatase": ("Alkaline Phosphatase", [609], ["ALP", "Alkalische Phosphatase", "alk phos"]),
    "ggt":                  ("GGT", [614], ["Gamma-GT", "γ-GT", "Gamma-Glutamyltransferase"]),
    "total_bilirubin":      ("Total Bilirubin", [333], ["Bilirubin total", "Gesamtbilirubin", "Bilirubin (gesamt)"]),
    "direct_bilirubin":     ("Direct Bilirubin", [332], ["Bilirubin direkt", "Bilirubin (direkt)"]),
    "albumin":              ("Albumin", [171], ["Albumin"]),

    # ── Coagulation ───────────────────────────────────────────────────────
    "pt_inr":     ("PT / INR", [3128], ["INR", "PT", "Prothrombin time", "Quick"]),
    "ptt":        ("aPTT", [597], ["PTT", "APTT", "activated partial thromboplastin time"]),
    "fibrinogen": ("Fibrinogen", [344, 345], ["Fibrinogen"]),
    "d_dimer":    ("D-Dimer", [402], ["D-dimer"]),

    # ── Cardiac markers ───────────────────────────────────────────────────
    "troponin_i": ("Troponin I", [270], ["Troponin", "cardiac troponin"]),
    "ck_mb":      ("CK-MB", [195, 612, 253], ["creatine kinase MB"]),
    "nt_probnp":  ("NT-proBNP", [571], ["BNP", "brain natriuretic peptide"]),

    # ── Arterial blood gas ────────────────────────────────────────────────
    "ph":           ("Arterial pH", [538, 688], ["pH", "arterial pH"]),
    "pco2":         ("PaCO2", [443, 687], ["pCO2", "PaCO2", "CO2 partial pressure"]),
    "po2":          ("PaO2", [444, 689], ["pO2", "PaO2", "O2 partial pressure"]),
    "sao2":         ("SaO2", [184, 690], ["SaO2", "oxygen saturation", "sO2"]),
    "base_excess":  ("Base Excess", [449, 668], ["BE", "base excess"]),
    "lactate":      ("Lactate", [465, 454, 657], ["Laktat", "lactic acid"]),

    # ── Inflammatory ─────────────────────────────────────────────────────
    "crp":            ("C-Reactive Protein", [341, 351], ["CRP", "C-reaktives Protein", "high-sensitive CRP", "hsCRP"]),
    "procalcitonin":  ("Procalcitonin", [263], ["PCT"]),

    # ── Pancreatic ────────────────────────────────────────────────────────
    "lipase":  ("Lipase", [623], ["Lipase"]),
    "amylase": ("Amylase", [608], ["Amylase"]),

    # ── Lipids ────────────────────────────────────────────────────────────
    "total_cholesterol": ("Total Cholesterol", [338], ["cholesterol", "Cholesterin"]),
    "hdl_cholesterol":   ("HDL Cholesterol", [350], ["HDL", "HDL-Cholesterin"]),
    "ldl_cholesterol":   ("LDL Cholesterol", [373], ["LDL", "LDL-Cholesterin"]),
    "triglycerides":     ("Triglycerides", [385], ["Triglyceride"]),

    # ── Thyroid ───────────────────────────────────────────────────────────
    "tsh":      ("TSH", [479], ["Thyreotropin", "thyroid stimulating hormone"]),
    "free_t4":  ("Free T4", [576], ["FT4", "freies Thyroxin"]),
    "free_t3":  ("Free T3", [575], ["FT3", "freies Trijodthyronin"]),
}

# Panel definitions: panel_name -> (display_name, aliases, [canonical_item_names])
# Mirrors mimic-iv-ext's panel naming so the doctor-agent vocabulary transfers
# across datasets where SICdb has an equivalent panel.
PANEL_DEFS: dict[str, tuple[str, list[str], list[str]]] = {
    "CBC": (
        "Complete Blood Count",
        ["complete blood count", "blood count", "Blutbild"],
        ["white_blood_cells", "red_blood_cells", "hemoglobin", "hematocrit", "platelets"],
    ),
    "BMP": (
        "Basic Metabolic Panel",
        ["basic metabolic panel", "electrolytes", "chemistry"],
        ["glucose", "bun", "creatinine", "sodium", "potassium", "chloride", "bicarbonate", "calcium"],
    ),
    "LFT": (
        "Liver Function Tests",
        ["liver function tests", "liver panel", "hepatic panel"],
        ["ast", "alt", "alkaline_phosphatase", "ggt", "total_bilirubin", "direct_bilirubin", "albumin"],
    ),
    "coagulation": (
        "Coagulation Panel",
        ["coag panel", "clotting studies", "Gerinnung"],
        ["pt_inr", "ptt", "fibrinogen", "d_dimer"],
    ),
    "cardiac_markers": (
        "Cardiac Markers",
        ["cardiac enzymes", "cardiac biomarkers"],
        ["troponin_i", "ck_mb", "nt_probnp"],
    ),
    "ABG": (
        "Arterial Blood Gas",
        ["arterial blood gas", "blood gas analysis", "BGA"],
        ["ph", "pco2", "po2", "sao2", "base_excess", "bicarbonate", "lactate"],
    ),
    "inflammatory": (
        "Inflammatory Markers",
        ["inflammation panel", "sepsis markers"],
        ["crp", "procalcitonin", "white_blood_cells"],
    ),
    "lipid_panel": (
        "Lipid Panel",
        ["lipid profile", "cholesterol panel"],
        ["total_cholesterol", "hdl_cholesterol", "ldl_cholesterol", "triglycerides"],
    ),
    "thyroid": (
        "Thyroid Panel",
        ["thyroid function tests", "thyroid studies"],
        ["tsh", "free_t4", "free_t3"],
    ),
    "renal_function": (
        "Renal Function",
        ["kidney function", "renal panel"],
        ["bun", "creatinine", "sodium", "potassium"],
    ),
    "pancreatic": (
        "Pancreatic Enzymes",
        ["pancreas panel"],
        ["lipase", "amylase"],
    ),
}


def _load_lab_references() -> pd.DataFrame:
    refs = pd.read_csv(REFERENCES_FILE)
    return refs[refs["ReferenceName"] == "Laboratory"].set_index("ReferenceGlobalID")


def build_lab_catalog() -> dict:
    lab_refs = _load_lab_references()

    items = []
    for canonical, (display, ref_ids, extra_aliases) in ITEM_DEFS.items():
        valid_ids = [rid for rid in ref_ids if rid in lab_refs.index]
        if not valid_ids:
            continue
        first = lab_refs.loc[valid_ids[0]]
        # Pull in the raw German label(s) + LOINC short name(s) as additional
        # aliases automatically, supplementing the hand-curated list above.
        auto_aliases: list[str] = []
        for rid in valid_ids:
            row = lab_refs.loc[rid]
            raw_name = str(row["ReferenceValue"]).replace(" (ZL)", "").replace(" (BGA)", "").strip()
            if raw_name and raw_name not in auto_aliases:
                auto_aliases.append(raw_name)
            loinc = row.get("LOINC_short")
            if isinstance(loinc, str) and loinc.strip() and loinc not in auto_aliases:
                auto_aliases.append(loinc.strip())

        panels_for_item = [p for p, (_, _, names) in PANEL_DEFS.items() if canonical in names]

        items.append({
            "canonical_name": canonical,
            "display_name": display,
            "aliases": sorted(set(extra_aliases + auto_aliases)),
            "itemids": valid_ids,
            "fluid": "Blood",
            "category": "Laboratory",
            "panel": panels_for_item,
            "source_table": "laboratory_tests",
            "unit": str(first.get("ReferenceUnit")) if pd.notna(first.get("ReferenceUnit")) else None,
        })

    panels = [
        {
            "panel_name": pname,
            "display_name": display,
            "aliases": aliases,
            "items": [name for name in names if name in ITEM_DEFS and
                      any(rid in lab_refs.index for rid in ITEM_DEFS[name][1])],
        }
        for pname, (display, aliases, names) in PANEL_DEFS.items()
    ]

    return {
        "version": "1.0",
        "description": (
            "SICdb-Ext lab provider catalog. itemid = SICdb LaboratoryID "
            "(d_references.ReferenceGlobalID where ReferenceName=='Laboratory'), "
            "NOT comparable to MIMIC d_labitems.itemid."
        ),
        "panels": panels,
        "items": items,
    }


def main() -> None:
    os.makedirs(DATASET_DIR, exist_ok=True)
    catalog = build_lab_catalog()
    with open(OUT_LAB, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)
    print(f"Wrote {OUT_LAB}: {len(catalog['items'])} items, {len(catalog['panels'])} panels")


if __name__ == "__main__":
    main()
