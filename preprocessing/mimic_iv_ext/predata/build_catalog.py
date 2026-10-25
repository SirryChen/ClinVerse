"""
Lab Provider & Radiology Catalog Builder
==========================================
Reads dataset/mimic-iv-ext/lab_test_mapping.csv and radiology_reports.csv
and produces two catalog JSON files:

  dataset/mimic-iv-ext/lab_provider_catalog.json
  dataset/mimic-iv-ext/radiology_catalog.json

Lab catalog structure (per item entry):
  {
    "canonical_name": "creatinine",
    "display_name": "Creatinine",
    "aliases": ["Cr", "serum creatinine", "kidney function"],
    "itemids": [50912],
    "fluid": "Blood",
    "category": "Chemistry",
    "panel": ["BMP", "CMP", "renal_function"]
  }

Panel index (separate section):
  {
    "panel_name": "BMP",
    "display_name": "Basic Metabolic Panel",
    "aliases": ["basic metabolic", "chem 7", "chemistry 7"],
    "items": ["glucose", "bun", "creatinine", "sodium", "potassium",
              "chloride", "bicarbonate", "calcium", "anion_gap"]
  }

Radiology catalog structure:
  {
    "canonical_name": "chest_xray",
    "display_name": "Chest X-ray",
    "aliases": ["CXR", "chest film", "chest radiograph", "portable chest", "chest PA"],
    "modality": "Radiograph",
    "regions": ["Chest"],
    "note_count": 435674
  }
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


import csv
import json
import os
import re
from collections import Counter, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DATASET_DIR = os.environ.get("CLINVERSE_NORMALIZED_DIR", os.path.join(REPO_ROOT, "data", "normalized", "mimic-iv-ext"))

LAB_MAPPING_CSV = os.path.join(DATASET_DIR, "lab_test_mapping.csv")
RADIOLOGY_CSV   = os.path.join(DATASET_DIR, "radiology_reports.csv")
OUT_LAB         = os.path.join(DATASET_DIR, "lab_provider_catalog.json")
OUT_RAD         = os.path.join(DATASET_DIR, "radiology_catalog.json")


# ---------------------------------------------------------------------------
# Panel definitions
# Each panel entry: panel_name -> (display_name, aliases, [canonical_item_names])
# ---------------------------------------------------------------------------
PANEL_DEFS: dict[str, tuple[str, list[str], list[str]]] = {
    "CBC": (
        "Complete Blood Count",
        ["complete blood count", "full blood count", "CBC with differential",
         "CBC w diff", "blood count", "hemogram"],
        ["white_blood_cells", "red_blood_cells", "hemoglobin", "hematocrit",
         "mcv", "mch", "mchc", "rdw", "platelet_count",
         "neutrophils", "lymphocytes", "monocytes", "eosinophils",
         "basophils", "bands"],
    ),
    "BMP": (
        "Basic Metabolic Panel",
        ["basic metabolic panel", "basic metabolic", "chem 7", "chemistry 7",
         "renal panel", "electrolytes"],
        ["glucose", "bun", "creatinine", "sodium", "potassium",
         "chloride", "bicarbonate", "calcium", "anion_gap"],
    ),
    "CMP": (
        "Comprehensive Metabolic Panel",
        ["comprehensive metabolic panel", "comprehensive metabolic", "chem 14",
         "chemistry 14", "liver panel", "hepatic panel"],
        ["glucose", "bun", "creatinine", "sodium", "potassium",
         "chloride", "bicarbonate", "calcium", "anion_gap",
         "alt", "ast", "alkaline_phosphatase", "bilirubin_total",
         "total_protein", "albumin"],
    ),
    "LFT": (
        "Liver Function Tests",
        ["liver function tests", "liver function", "hepatic panel", "LFTs"],
        ["alt", "ast", "alkaline_phosphatase", "ggt",
         "bilirubin_total", "bilirubin_direct", "total_protein", "albumin"],
    ),
    "coagulation": (
        "Coagulation Panel",
        ["coags", "coagulation studies", "clotting studies", "PT/PTT/INR",
         "coagulation panel"],
        ["pt", "ptt", "inr", "fibrinogen", "d_dimer"],
    ),
    "cardiac_markers": (
        "Cardiac Markers",
        ["cardiac enzymes", "cardiac biomarkers", "troponin panel",
         "heart enzymes", "cardiac panel"],
        ["troponin_t", "troponin_i", "ck_mb", "nt_probnp"],
    ),
    "ABG": (
        "Arterial Blood Gas",
        ["arterial blood gas", "blood gas", "arterial gas", "ABGs"],
        ["ph", "pco2", "po2", "bicarbonate_wb", "base_excess",
         "oxygen_saturation", "lactate_bg", "carboxyhemoglobin"],
    ),
    "VBG": (
        "Venous Blood Gas",
        ["venous blood gas", "venous gas", "VBGs"],
        ["ph", "pco2", "po2", "bicarbonate_wb", "base_excess",
         "oxygen_saturation", "lactate_bg"],
    ),
    "UA": (
        "Urinalysis",
        ["urinalysis", "urine analysis", "urine dip", "dipstick",
         "urine microscopy"],
        ["urine_color", "urine_appearance", "urine_glucose", "urine_protein",
         "urine_blood", "urine_ketone", "urine_nitrite", "urine_leukocytes",
         "urine_specific_gravity", "urine_ph", "urine_wbc", "urine_rbc",
         "urine_bacteria", "urine_bilirubin", "urine_urobilinogen",
         "urine_epithelial_cells"],
    ),
    "lipid_panel": (
        "Lipid Panel",
        ["lipid panel", "cholesterol panel", "fasting lipids", "lipids"],
        ["cholesterol_total", "cholesterol_hdl", "cholesterol_ldl",
         "triglycerides"],
    ),
    "thyroid": (
        "Thyroid Function Tests",
        ["thyroid panel", "thyroid function", "TFTs"],
        ["tsh", "free_t4", "t3"],
    ),
    "iron_studies": (
        "Iron Studies",
        ["iron panel", "iron studies", "anemia workup"],
        ["iron", "ferritin", "tibc"],
    ),
    "renal_function": (
        "Renal Function Panel",
        ["kidney function", "renal function", "creatinine panel"],
        ["creatinine", "bun", "sodium", "potassium",
         "bicarbonate", "anion_gap"],
    ),
    "inflammatory": (
        "Inflammatory Markers",
        ["inflammatory markers", "acute phase reactants", "sepsis markers"],
        ["crp", "esr", "procalcitonin"],
    ),
}


# ---------------------------------------------------------------------------
# Individual item definitions
# Each entry: canonical_name -> (display_name, itemids, aliases, panels, fluid, category)
# ---------------------------------------------------------------------------
ITEM_DEFS: list[dict] = [
    # ---- CBC items ----
    {"canonical_name": "white_blood_cells",
     "display_name": "White Blood Cells",
     "itemids": [51301, 51300],
     "aliases": ["WBC", "leukocytes", "white cells", "leukocyte count"],
     "panel": ["CBC"],
     "fluid": "Blood", "category": "Hematology"},

    {"canonical_name": "red_blood_cells",
     "display_name": "Red Blood Cells",
     "itemids": [51279],
     "aliases": ["RBC", "erythrocytes", "red cells"],
     "panel": ["CBC"],
     "fluid": "Blood", "category": "Hematology"},

    {"canonical_name": "hemoglobin",
     "display_name": "Hemoglobin",
     "itemids": [51222],
     "aliases": ["Hgb", "Hb", "haemoglobin"],
     "panel": ["CBC"],
     "fluid": "Blood", "category": "Hematology"},

    {"canonical_name": "hematocrit",
     "display_name": "Hematocrit",
     "itemids": [51221],
     "aliases": ["Hct", "PCV", "packed cell volume"],
     "panel": ["CBC"],
     "fluid": "Blood", "category": "Hematology"},

    {"canonical_name": "mcv",
     "display_name": "MCV",
     "itemids": [51250],
     "aliases": ["mean corpuscular volume", "mean cell volume"],
     "panel": ["CBC"],
     "fluid": "Blood", "category": "Hematology"},

    {"canonical_name": "mch",
     "display_name": "MCH",
     "itemids": [51248],
     "aliases": ["mean corpuscular hemoglobin"],
     "panel": ["CBC"],
     "fluid": "Blood", "category": "Hematology"},

    {"canonical_name": "mchc",
     "display_name": "MCHC",
     "itemids": [51249],
     "aliases": ["mean corpuscular hemoglobin concentration"],
     "panel": ["CBC"],
     "fluid": "Blood", "category": "Hematology"},

    {"canonical_name": "rdw",
     "display_name": "RDW",
     "itemids": [51277],
     "aliases": ["red cell distribution width", "red blood cell distribution width"],
     "panel": ["CBC"],
     "fluid": "Blood", "category": "Hematology"},

    {"canonical_name": "platelet_count",
     "display_name": "Platelet Count",
     "itemids": [51265],
     "aliases": ["platelets", "thrombocytes", "PLT", "plt count"],
     "panel": ["CBC"],
     "fluid": "Blood", "category": "Hematology"},

    {"canonical_name": "neutrophils",
     "display_name": "Neutrophils",
     "itemids": [51256],
     "aliases": ["neutrophil count", "PMN", "segs", "granulocytes",
                 "absolute neutrophil count", "ANC"],
     "panel": ["CBC"],
     "fluid": "Blood", "category": "Hematology"},

    {"canonical_name": "lymphocytes",
     "display_name": "Lymphocytes",
     "itemids": [51244],
     "aliases": ["lymphocyte count", "absolute lymphocyte count", "ALC"],
     "panel": ["CBC"],
     "fluid": "Blood", "category": "Hematology"},

    {"canonical_name": "monocytes",
     "display_name": "Monocytes",
     "itemids": [51254],
     "aliases": ["monocyte count"],
     "panel": ["CBC"],
     "fluid": "Blood", "category": "Hematology"},

    {"canonical_name": "eosinophils",
     "display_name": "Eosinophils",
     "itemids": [51200],
     "aliases": ["eosinophil count", "eos"],
     "panel": ["CBC"],
     "fluid": "Blood", "category": "Hematology"},

    {"canonical_name": "basophils",
     "display_name": "Basophils",
     "itemids": [51146],
     "aliases": ["basophil count"],
     "panel": ["CBC"],
     "fluid": "Blood", "category": "Hematology"},

    {"canonical_name": "bands",
     "display_name": "Bands",
     "itemids": [51144],
     "aliases": ["band neutrophils", "band cells", "stab cells",
                 "immature granulocytes"],
     "panel": ["CBC"],
     "fluid": "Blood", "category": "Hematology"},

    # ---- BMP / CMP / renal items ----
    {"canonical_name": "glucose",
     "display_name": "Glucose",
     "itemids": [50931],
     "aliases": ["blood glucose", "serum glucose", "dextrose", "blood sugar",
                 "fasting glucose"],
     "panel": ["BMP", "CMP"],
     "fluid": "Blood", "category": "Chemistry"},

    {"canonical_name": "bun",
     "display_name": "BUN",
     "itemids": [51006],
     "aliases": ["blood urea nitrogen", "urea nitrogen", "urea", "BUN",
                 "blood urea"],
     "panel": ["BMP", "CMP", "renal_function"],
     "fluid": "Blood", "category": "Chemistry"},

    {"canonical_name": "creatinine",
     "display_name": "Creatinine",
     "itemids": [50912],
     "aliases": ["Cr", "serum creatinine", "kidney function", "creat"],
     "panel": ["BMP", "CMP", "renal_function"],
     "fluid": "Blood", "category": "Chemistry"},

    {"canonical_name": "sodium",
     "display_name": "Sodium",
     "itemids": [50983],
     "aliases": ["Na", "serum sodium", "Na+"],
     "panel": ["BMP", "CMP", "renal_function"],
     "fluid": "Blood", "category": "Chemistry"},

    {"canonical_name": "potassium",
     "display_name": "Potassium",
     "itemids": [50971],
     "aliases": ["K", "serum potassium", "K+"],
     "panel": ["BMP", "CMP", "renal_function"],
     "fluid": "Blood", "category": "Chemistry"},

    {"canonical_name": "chloride",
     "display_name": "Chloride",
     "itemids": [50902],
     "aliases": ["Cl", "serum chloride", "Cl-"],
     "panel": ["BMP", "CMP"],
     "fluid": "Blood", "category": "Chemistry"},

    {"canonical_name": "bicarbonate",
     "display_name": "Bicarbonate",
     "itemids": [50882],
     "aliases": ["CO2", "HCO3", "bicarb", "serum bicarbonate",
                 "total CO2", "tCO2"],
     "panel": ["BMP", "CMP", "renal_function"],
     "fluid": "Blood", "category": "Chemistry"},

    {"canonical_name": "calcium",
     "display_name": "Calcium",
     "itemids": [50893],
     "aliases": ["Ca", "total calcium", "serum calcium", "Ca2+"],
     "panel": ["BMP", "CMP"],
     "fluid": "Blood", "category": "Chemistry"},

    {"canonical_name": "anion_gap",
     "display_name": "Anion Gap",
     "itemids": [50868],
     "aliases": ["AG", "anion gap"],
     "panel": ["BMP", "CMP", "renal_function"],
     "fluid": "Blood", "category": "Chemistry"},

    {"canonical_name": "magnesium",
     "display_name": "Magnesium",
     "itemids": [50960],
     "aliases": ["Mg", "serum magnesium", "Mg2+"],
     "panel": [],
     "fluid": "Blood", "category": "Chemistry"},

    {"canonical_name": "phosphate",
     "display_name": "Phosphate",
     "itemids": [50970],
     "aliases": ["Phos", "phosphorus", "serum phosphate", "inorganic phosphate"],
     "panel": [],
     "fluid": "Blood", "category": "Chemistry"},

    # ---- LFT / CMP items ----
    {"canonical_name": "alt",
     "display_name": "ALT",
     "itemids": [50861],
     "aliases": ["alanine aminotransferase", "SGPT", "alanine transaminase",
                 "ALT", "liver enzymes"],
     "panel": ["CMP", "LFT"],
     "fluid": "Blood", "category": "Chemistry"},

    {"canonical_name": "ast",
     "display_name": "AST",
     "itemids": [50878],
     "aliases": ["aspartate aminotransferase", "SGOT", "aspartate transaminase",
                 "AST"],
     "panel": ["CMP", "LFT"],
     "fluid": "Blood", "category": "Chemistry"},

    {"canonical_name": "alkaline_phosphatase",
     "display_name": "Alkaline Phosphatase",
     "itemids": [50863],
     "aliases": ["ALP", "alk phos", "alkaline phosphatase"],
     "panel": ["CMP", "LFT"],
     "fluid": "Blood", "category": "Chemistry"},

    {"canonical_name": "ggt",
     "display_name": "GGT",
     "itemids": [50927],
     "aliases": ["gamma-glutamyltransferase", "gamma GT", "GGT",
                 "gamma glutamyl transferase"],
     "panel": ["LFT"],
     "fluid": "Blood", "category": "Chemistry"},

    {"canonical_name": "bilirubin_total",
     "display_name": "Bilirubin, Total",
     "itemids": [50885],
     "aliases": ["total bilirubin", "T bili", "tbili", "bilirubin"],
     "panel": ["CMP", "LFT"],
     "fluid": "Blood", "category": "Chemistry"},

    {"canonical_name": "bilirubin_direct",
     "display_name": "Bilirubin, Direct",
     "itemids": [50883],
     "aliases": ["direct bilirubin", "conjugated bilirubin", "D bili"],
     "panel": ["LFT"],
     "fluid": "Blood", "category": "Chemistry"},

    {"canonical_name": "bilirubin_indirect",
     "display_name": "Bilirubin, Indirect",
     "itemids": [50884],
     "aliases": ["indirect bilirubin", "unconjugated bilirubin"],
     "panel": ["LFT"],
     "fluid": "Blood", "category": "Chemistry"},

    {"canonical_name": "total_protein",
     "display_name": "Total Protein",
     "itemids": [50976, 53096],
     "aliases": ["TP", "serum protein", "total serum protein"],
     "panel": ["CMP", "LFT"],
     "fluid": "Blood", "category": "Chemistry"},

    {"canonical_name": "albumin",
     "display_name": "Albumin",
     "itemids": [50862],
     "aliases": ["serum albumin", "alb"],
     "panel": ["CMP", "LFT"],
     "fluid": "Blood", "category": "Chemistry"},

    # ---- Coagulation ----
    {"canonical_name": "pt",
     "display_name": "PT",
     "itemids": [51274],
     "aliases": ["prothrombin time", "pro time", "PT"],
     "panel": ["coagulation"],
     "fluid": "Blood", "category": "Hematology"},

    {"canonical_name": "ptt",
     "display_name": "PTT",
     "itemids": [51275],
     "aliases": ["partial thromboplastin time", "aPTT", "activated PTT",
                 "activated partial thromboplastin time"],
     "panel": ["coagulation"],
     "fluid": "Blood", "category": "Hematology"},

    {"canonical_name": "inr",
     "display_name": "INR",
     "itemids": [51237, 51675],
     "aliases": ["international normalized ratio", "INR(PT)"],
     "panel": ["coagulation"],
     "fluid": "Blood", "category": "Hematology"},

    {"canonical_name": "fibrinogen",
     "display_name": "Fibrinogen",
     "itemids": [52116, 51214],
     "aliases": ["fibrinogen level", "clotting factor I"],
     "panel": ["coagulation"],
     "fluid": "Blood", "category": "Hematology"},

    {"canonical_name": "d_dimer",
     "display_name": "D-Dimer",
     "itemids": [51196, 50915],
     "aliases": ["D dimer", "fibrin degradation", "FDP"],
     "panel": ["coagulation"],
     "fluid": "Blood", "category": "Hematology"},

    # ---- Cardiac markers ----
    {"canonical_name": "troponin_t",
     "display_name": "Troponin T",
     "itemids": [51003],
     "aliases": ["TnT", "high sensitivity troponin T", "hsTnT",
                 "cardiac troponin T"],
     "panel": ["cardiac_markers"],
     "fluid": "Blood", "category": "Chemistry"},

    {"canonical_name": "troponin_i",
     "display_name": "Troponin I",
     "itemids": [51002],
     "aliases": ["TnI", "cardiac troponin I", "high sensitivity troponin I",
                 "hsTnI"],
     "panel": ["cardiac_markers"],
     "fluid": "Blood", "category": "Chemistry"},

    {"canonical_name": "ck_mb",
     "display_name": "CK-MB",
     "itemids": [50911],
     "aliases": ["CK-MB isoenzyme", "creatine kinase MB", "CKMB"],
     "panel": ["cardiac_markers"],
     "fluid": "Blood", "category": "Chemistry"},

    {"canonical_name": "nt_probnp",
     "display_name": "NT-proBNP",
     "itemids": [50963],
     "aliases": ["NTproBNP", "BNP", "brain natriuretic peptide",
                 "N-terminal proBNP", "heart failure marker"],
     "panel": ["cardiac_markers"],
     "fluid": "Blood", "category": "Chemistry"},

    # ---- ABG / VBG items ----
    {"canonical_name": "ph",
     "display_name": "pH",
     "itemids": [50820],
     "aliases": ["blood pH", "arterial pH", "acid-base"],
     "panel": ["ABG", "VBG"],
     "fluid": "Blood", "category": "Blood Gas"},

    {"canonical_name": "pco2",
     "display_name": "pCO2",
     "itemids": [50818],
     "aliases": ["partial pressure CO2", "carbon dioxide tension", "PCO2"],
     "panel": ["ABG", "VBG"],
     "fluid": "Blood", "category": "Blood Gas"},

    {"canonical_name": "po2",
     "display_name": "pO2",
     "itemids": [50821],
     "aliases": ["partial pressure O2", "oxygen tension", "PO2"],
     "panel": ["ABG", "VBG"],
     "fluid": "Blood", "category": "Blood Gas"},

    {"canonical_name": "bicarbonate_wb",
     "display_name": "Bicarbonate (Blood Gas)",
     "itemids": [50803],
     "aliases": ["HCO3", "calculated bicarbonate", "bicarb ABG"],
     "panel": ["ABG", "VBG"],
     "fluid": "Blood", "category": "Blood Gas"},

    {"canonical_name": "base_excess",
     "display_name": "Base Excess",
     "itemids": [50802],
     "aliases": ["BE", "base deficit", "standard base excess"],
     "panel": ["ABG", "VBG"],
     "fluid": "Blood", "category": "Blood Gas"},

    {"canonical_name": "oxygen_saturation",
     "display_name": "Oxygen Saturation",
     "itemids": [50817],
     "aliases": ["O2 sat", "SaO2", "SO2", "oxygen sat ABG"],
     "panel": ["ABG", "VBG"],
     "fluid": "Blood", "category": "Blood Gas"},

    {"canonical_name": "lactate_bg",
     "display_name": "Lactate",
     "itemids": [50813, 52442],
     "aliases": ["blood lactate", "lactic acid", "serum lactate",
                 "lactate level"],
     "panel": ["ABG", "VBG"],
     "fluid": "Blood", "category": "Blood Gas"},

    {"canonical_name": "carboxyhemoglobin",
     "display_name": "Carboxyhemoglobin",
     "itemids": [50805],
     "aliases": ["COHb", "carbon monoxide hemoglobin", "CO-Hb"],
     "panel": ["ABG"],
     "fluid": "Blood", "category": "Blood Gas"},

    # ---- Standalone items ----
    {"canonical_name": "lipase",
     "display_name": "Lipase",
     "itemids": [50956],
     "aliases": ["serum lipase", "pancreatic lipase"],
     "panel": [],
     "fluid": "Blood", "category": "Chemistry"},

    {"canonical_name": "amylase",
     "display_name": "Amylase",
     "itemids": [50867],
     "aliases": ["serum amylase", "pancreatic amylase"],
     "panel": [],
     "fluid": "Blood", "category": "Chemistry"},

    {"canonical_name": "crp",
     "display_name": "C-Reactive Protein",
     "itemids": [50889],
     "aliases": ["CRP", "c reactive protein", "acute phase protein",
                 "inflammation marker"],
     "panel": ["inflammatory"],
     "fluid": "Blood", "category": "Chemistry"},

    {"canonical_name": "procalcitonin",
     "display_name": "Procalcitonin",
     "itemids": [50975],
     "aliases": ["PCT", "sepsis marker", "procalcitonin level"],
     "panel": ["inflammatory"],
     "fluid": "Blood", "category": "Chemistry"},

    {"canonical_name": "ammonia",
     "display_name": "Ammonia",
     "itemids": [50866],
     "aliases": ["serum ammonia", "plasma ammonia", "NH3"],
     "panel": [],
     "fluid": "Blood", "category": "Chemistry"},

    {"canonical_name": "ldh",
     "display_name": "LDH",
     "itemids": [50954],
     "aliases": ["lactate dehydrogenase", "lactic dehydrogenase",
                 "LD", "LDH"],
     "panel": [],
     "fluid": "Blood", "category": "Chemistry"},

    {"canonical_name": "uric_acid",
     "display_name": "Uric Acid",
     "itemids": [51007],
     "aliases": ["urate", "serum uric acid", "gout marker"],
     "panel": [],
     "fluid": "Blood", "category": "Chemistry"},

    # ---- Thyroid ----
    {"canonical_name": "tsh",
     "display_name": "TSH",
     "itemids": [50993],
     "aliases": ["thyroid stimulating hormone", "thyrotropin",
                 "thyroid function"],
     "panel": ["thyroid"],
     "fluid": "Blood", "category": "Chemistry"},

    {"canonical_name": "free_t4",
     "display_name": "Free T4",
     "itemids": [50994],
     "aliases": ["FT4", "free thyroxine", "thyroxine"],
     "panel": ["thyroid"],
     "fluid": "Blood", "category": "Chemistry"},

    {"canonical_name": "t3",
     "display_name": "T3",
     "itemids": [51001],
     "aliases": ["triiodothyronine", "T3", "free T3"],
     "panel": ["thyroid"],
     "fluid": "Blood", "category": "Chemistry"},

    # ---- Iron studies ----
    {"canonical_name": "iron",
     "display_name": "Iron",
     "itemids": [50952],
     "aliases": ["serum iron", "Fe", "iron level"],
     "panel": ["iron_studies"],
     "fluid": "Blood", "category": "Chemistry"},

    {"canonical_name": "ferritin",
     "display_name": "Ferritin",
     "itemids": [50924],
     "aliases": ["serum ferritin", "iron storage"],
     "panel": ["iron_studies"],
     "fluid": "Blood", "category": "Chemistry"},

    {"canonical_name": "tibc",
     "display_name": "TIBC",
     "itemids": [50953],
     "aliases": ["total iron binding capacity", "iron binding capacity",
                 "TIBC"],
     "panel": ["iron_studies"],
     "fluid": "Blood", "category": "Chemistry"},

    # ---- Lipid panel ----
    {"canonical_name": "cholesterol_total",
     "display_name": "Total Cholesterol",
     "itemids": [50907],
     "aliases": ["total cholesterol", "cholesterol"],
     "panel": ["lipid_panel"],
     "fluid": "Blood", "category": "Chemistry"},

    {"canonical_name": "cholesterol_hdl",
     "display_name": "HDL Cholesterol",
     "itemids": [50904],
     "aliases": ["HDL", "HDL cholesterol", "good cholesterol"],
     "panel": ["lipid_panel"],
     "fluid": "Blood", "category": "Chemistry"},

    {"canonical_name": "cholesterol_ldl",
     "display_name": "LDL Cholesterol",
     "itemids": [50905],
     "aliases": ["LDL", "LDL cholesterol", "bad cholesterol",
                 "LDL-C"],
     "panel": ["lipid_panel"],
     "fluid": "Blood", "category": "Chemistry"},

    {"canonical_name": "triglycerides",
     "display_name": "Triglycerides",
     "itemids": [51000],
     "aliases": ["TG", "trigs", "serum triglycerides", "fats"],
     "panel": ["lipid_panel"],
     "fluid": "Blood", "category": "Chemistry"},

    # ---- Urine items (UA) ----
    {"canonical_name": "urine_color",
     "display_name": "Urine Color",
     "itemids": [51508],
     "aliases": [],
     "panel": ["UA"],
     "fluid": "Urine", "category": "Hematology"},

    {"canonical_name": "urine_appearance",
     "display_name": "Urine Appearance",
     "itemids": [51506],
     "aliases": ["urine clarity", "urine turbidity"],
     "panel": ["UA"],
     "fluid": "Urine", "category": "Hematology"},

    {"canonical_name": "urine_glucose",
     "display_name": "Urine Glucose",
     "itemids": [51478],
     "aliases": ["glucosuria"],
     "panel": ["UA"],
     "fluid": "Urine", "category": "Hematology"},

    {"canonical_name": "urine_protein",
     "display_name": "Urine Protein",
     "itemids": [51492],
     "aliases": ["proteinuria", "urine albumin dipstick"],
     "panel": ["UA"],
     "fluid": "Urine", "category": "Hematology"},

    {"canonical_name": "urine_blood",
     "display_name": "Urine Blood",
     "itemids": [51466],
     "aliases": ["hematuria", "blood in urine", "urine hemoglobin"],
     "panel": ["UA"],
     "fluid": "Urine", "category": "Hematology"},

    {"canonical_name": "urine_ketone",
     "display_name": "Urine Ketones",
     "itemids": [51484],
     "aliases": ["ketonuria", "urine acetone"],
     "panel": ["UA"],
     "fluid": "Urine", "category": "Hematology"},

    {"canonical_name": "urine_nitrite",
     "display_name": "Urine Nitrite",
     "itemids": [51487],
     "aliases": ["nitrites", "urine bacteria screen"],
     "panel": ["UA"],
     "fluid": "Urine", "category": "Hematology"},

    {"canonical_name": "urine_leukocytes",
     "display_name": "Urine Leukocytes",
     "itemids": [51486],
     "aliases": ["leukocyte esterase", "LE", "WBCs urine dip"],
     "panel": ["UA"],
     "fluid": "Urine", "category": "Hematology"},

    {"canonical_name": "urine_specific_gravity",
     "display_name": "Urine Specific Gravity",
     "itemids": [51498],
     "aliases": ["SG", "urine concentration"],
     "panel": ["UA"],
     "fluid": "Urine", "category": "Hematology"},

    {"canonical_name": "urine_ph",
     "display_name": "Urine pH",
     "itemids": [51491],
     "aliases": ["urine acidity", "urine acid-base"],
     "panel": ["UA"],
     "fluid": "Urine", "category": "Hematology"},

    {"canonical_name": "urine_wbc",
     "display_name": "Urine WBC",
     "itemids": [51516],
     "aliases": ["white blood cells urine", "pus cells", "pyuria"],
     "panel": ["UA"],
     "fluid": "Urine", "category": "Hematology"},

    {"canonical_name": "urine_rbc",
     "display_name": "Urine RBC",
     "itemids": [51493],
     "aliases": ["red blood cells urine", "urine erythrocytes"],
     "panel": ["UA"],
     "fluid": "Urine", "category": "Hematology"},

    {"canonical_name": "urine_bacteria",
     "display_name": "Urine Bacteria",
     "itemids": [51463],
     "aliases": ["bacteriuria", "urine organisms"],
     "panel": ["UA"],
     "fluid": "Urine", "category": "Hematology"},

    {"canonical_name": "urine_bilirubin",
     "display_name": "Urine Bilirubin",
     "itemids": [51464],
     "aliases": [],
     "panel": ["UA"],
     "fluid": "Urine", "category": "Hematology"},

    {"canonical_name": "urine_urobilinogen",
     "display_name": "Urine Urobilinogen",
     "itemids": [51514],
     "aliases": [],
     "panel": ["UA"],
     "fluid": "Urine", "category": "Hematology"},

    {"canonical_name": "urine_epithelial_cells",
     "display_name": "Urine Epithelial Cells",
     "itemids": [51476],
     "aliases": ["squamous epithelial cells", "epithelials"],
     "panel": ["UA"],
     "fluid": "Urine", "category": "Hematology"},

    # ---- Microbiology (separate table, no itemids) ----
    {"canonical_name": "blood_culture",
     "display_name": "Blood Culture",
     "itemids": [],
     "aliases": ["blood cx", "bacteremia workup", "sepsis workup",
                 "hemoculture"],
     "panel": [],
     "fluid": "Blood", "category": "Microbiology",
     "source_table": "microbiology"},

    {"canonical_name": "urine_culture",
     "display_name": "Urine Culture",
     "itemids": [],
     "aliases": ["urine cx", "urine C&S", "UTI workup"],
     "panel": [],
     "fluid": "Urine", "category": "Microbiology",
     "source_table": "microbiology"},

    {"canonical_name": "sputum_culture",
     "display_name": "Sputum Culture",
     "itemids": [],
     "aliases": ["sputum cx", "respiratory culture", "BAL culture",
                 "bronchial culture"],
     "panel": [],
     "fluid": "Sputum", "category": "Microbiology",
     "source_table": "microbiology"},
]


# ---------------------------------------------------------------------------
# Radiology catalog
# ---------------------------------------------------------------------------
RADIOLOGY_DEFS: list[dict] = [
    {
        "canonical_name": "chest_xray",
        "display_name": "Chest X-ray",
        "aliases": ["CXR", "chest film", "chest radiograph",
                    "portable chest", "chest PA", "chest AP",
                    "plain chest", "chest x ray"],
        "modality": "Radiograph",
        "regions": ["Chest"],
    },
    {
        "canonical_name": "ct_head",
        "display_name": "CT Head",
        "aliases": ["head CT", "brain CT", "CT brain",
                    "non-contrast head CT", "NCHCT", "CT scan head"],
        "modality": "CT",
        "regions": ["Head"],
    },
    {
        "canonical_name": "ct_chest",
        "display_name": "CT Chest",
        "aliases": ["chest CT", "CT thorax", "pulmonary CT",
                    "CTA chest", "CT pulmonary embolism", "CTPE",
                    "CT angiography chest"],
        "modality": "CT",
        "regions": ["Chest"],
    },
    {
        "canonical_name": "ct_abdomen_pelvis",
        "display_name": "CT Abdomen/Pelvis",
        "aliases": ["CT abdomen", "CT pelvis", "CT AP",
                    "CT abd pelvis", "CT abdomen and pelvis",
                    "abdominal CT"],
        "modality": "CT",
        "regions": ["Abdomen", "Pelvis"],
    },
    {
        "canonical_name": "ct_spine",
        "display_name": "CT Spine",
        "aliases": ["CT cervical spine", "CT lumbar spine",
                    "CT thoracic spine", "CT c-spine"],
        "modality": "CT",
        "regions": ["Spine"],
    },
    {
        "canonical_name": "ct_neck",
        "display_name": "CT Neck",
        "aliases": ["CT soft tissue neck", "CT neck with contrast"],
        "modality": "CT",
        "regions": ["Neck"],
    },
    {
        "canonical_name": "mri_head",
        "display_name": "MRI Head/Brain",
        "aliases": ["MRI brain", "MR head", "brain MRI",
                    "MRA brain", "MRV brain", "DWI brain"],
        "modality": "MRI",
        "regions": ["Head"],
    },
    {
        "canonical_name": "mri_spine",
        "display_name": "MRI Spine",
        "aliases": ["MRI cervical spine", "MRI lumbar spine",
                    "MRI c-spine", "MRI l-spine"],
        "modality": "MRI",
        "regions": ["Spine", "Neck"],
    },
    {
        "canonical_name": "mri_abdomen",
        "display_name": "MRI Abdomen",
        "aliases": ["MRCP", "MR abdomen", "liver MRI",
                    "MR cholangiopancreatography"],
        "modality": "MRI",
        "regions": ["Abdomen"],
    },
    {
        "canonical_name": "ultrasound_abdomen",
        "display_name": "Abdominal Ultrasound",
        "aliases": ["abdominal US", "RUQ ultrasound",
                    "liver gallbladder US", "biliary US",
                    "renal US", "abdominal sonogram"],
        "modality": "Ultrasound",
        "regions": ["Abdomen", "Unknown"],
    },
    {
        "canonical_name": "ultrasound_venous",
        "display_name": "Venous Duplex Ultrasound",
        "aliases": ["DVT ultrasound", "lower extremity duplex",
                    "venous duplex", "compression US veins",
                    "upper extremity veins US"],
        "modality": "Ultrasound",
        "regions": ["Venous"],
    },
    {
        "canonical_name": "ultrasound_extremity",
        "display_name": "Extremity Ultrasound",
        "aliases": ["limb ultrasound", "extremity US"],
        "modality": "Ultrasound",
        "regions": ["Extremity"],
    },
    {
        "canonical_name": "echocardiogram",
        "display_name": "Echocardiogram",
        "aliases": ["echo", "TTE", "TEE", "transthoracic echo",
                    "transesophageal echo", "cardiac echo",
                    "heart ultrasound"],
        "modality": "Ultrasound",
        "regions": ["Chest"],
        "note": "MIMIC has limited echo in radiology table; may be in cardiology notes",
    },
]


def _to_snake(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def load_lab_mapping() -> dict[str, dict]:
    """Return {itemid_str: row_dict} from lab_test_mapping.csv."""
    mapping = {}
    with open(LAB_MAPPING_CSV, newline="") as f:
        for row in csv.DictReader(f):
            mapping[row["itemid"]] = row
    return mapping


def load_radiology_note_counts() -> Counter:
    """Count (modality, region) occurrences in radiology_reports.csv."""
    counts: Counter = Counter()
    with open(RADIOLOGY_CSV, newline="") as f:
        for row in csv.DictReader(f):
            counts[(row["modality"], row["region"])] += 1
    return counts


def build_lab_catalog(lab_mapping: dict) -> dict:
    catalog_items = []
    for item in ITEM_DEFS:
        entry = {
            "canonical_name": item["canonical_name"],
            "display_name": item["display_name"],
            "aliases": item["aliases"],
            "itemids": item["itemids"],
            "fluid": item["fluid"],
            "category": item["category"],
            "panel": item["panel"],
        }
        if "source_table" in item:
            entry["source_table"] = item["source_table"]
        else:
            entry["source_table"] = "laboratory_tests"

        # Verify itemids exist in catalog
        missing = [i for i in item["itemids"] if str(i) not in lab_mapping]
        if missing:
            entry["_warning"] = f"itemids not in lab_test_mapping: {missing}"

        catalog_items.append(entry)

    panels = []
    for panel_name, (display, aliases, items) in PANEL_DEFS.items():
        panels.append({
            "panel_name": panel_name,
            "display_name": display,
            "aliases": aliases,
            "items": items,
        })

    return {
        "version": "1.0",
        "description": (
            "Lab Provider catalog for MIMIC-IV-Ext. Maps canonical test names "
            "and panel names to MIMIC-IV itemids. Use canonical_name or any "
            "alias when requesting tests from the Lab Provider."
        ),
        "panels": panels,
        "items": catalog_items,
    }


def build_radiology_catalog(rad_counts: Counter) -> dict:
    catalog = []
    for defn in RADIOLOGY_DEFS:
        total = sum(
            rad_counts.get((defn["modality"], r), 0)
            for r in defn["regions"]
        )
        entry = {
            "canonical_name": defn["canonical_name"],
            "display_name": defn["display_name"],
            "aliases": defn["aliases"],
            "modality": defn["modality"],
            "regions": defn["regions"],
            "note_count": total,
        }
        if "note" in defn:
            entry["note"] = defn["note"]
        catalog.append(entry)

    return {
        "version": "1.0",
        "description": (
            "Radiology catalog for MIMIC-IV-Ext. Maps canonical study names "
            "to modality and region filters used in radiology_reports.csv. "
            "note_count reflects total reports of that type in the dataset."
        ),
        "studies": catalog,
    }


def main() -> None:
    print(f"Reading lab mapping from {LAB_MAPPING_CSV}")
    lab_mapping = load_lab_mapping()
    print(f"  {len(lab_mapping)} items loaded")

    print(f"Reading radiology data from {RADIOLOGY_CSV}")
    rad_counts = load_radiology_note_counts()
    print(f"  {sum(rad_counts.values())} reports, "
          f"{len(rad_counts)} unique modality/region combos")

    lab_catalog = build_lab_catalog(lab_mapping)
    rad_catalog = build_radiology_catalog(rad_counts)

    with open(OUT_LAB, "w") as f:
        json.dump(lab_catalog, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {OUT_LAB}")
    print(f"  {len(lab_catalog['panels'])} panels, "
          f"{len(lab_catalog['items'])} items")

    with open(OUT_RAD, "w") as f:
        json.dump(rad_catalog, f, indent=2, ensure_ascii=False)
    print(f"Wrote {OUT_RAD}")
    print(f"  {len(rad_catalog['studies'])} study types")

    # Print any warnings
    warnings = [i for i in lab_catalog["items"] if "_warning" in i]
    if warnings:
        print(f"\nWarnings ({len(warnings)} items with unmapped itemids):")
        for w in warnings:
            print(f"  {w['canonical_name']}: {w['_warning']}")
    else:
        print("\nAll itemids verified in lab_test_mapping.")


if __name__ == "__main__":
    main()
