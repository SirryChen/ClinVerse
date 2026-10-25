"""Build evidence_inventory for a case from MIMIC admission + trimmed Synthea persona."""

from __future__ import annotations

import re

from core.json_catalog import LAB_NAME_ZH
from core.microbiology import is_abnormal as micro_abnormal


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40]


def lab_evidence_content(adapter, itemid, valuestr: str, flag, is_abnormal: bool) -> str:
    """Render a lab record as a human-readable bilingual evidence string,
    e.g. "White Blood Cells（白细胞计数）: 7.2 [ABNORMAL]". Falls back to the
    raw itemid/value/flag format when the dataset adapter can't resolve the
    item (no adapter passed, or the itemid is in neither the curated catalog
    nor lab_test_mapping.csv)."""
    item = adapter.lab_catalog.resolve_itemid(itemid) if adapter is not None else None
    if item is None:
        return f"itemid={itemid} value={valuestr} flag={flag}"

    name_en = item.get("display_name") or item.get("canonical_name") or str(itemid)
    name_zh = LAB_NAME_ZH.get(item.get("canonical_name", ""))
    name = f"{name_en}（{name_zh}）" if name_zh and name_zh != name_en else name_en

    suffix = " [ABNORMAL]" if is_abnormal else ""
    return f"{name}: {valuestr}{suffix}"


def build_evidence_inventory(admission: dict, trimmed_persona: dict, adapter=None) -> list[dict]:
    inventory: list[dict] = []

    demo = admission["demographics"]

    # --- Initially visible demographics ---
    inventory.append({
        "evidence_id": "mimic_demo_age",
        "source": "mimic", "type": "demographics",
        "content": f"{demo.get('anchor_age', '?')} years old",
        "initially_visible": True, "collected": False,
    })
    inventory.append({
        "evidence_id": "mimic_demo_gender",
        "source": "mimic", "type": "demographics",
        "content": demo.get("gender", ""),
        "initially_visible": True, "collected": False,
    })
    inventory.append({
        "evidence_id": "mimic_demo_admission_type",
        "source": "mimic", "type": "demographics",
        "content": demo.get("admission_type", ""),
        "initially_visible": True, "collected": False,
    })

    # --- Initially visible vital signs (from PE first line) ---
    vital_text = admission.get("vital_signs", "")
    if vital_text:
        inventory.append({
            "evidence_id": "mimic_vitals_initial",
            "source": "mimic", "type": "vital_signs",
            "content": vital_text,
            "initially_visible": True, "collected": False,
        })

    # --- HPI (chief complaint context, initially visible) ---
    hpi = admission.get("hpi", "")
    if hpi:
        inventory.append({
            "evidence_id": "mimic_hpi",
            "source": "mimic", "type": "hpi",
            "content": hpi,
            "initially_visible": True, "collected": False,
        })

    # --- Labs (not initially visible) ---
    for lab in admission.get("labs", []):
        itemid = lab["itemid"]
        flag = lab.get("flag", "")
        is_abnormal = str(flag).lower() in ("abnormal", "critical", "delta")
        inventory.append({
            "evidence_id": f"mimic_lab_{itemid}",
            "source": "mimic", "type": "lab",
            "content": lab_evidence_content(adapter, itemid, lab.get("valuestr", ""), flag, is_abnormal),
            "itemid": itemid,
            "is_abnormal": is_abnormal,
            "initially_visible": False, "collected": False,
        })

    # --- Microbiology (not initially visible) ---
    for i, micro in enumerate(admission.get("microbiology", [])):
        inventory.append({
            "evidence_id": f"mimic_micro_{i}",
            "source": "mimic", "type": "microbiology",
            "content": f"{micro.get('test_name','')} {micro.get('spec_type_desc','')} {micro.get('valuestr','')}",
            "is_abnormal": micro_abnormal(micro),
            "initially_visible": False, "collected": False,
        })

    # --- Radiology (not initially visible) ---
    for rad in admission.get("radiology", []):
        note_id = rad.get("note_id", "")
        inventory.append({
            "evidence_id": f"mimic_rad_{note_id or _slug(rad.get('exam_name',''))}",
            "source": "mimic", "type": "radiology",
            "content": f"{rad.get('modality','')} {rad.get('region','')} {rad.get('exam_name','')}",
            "modality": rad.get("modality", ""),
            "region": rad.get("region", ""),
            "initially_visible": False, "collected": False,
        })

    # --- Synthea PMH (not initially visible) ---
    for cond in trimmed_persona.get("conditions", []):
        eid = f"synthea_pmh_{_slug(cond['display'])}"
        inventory.append({
            "evidence_id": eid,
            "source": "synthea", "type": "past_medical_history",
            "content": cond["display"],
            "onset_date": cond.get("onset_date", ""),
            "initially_visible": False, "collected": False,
        })

    # --- Synthea medications (not initially visible) ---
    for med in trimmed_persona.get("medications", []):
        eid = f"synthea_med_{_slug(med['display'])}"
        inventory.append({
            "evidence_id": eid,
            "source": "synthea", "type": "medication",
            "content": med["display"],
            "initially_visible": False, "collected": False,
        })

    # --- Synthea allergies (not initially visible) ---
    for allergy in trimmed_persona.get("allergies", []):
        eid = f"synthea_allergy_{_slug(allergy['substance'])}"
        inventory.append({
            "evidence_id": eid,
            "source": "synthea", "type": "allergy",
            "content": allergy["substance"],
            "initially_visible": False, "collected": False,
        })

    # --- Synthea immunizations (not initially visible) ---
    for imm in trimmed_persona.get("immunizations", []):
        eid = f"synthea_imm_{_slug(imm['vaccine'])}"
        inventory.append({
            "evidence_id": eid,
            "source": "synthea", "type": "immunization",
            "content": imm["vaccine"],
            "initially_visible": False, "collected": False,
        })

    # --- Synthea social/lifestyle from observations ---
    lifestyle_keywords = ["smoking", "tobacco", "alcohol", "bmi", "body mass", "employment", "education"]
    for obs in trimmed_persona.get("observations", []):
        display = obs.get("display", "").lower()
        if any(kw in display for kw in lifestyle_keywords):
            eid = f"synthea_social_{_slug(obs['display'])}"
            value_str = f"{obs.get('value', '')} {obs.get('unit', '')}".strip()
            inventory.append({
                "evidence_id": eid,
                "source": "synthea", "type": "social_history",
                "content": f"{obs['display']}: {value_str}",
                "initially_visible": False, "collected": False,
            })

    # Deduplicate by evidence_id (keep first occurrence)
    seen = set()
    deduped = []
    for item in inventory:
        if item["evidence_id"] not in seen:
            seen.add(item["evidence_id"])
            deduped.append(item)

    return deduped
