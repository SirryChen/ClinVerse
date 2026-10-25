"""ClinLab prompt construction shared by training and prediction."""
import math

SYSTEM_PROMPT = (
    "You are a clinical laboratory assistant with complete access to a patient's medical record. "
    "Given the patient's full clinical information and the specified laboratory test, "
    "predict the final test result. "
    "Respond with ONLY the numeric result (number only). Do not include units, labels, or explanations."
)

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

def _truncate(text: str, max_chars: int) -> str:
    if _is_missing(text):
        return ""
    text = str(text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " [...]"

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
    cfg_data: dict | None = None,
) -> str:
    cfg_data = cfg_data or {}
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
    hpi_text = _truncate(hpi, cfg_data.get("hpi_max_chars", 500))
    if hpi_text:
        lines.append("\n## History of Present Illness")
        lines.append(hpi_text)

    # ── 6. Physical Examination ───────────────────────────────────────────────
    pe_text = _truncate(pe, cfg_data.get("pe_max_chars", 2000))
    if pe_text:
        lines.append("\n## Physical Examination")
        lines.append(pe_text)

    # ── 7. Hospital Course ────────────────────────────────────────────────────
    hc_text = _truncate(hospital_course, cfg_data.get("hospital_course_max_chars", 6000))
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
