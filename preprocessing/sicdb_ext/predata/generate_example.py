"""
Generate example-json.txt from dataset/sicdb-ext/ CSV outputs.
Usage: python data/sicdb-ext/predata/generate_example.py [--hadm-id ID ...]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import config

DEFAULT_HADM_IDS = ["100023", "100027", "100130"]


def _nan_to_none(v):
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, str) and v.lower() in ("nan", "none", ""):
        return None
    return v


def _load_table(name: str) -> pd.DataFrame:
    path = Path(config.BASE_OUT) / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _filter_hadm(df: pd.DataFrame, hadm_id: int) -> pd.DataFrame:
    if df.empty or "hadm_id" not in df.columns:
        return pd.DataFrame()
    return df[df["hadm_id"] == hadm_id].copy()


def build_case(hadm_id: int, dx_index: dict) -> dict:
    key = str(hadm_id)
    primary = dx_index[key]
    subject_id = int(primary["subject_id"])

    demo_df = _filter_hadm(_load_table("patient_demographics.csv"), hadm_id)
    demo = demo_df.iloc[0].to_dict() if not demo_df.empty else {}

    hpi_df = _filter_hadm(_load_table("history_of_present_illness.csv"), hadm_id)
    pe_df = _filter_hadm(_load_table("physical_examination.csv"), hadm_id)
    course_df = _filter_hadm(_load_table("hospital_course.csv"), hadm_id)
    ddx_df = _filter_hadm(_load_table("discharge_diagnosis.csv"), hadm_id)
    dpr_df = _filter_hadm(_load_table("discharge_procedures.csv"), hadm_id)
    labs_df = _filter_hadm(_load_table("laboratory_tests.csv"), hadm_id)
    meds_df = _filter_hadm(_load_table("discharge_medications.csv"), hadm_id)
    icd_dx_df = _filter_hadm(_load_table("icd_diagnosis.csv"), hadm_id)
    icd_proc_df = _filter_hadm(_load_table("icd_procedures.csv"), hadm_id)

    labs = [
        {
            "itemid": int(r["itemid"]) if pd.notna(r.get("itemid")) else None,
            "valuestr": _nan_to_none(r.get("valuestr")),
            "flag": _nan_to_none(r.get("flag")),
            "ref_range_lower": _nan_to_none(r.get("ref_range_lower")),
            "ref_range_upper": _nan_to_none(r.get("ref_range_upper")),
        }
        for _, r in labs_df.iterrows()
    ]

    medications = [
        {
            "drug": _nan_to_none(r.get("drug")),
            "prod_strength": _nan_to_none(r.get("prod_strength")),
            "dose_val_rx": _nan_to_none(r.get("dose_val_rx")),
            "dose_unit_rx": _nan_to_none(r.get("dose_unit_rx")),
            "route": _nan_to_none(r.get("route")),
            "doses_per_24_hrs": _nan_to_none(r.get("doses_per_24_hrs")),
        }
        for _, r in meds_df.iterrows()
    ]

    icd_diagnoses = [
        {
            "seq_num": int(r["seq_num"]) if pd.notna(r.get("seq_num")) else None,
            "icd_code": _nan_to_none(r.get("icd_code")),
            "icd_version": int(r["icd_version"]) if pd.notna(r.get("icd_version")) else None,
            "icd_diagnosis": _nan_to_none(r.get("icd_diagnosis")),
        }
        for _, r in icd_dx_df.sort_values("seq_num").iterrows()
    ]

    icd_procedures = [
        {
            "icd_code": _nan_to_none(r.get("icd_code")),
            "icd_title": _nan_to_none(r.get("icd_title")),
            "icd_version": int(r["icd_version"]) if pd.notna(r.get("icd_version")) else None,
        }
        for _, r in icd_proc_df.iterrows()
    ]

    discharge_procedures = []
    if not dpr_df.empty and "discharge_procedure" in dpr_df.columns:
        discharge_procedures = [
            str(v) for v in dpr_df["discharge_procedure"].dropna().tolist() if str(v).strip()
        ]

    return {
        "hadm_id": hadm_id,
        "subject_id": subject_id,
        "demographics": {
            "gender": _nan_to_none(demo.get("gender")),
            "anchor_age": int(demo["anchor_age"]) if pd.notna(demo.get("anchor_age")) else None,
            "race": _nan_to_none(demo.get("race")),
            "marital_status": _nan_to_none(demo.get("marital_status")),
            "insurance": _nan_to_none(demo.get("insurance")),
            "admission_type": _nan_to_none(demo.get("admission_type")),
            "admission_location": _nan_to_none(demo.get("admission_location")),
            "discharge_location": _nan_to_none(demo.get("discharge_location")),
            "hospital_expire_flag": int(demo["hospital_expire_flag"]) if pd.notna(demo.get("hospital_expire_flag")) else None,
        },
        "primary_diagnosis": {
            "icd_code": primary.get("icd_code"),
            "icd_version": primary.get("icd_version"),
            "long_title": primary.get("long_title"),
        },
        "clinical_notes": {
            # discharge_diagnosis is legitimately empty for most SICdb cases
            # (only populated when DiagnosisT2 differs from the primary
            # diagnosis) — pandas reads that empty CSV cell back as NaN, so
            # it must go through _nan_to_none to render as "" rather than NaN.
            "hpi": _nan_to_none(hpi_df.iloc[0]["hpi"]) if not hpi_df.empty else "",
            "pe": _nan_to_none(pe_df.iloc[0]["pe"]) if not pe_df.empty else "",
            "hospital_course": _nan_to_none(course_df.iloc[0]["hospital_course"]) if not course_df.empty else "",
            "discharge_diagnosis": (_nan_to_none(ddx_df.iloc[0]["discharge_diagnosis"]) or "") if not ddx_df.empty else "",
            "discharge_procedures": discharge_procedures,
        },
        "labs": labs,
        "microbiology": [],
        "medications": medications,
        "icd_diagnoses": icd_diagnoses,
        "icd_procedures": icd_procedures,
    }


_COMMENTS = {
    "hadm_id": "ICU 入院 ID（CaseID，数据集主键）",
    "subject_id": "患者 ID（PatientID，识别再入院）",
    "demographics": "患者基本信息 + 本次入院元数据",
    "gender": "性别",
    "anchor_age": "入院时年龄（5 岁分档）",
    "race": "种族（SICdb 未提供）",
    "marital_status": "婚姻状况",
    "insurance": "保险",
    "admission_type": "手术/入院类型",
    "admission_location": "转诊科室",
    "discharge_location": "出院科室",
    "hospital_expire_flag": "是否死亡（1 年随访）",
    "primary_diagnosis": "主诊断（cases.ICD10Main，评估 ground truth）",
    "icd_code": "ICD-10 编码",
    "icd_version": "ICD 版本（10）",
    "long_title": "ICD 诊断名称（德文）",
    "clinical_notes": "组装的临床文本（仅「已接受」病例有）",
    "hpi": "现病史（入院信息 + 主诊断 + 既往史 + SAPS III）",
    "pe": "入 ICU 24h 内生命体征 / 呼吸机参数",
    "hospital_course": "住院经过（管路/设备 + 干预 + 住院天数）",
    "discharge_diagnosis": "出院诊断",
    "discharge_procedures": "管路/设备列表（data_range 代理）",
    "labs": "实验室检查（每 LaboratoryID 首次结果）",
    "itemid": "检验项目 ID（= LaboratoryID）",
    "valuestr": "结果值（含单位）",
    "flag": "异常标记",
    "ref_range_lower": "参考范围下限",
    "ref_range_upper": "参考范围上限",
    "microbiology": "微生物（SICdb 无独立表）",
    "medications": "ICU 期间全部用药",
    "drug": "药品名称",
    "prod_strength": "规格",
    "dose_val_rx": "剂量",
    "dose_unit_rx": "剂量单位",
    "route": "给药途径",
    "doses_per_24_hrs": "每日次数",
    "icd_diagnoses": "全部 ICD 诊断",
    "seq_num": "诊断序号",
    "icd_diagnosis": "诊断名称",
    "icd_procedures": "管路/设备事件（data_range）",
    "icd_title": "操作/设备名称",
}

COMMENT_COL = 68


def _align(prefix: str, comment: str = "", *, comma: bool = False) -> str:
    text = prefix + ("," if comma else "")
    if not comment:
        return text
    if len(text) >= COMMENT_COL:
        return f"{text}  // {comment}"
    return f"{text}{' ' * (COMMENT_COL - len(text))}// {comment}"


def _render_field(indent: int, key: str, val, comment: str = "", *, comma: bool = True) -> str:
    sp = " " * indent
    js = json.dumps(val, ensure_ascii=False)
    return _align(f'{sp}"{key}": {js}', comment, comma=comma)


def _render_open(indent: int, key: str, open_char: str, comment: str = "") -> str:
    sp = " " * indent
    return _align(f'{sp}"{key}": {open_char}', comment, comma=False)


def _render_object_lines(indent: int, items: list[tuple[str, object]], comments: dict[str, str]) -> list[str]:
    lines = []
    for i, (k, v) in enumerate(items):
        lines.append(_render_field(indent, k, v, comments.get(k, ""), comma=(i < len(items) - 1)))
    return lines


def _render_array_of_objects(indent: int, items: list[dict], field_comments: dict[str, str]) -> list[str]:
    lines: list[str] = []
    obj_indent = indent + 2
    field_indent = indent + 4
    for i, item in enumerate(items):
        lines.append(f'{" " * obj_indent}{{')
        keys = list(item.keys())
        for j, k in enumerate(keys):
            lines.append(_render_field(field_indent, k, item[k], field_comments.get(k, ""), comma=(j < len(keys) - 1)))
        lines.append(f'{" " * obj_indent}}}{"," if i < len(items) - 1 else ""}')
    return lines


def render_case_jsonc(case: dict, case_num: int) -> str:
    pd_ = case["primary_diagnosis"]
    header = [
        "// " + "=" * 72,
        f"// 样例 {case_num}  hadm_id={case['hadm_id']}  subject_id={case['subject_id']}",
        f"// 主诊断: [{pd_.get('icd_version')}] {pd_.get('icd_code')} — {pd_.get('long_title')}",
        f"// labs={len(case['labs'])}, micro={len(case['microbiology'])}, meds={len(case['medications'])}, "
        f"icd_dx={len(case['icd_diagnoses'])}, icd_proc={len(case['icd_procedures'])}",
        "// " + "=" * 72,
    ]
    return "\n".join(header) + "\n" + _render_case_with_comments(case)


def _render_case_with_comments(case: dict) -> str:
    lines = ["{"]
    for k, c in [("hadm_id", _COMMENTS["hadm_id"]), ("subject_id", _COMMENTS["subject_id"])]:
        lines.append(_render_field(2, k, case[k], c))

    lines.append(_render_open(2, "demographics", "{", _COMMENTS["demographics"]))
    lines.extend(_render_object_lines(4, list(case["demographics"].items()), _COMMENTS))
    lines.append("  },")

    lines.append(_render_open(2, "primary_diagnosis", "{", _COMMENTS["primary_diagnosis"]))
    lines.extend(_render_object_lines(4, list(case["primary_diagnosis"].items()), _COMMENTS))
    lines.append("  },")

    lines.append(_render_open(2, "clinical_notes", "{", _COMMENTS["clinical_notes"]))
    cn = case["clinical_notes"]
    for i, k in enumerate(["hpi", "pe", "hospital_course", "discharge_diagnosis", "discharge_procedures"]):
        lines.append(_render_field(4, k, cn[k], _COMMENTS[k], comma=(i < 4)))
    lines.append("  },")

    for section, comment in [
        ("labs", _COMMENTS["labs"]),
        ("microbiology", _COMMENTS["microbiology"]),
        ("medications", _COMMENTS["medications"]),
        ("icd_diagnoses", _COMMENTS["icd_diagnoses"]),
        ("icd_procedures", _COMMENTS["icd_procedures"]),
    ]:
        suffix = "" if section == "icd_procedures" else ","
        lines.append(_render_open(2, section, "[", comment))
        if case[section]:
            lines.extend(_render_array_of_objects(2, case[section], _COMMENTS))
        lines.append(f"  ]{suffix}")

    lines.append("}")
    return "\n".join(lines)


def main(hadm_ids: list[str] | None = None):
    hadm_ids = hadm_ids or DEFAULT_HADM_IDS
    dx_path = Path(config.OUT_DX_INDEX)
    if not dx_path.exists():
        raise SystemExit(f"Missing {dx_path}; run predata/run.py first.")
    dx_index = json.load(open(dx_path))

    parts = [
        "// SICdb-Ext 数据集样例（JSONC 格式，// 为行尾注释）",
        f"// 从 dataset/sicdb-ext/ 处理结果中提取的 {len(hadm_ids)} 个完整病例",
        "// 字段与 CSV 输出一一对应，文本内容无损保留",
        "// 注：无 radiology_reports / microbiology；诊断为 ICD-10（德文）",
        "",
    ]
    case_num = 0
    for hid in hadm_ids:
        if hid not in dx_index:
            print(f"Warning: hadm_id {hid} not in index, skipping")
            continue
        case_num += 1
        parts.append(render_case_jsonc(build_case(int(hid), dx_index), case_num))
        parts.append("")

    out_path = _HERE / "example-json.txt"
    out_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hadm-id", action="append", dest="hadm_ids", help="CaseID / hadm_id")
    args = parser.parse_args()
    main(args.hadm_ids)
