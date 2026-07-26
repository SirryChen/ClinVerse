"""Lab Provider: resolves test orders to real lab results via a dataset
adapter's lab/radiology catalog (core.ports.DatasetAdapter) — dataset-agnostic
by construction, since it never reads a catalog file or itemid scheme itself.

When a requested test has no real result in the case, the fine-tuned
lab-provider model (LoRA adapter on Qwen) is used to predict the value
from the patient's full omniscient clinical context — matching the
prompt format used during training (prepare_data.py). That model is
MIMIC-trained today (train/provider/mimic-iv-ext/); for datasets without an
equivalent trained adapter (e.g. sicdb-ext), simply omit adapter_path /
model_path / lab_mapping_path and prediction degrades to "not performed".
"""

from __future__ import annotations

import logging
import inspect
import math
import re
from pathlib import Path

from core.json_catalog import LAB_NAME_ZH as _LAB_NAME_ZH
from core.ports import DatasetAdapter

from .base import BaseProvider

log = logging.getLogger(__name__)

_MODEL_CACHE: dict[tuple[str, str | None, str | None], tuple[object, object]] = {}
_FAILED_MODEL_KEYS: set[tuple[str, str | None, str | None]] = set()

# ── Training-consistent constants ──────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a clinical laboratory assistant with complete access to a patient's medical record. "
    "Given the patient's full clinical information and the specified laboratory test, "
    "predict the final test result. "
    "Respond with ONLY the numeric result (number only). Do not include units, labels, or explanations."
)

# Truncation limits matching config.yaml training settings
_HPI_MAX_CHARS = 500
_PE_MAX_CHARS = 2000
_HC_MAX_CHARS = 6000

_ABNORMAL_LABEL_ZH = {"high": "↑偏高", "low": "↓偏低", "abnormal": "↑异常", "delta": "↓偏低", "": ""}
_NOT_PERFORMED_ZH = "本病例未执行此检查"
_NOT_AVAILABLE_ZH = "本病例不支持此项检查"
_NOT_RECOGNIZED_ZH = "检查项目无法识别"
_NO_REPORT_ZH = "本病例无相关影像报告"
_PREDICTED_TAG_ZH = "【AI预测】"
_PREDICTED_TAG_EN = "[AI predicted]"
_OUT_OF_SCOPE_HEADER_ZH = "超出本院检验科范围"
_OUT_OF_SCOPE_HEADER_EN = "Out of catalog scope"
_OUT_OF_SCOPE_IMAGING_HEADER_ZH = "超出本院影像科范围"
_OUT_OF_SCOPE_IMAGING_HEADER_EN = "Out of imaging catalog scope"
# Whole hospital system has no radiology department at all (e.g. sicdb-ext) —
# distinct from "this particular study isn't in the catalog" (out-of-scope)
# or "catalog has the study but this case has no matching report" (_NO_REPORT_ZH).
_NO_IMAGING_DEPARTMENT_ZH = "本院无影像科，无法开具影像检查"
_NO_IMAGING_DEPARTMENT_EN = "This hospital system has no radiology department; imaging cannot be ordered."
_CHAT_TURN_END_TOKENS = ("<|im_end|>", "<|eot_id|>", "<|end|>")
_ROLE_MARKER_RE = re.compile(r"(?im)^\s*(user|assistant|system)\s*$")
_THINK_BLOCK_RE = re.compile(r"(?is)<think>.*?</think>")


# ── Prompt-building helpers (must match prepare_data.py exactly) ────────────────

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


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def _apply_chat_template(tokenizer, messages: list[dict], *, enable_thinking: bool = False, **kwargs) -> str:
    """Apply the same no-thinking chat template contract used by provider evaluation."""
    sig = inspect.signature(tokenizer.apply_chat_template)
    accepts_template_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in sig.parameters.values()
    )
    if "enable_thinking" in sig.parameters or accepts_template_kwargs:
        kwargs["enable_thinking"] = enable_thinking
    try:
        text = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        text = tokenizer.apply_chat_template(messages, **kwargs)
    if not isinstance(text, str):
        raise TypeError(f"apply_chat_template returned {type(text).__name__}, expected str")
    return text


def _resolve_eos_ids(tokenizer) -> list[int]:
    ids = set()
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    vocab = tokenizer.get_vocab()
    for token in _CHAT_TURN_END_TOKENS:
        if token in vocab:
            ids.add(vocab[token])
    return sorted(ids)


def _parse_model_output(text: str) -> str | None:
    """Strip legacy tags and truncate hallucinated follow-up chat turns."""
    text = re.sub(r"\[ABNORMAL\]", "", text, flags=re.IGNORECASE).strip()
    text = _THINK_BLOCK_RE.sub("", text).strip()
    match = _ROLE_MARKER_RE.search(text)
    if match:
        text = text[:match.start()].strip()
    text = text.replace("<think>", "").replace("</think>", "").strip()
    return text or None


# ── Prompt builder (replicates prepare_data.py::_build_user_prompt) ────────────

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
) -> str:
    lines: list[str] = []

    # 1. Target (top)
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

    # 2. Demographics
    lines.append("\n## Patient Profile")
    lines.append(
        f"- Age: {demo.get('anchor_age', '?')} | "
        f"Sex: {demo.get('gender', '?')} | "
        f"Race: {demo.get('race', '?')}"
    )
    lines.append(
        f"- Admission type: {demo.get('admission_type', '?')} | "
        f"Insurance: {demo.get('insurance', '?')}"
    )
    adm_loc = demo.get("admission_location", "")
    dis_loc = demo.get("discharge_location", "")
    if not _is_missing(adm_loc):
        lines.append(f"- Admitted from: {adm_loc}")
    if not _is_missing(dis_loc):
        lines.append(f"- Discharge destination: {dis_loc}")
    if demo.get("hospital_expire_flag") == 1:
        lines.append("- Hospital expiry: YES")

    # 3. Discharge Diagnosis
    dd_text = _truncate(discharge_diagnosis, 500)
    if dd_text:
        lines.append("\n## Discharge Diagnosis")
        lines.append(dd_text)

    # 4. ICD Diagnoses / Comorbidities
    valid_dx = [d for d in diagnoses if not _is_missing(d.get("icd_diagnosis"))]
    if valid_dx:
        lines.append("\n## Diagnoses / Comorbidities")
        for dx in valid_dx:
            seq   = dx.get("seq_num", "")
            title = dx["icd_diagnosis"]
            code  = dx.get("icd_code", "")
            ver   = dx.get("icd_version", "")
            prefix = f"[#{seq}] " if not _is_missing(seq) else ""
            suffix = f" (ICD-{ver}: {code})" if not _is_missing(code) else ""
            lines.append(f"- {prefix}{title}{suffix}")

    # 5. HPI
    hpi_text = _truncate(hpi, _HPI_MAX_CHARS)
    if hpi_text:
        lines.append("\n## History of Present Illness")
        lines.append(hpi_text)

    # 6. Physical Examination
    pe_text = _truncate(pe, _PE_MAX_CHARS)
    if pe_text:
        lines.append("\n## Physical Examination")
        lines.append(pe_text)

    # 7. Hospital Course
    hc_text = _truncate(hospital_course, _HC_MAX_CHARS)
    if hc_text:
        lines.append("\n## Hospital Course")
        lines.append(hc_text)

    # 8. Medications
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

    # 9. Microbiology
    valid_micro = [m for m in microbiology if not _is_missing(m.get("valuestr"))]
    if valid_micro:
        lines.append("\n## Microbiology")
        for mic in valid_micro:
            test   = mic.get("test_name", "")
            spec   = mic.get("spec_type_desc", "")
            result = mic.get("valuestr", "")
            abx    = mic.get("antibiogram", "")
            label_str = f"{test} ({spec})" if not _is_missing(spec) else test
            line = f"- {label_str}: {result}"
            if not _is_missing(abx):
                line += f" | Antibiogram: {abx}"
            lines.append(line)

    # 10. Available Laboratory Results (context; placed last)
    if context_labs:
        lines.append("\n## Available Laboratory Results")
        for lab in context_labs:
            flag = _fmt_flag(lab.get("flag"))
            lines.append(f"- {lab['label']}: {lab['valuestr']}{flag}")

    return "\n".join(lines)


# ── LabProvider ────────────────────────────────────────────────────────────────

class LabProvider(BaseProvider):
    """Dataset-agnostic by construction: all catalog/itemid-scheme knowledge
    lives behind `adapter` (core.ports.DatasetAdapter) — see
    adapters/mimic_iv_ext/ and adapters/sicdb_ext/ for the two plugins."""

    def __init__(
        self,
        adapter: DatasetAdapter,
        language: str = "en",
        adapter_path: str | Path | None = None,
        model_path: str | Path | None = None,
        lab_mapping_path: str | Path | None = None,
        local_device: str | None = None,
        local_batch_size: int = 4,
        local_generation_timeout: float | None = 60,
        llm_client=None,
        llm_model: str | None = None,
        llm_extra_body: dict | None = None,
    ):
        self._adapter = adapter
        self._language = language

        # Model prediction config (lazy-loaded)
        self._adapter_path = str(adapter_path) if adapter_path else None
        self._model_path   = str(model_path)   if model_path   else None
        self._local_device = local_device
        self._local_batch_size = max(1, int(local_batch_size))
        self._model        = None
        self._tokenizer    = None
        self._local_generation_timeout = local_generation_timeout

        # Chat-API fallback for missing-lab prediction, used when no local
        # fine-tuned adapter is configured (e.g. our own trained model isn't
        # deployed yet) — same prompt contract as the local model, just a
        # different backend for _predict_lab().
        self._llm_client = llm_client
        self._llm_model = llm_model
        self._llm_extra_body = llm_extra_body or {}

        # Lab mapping: itemid → training label / category (needed for prompt & context)
        self._id_map: dict[str, str] = {}
        self._item_category_map: dict[str, str] = {}
        if lab_mapping_path and (self._adapter_path or self._model_path):
            import pandas as pd
            mapping = pd.read_csv(lab_mapping_path, dtype={"itemid": str})
            self._id_map = dict(zip(mapping["itemid"].astype(str), mapping["label"]))
            self._item_category_map = dict(
                zip(mapping["itemid"].astype(str), mapping["category"].fillna(""))
            )
            log.info("Loaded %d itemid mappings for lab prediction", len(self._id_map))

    # ── Public interface ────────────────────────────────────────────────────────

    def respond(self, action_args: dict, case: dict) -> tuple[str, dict]:
        raise NotImplementedError("Use order_tests() or order_imaging() directly.")

    def order_tests(self, action_args: dict, case: dict) -> tuple[str, dict]:
        tests = action_args.get("tests", [])
        view = case["simulation_views"]["lab_provider_view"]
        labs_by_itemid: dict[str, list[dict]] = view.get("labs_by_itemid", {})
        micro_list: list[dict] = view.get("microbiology", [])
        catalog = self._adapter.lab_catalog

        resolved_items: list[dict] = []
        out_of_scope: list[str] = []  # not in catalog at all

        for test in tests:
            panel_items = catalog.resolve_panel(test)
            if panel_items is not None:
                for item in panel_items:
                    if item not in resolved_items:
                        resolved_items.append(item)
                continue
            item = catalog.resolve(test)
            if item is not None:
                if item not in resolved_items:
                    resolved_items.append(item)
                continue
            t_norm = _normalize(test)
            if any(kw in t_norm for kw in ["culture", "blood_culture", "urine_culture", "sputum_culture"]):
                resolved_items.append({"canonical_name": t_norm, "itemids": [], "source_table": "microbiology"})
                continue
            out_of_scope.append(test)

        result_lines: list[str] = []
        returned_evidence_ids: list[str] = []
        returned_abnormal_ids: list[str] = []
        predicted_by_key = self._predict_missing_labs_batch(resolved_items, labs_by_itemid, case)

        for item in resolved_items:
            if item.get("source_table") == "microbiology":
                lines, eids, abns = self._format_micro(item["canonical_name"], micro_list, case)
            else:
                lines, eids, abns = self._format_lab(
                    item,
                    labs_by_itemid,
                    predicted=self._predicted_value_for_item(item, predicted_by_key),
                )
            result_lines.extend(lines)
            returned_evidence_ids.extend(eids)
            returned_abnormal_ids.extend(abns)

        # Out-of-scope block — clearly separated for transcript analysis
        if out_of_scope:
            n = len(out_of_scope)
            names = ", ".join(out_of_scope)
            if self._language == "zh":
                result_lines.append(
                    f"\n[{_OUT_OF_SCOPE_HEADER_ZH} ({n}项)：{names}]"
                )
            else:
                result_lines.append(
                    f"\n[{_OUT_OF_SCOPE_HEADER_EN} ({n}): {names}]"
                )

        if self._language == "zh":
            response = "\n".join(result_lines) if result_lines else "所请求的检查项目暂无结果。"
        else:
            response = "\n".join(result_lines) if result_lines else "No results available for requested tests."
        metadata = {
            "provider": "lab_provider",
            "requested_tests": tests,
            "returned_evidence_ids": returned_evidence_ids,
            "returned_abnormal_evidence_ids": returned_abnormal_ids,
            "out_of_scope_tests": out_of_scope,
        }
        return response, metadata

    def order_imaging(self, action_args: dict, case: dict) -> tuple[str, dict]:
        studies = action_args.get("studies", [])

        if not self._adapter.has_radiology:
            # Distinct from "out of scope" or "no report in this case" — this
            # hospital system has no radiology department at all (e.g.
            # sicdb-ext), so don't pretend the studies were even looked up.
            msg = _NO_IMAGING_DEPARTMENT_ZH if self._language == "zh" else _NO_IMAGING_DEPARTMENT_EN
            return msg, {
                "provider": "lab_provider",
                "requested_studies": studies,
                "returned_evidence_ids": [],
                "returned_abnormal_evidence_ids": [],
                "out_of_scope_studies": [],
            }

        view = case["simulation_views"]["lab_provider_view"]
        radiology_list: list[dict] = view.get("radiology", [])
        rad_catalog = self._adapter.radiology_catalog

        result_lines: list[str] = []
        returned_evidence_ids: list[str] = []
        out_of_scope_studies: list[str] = []  # not in radiology catalog at all

        for study in studies:
            study_def = rad_catalog.resolve(study)
            if not study_def:
                out_of_scope_studies.append(study)
                continue

            modality = study_def["modality"]
            regions = study_def["regions"]
            matches = [
                r for r in radiology_list
                if r.get("modality") == modality and r.get("region", "") in regions
            ]
            if not matches:
                if self._language == "zh":
                    result_lines.append(f"[{study_def['display_name']}：{_NO_REPORT_ZH}]")
                else:
                    result_lines.append(f"[{study_def['display_name']}: no report found in this case]")
                continue

            for rad in matches:
                note_id = rad.get("note_id", "")
                eid = f"mimic_rad_{note_id or _normalize(rad.get('exam_name',''))}"
                if self._language == "zh":
                    result_lines.append(f"\n=== 影像报告：{rad.get('exam_name', study_def['display_name'])} ===")
                else:
                    result_lines.append(f"\n=== {rad.get('exam_name', study_def['display_name'])} ===")
                result_lines.append(rad.get("text", "Report text not available.").strip())
                returned_evidence_ids.append(eid)

        # Out-of-scope imaging block
        if out_of_scope_studies:
            n = len(out_of_scope_studies)
            names = ", ".join(out_of_scope_studies)
            if self._language == "zh":
                result_lines.append(f"\n[{_OUT_OF_SCOPE_IMAGING_HEADER_ZH} ({n}项)：{names}]")
            else:
                result_lines.append(f"\n[{_OUT_OF_SCOPE_IMAGING_HEADER_EN} ({n}): {names}]")

        if self._language == "zh":
            response = "\n".join(result_lines) if result_lines else "所请求的影像检查暂无结果。"
        else:
            response = "\n".join(result_lines) if result_lines else "No imaging results available."
        return response, {
            "provider": "lab_provider",
            "requested_studies": studies,
            "returned_evidence_ids": returned_evidence_ids,
            "returned_abnormal_evidence_ids": [],
            "out_of_scope_studies": out_of_scope_studies,
        }

    # ── Lab formatting ──────────────────────────────────────────────────────────

    def _format_lab(
        self,
        item: dict,
        labs_by_itemid: dict[str, list[dict]],
        predicted: str | None = None,
    ) -> tuple[list[str], list[str], list[str]]:
        lines: list[str] = []
        eids: list[str] = []
        abns: list[str] = []
        canonical = item.get("canonical_name", "")
        en_name = item.get("display_name", canonical)
        if self._language == "zh":
            name = _LAB_NAME_ZH.get(canonical, en_name)
        else:
            name = en_name

        for iid in item.get("itemids", []):
            rows = labs_by_itemid.get(str(iid), [])
            if rows:
                row = rows[0]
                flag = str(row.get("flag", ""))
                if self._language == "zh":
                    flag_str = "  ↑偏高" if "abnormal" in flag.lower() else ("  ↓偏低" if "delta" in flag.lower() else "")
                else:
                    flag_str = " ↑" if "abnormal" in flag.lower() else (" ↓" if "delta" in flag.lower() else "")
                sep = "：" if self._language == "zh" else ": "
                lines.append(f"{name}{sep}{row.get('valuestr', 'N/A')}{flag_str}")
                eid = f"mimic_lab_{iid}"
                eids.append(eid)
                if flag_str:
                    abns.append(eid)
                return lines, eids, abns

        # No real result — use any precomputed model prediction
        sep = "：" if self._language == "zh" else ": "
        if predicted is not None:
            tag = _PREDICTED_TAG_ZH if self._language == "zh" else _PREDICTED_TAG_EN
            lines.append(f"{name}{sep}{predicted} {tag}")
        else:
            if self._language == "zh":
                lines.append(f"{name}：{_NOT_PERFORMED_ZH}")
            else:
                lines.append(f"{name}: not performed in this case")

        return lines, eids, abns

    def _format_micro(
        self,
        test_type: str,
        micro_list: list[dict],
        case: dict,
    ) -> tuple[list[str], list[str], list[str]]:
        lines: list[str] = []
        eids: list[str] = []

        type_kw = test_type.replace("_culture", "").replace("_", " ")
        matching = (
            [m for m in micro_list if type_kw in (m.get("spec_type_desc", "") + m.get("test_name", "")).lower()]
            or micro_list
        )

        if not matching:
            if self._language == "zh":
                lines.append(f"[{test_type}：暂无微生物检测结果]")
            else:
                lines.append(f"[{test_type}: no result available]")
            return lines, eids, []

        for i, m in enumerate(matching[:3]):
            if self._language == "zh":
                lines.append(
                    f"微生物检测（{m.get('spec_type_desc', test_type)}）："
                    f"{m.get('test_name', '')} — {m.get('valuestr', '无菌生长')}"
                )
            else:
                lines.append(
                    f"Microbiology ({m.get('spec_type_desc', test_type)}): "
                    f"{m.get('test_name', '')} — {m.get('valuestr', 'No growth')}"
                )
            eids.append(f"mimic_micro_{i}")

        return lines, eids, []

    # ── Model prediction internals ──────────────────────────────────────────────

    def _ensure_model_loaded(self) -> bool:
        """Lazy-load the local prediction model once per process.

        Reuses a shared cache across LabProvider instances so batch eval does
        not reload the 2B model for every case.
        """
        if self._model is not None:
            return True
        if not self._model_path:
            return False
        cache_key = (self._model_path, self._adapter_path, self._local_device)
        if cache_key in _MODEL_CACHE:
            self._tokenizer, self._model = _MODEL_CACHE[cache_key]
            return True
        if cache_key in _FAILED_MODEL_KEYS:
            return False
        try:
            import torch
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

            log.info("Loading base model from %s", self._model_path)
            tokenizer = AutoTokenizer.from_pretrained(
                self._model_path, trust_remote_code=True, padding_side="left"
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model_config = AutoConfig.from_pretrained(
                self._model_path,
                trust_remote_code=True,
            )
            # Qwen3.5 checkpoints are multimodal and declare
            # Qwen3_5ForConditionalGeneration at the top level. The lab
            # predictor is text-only, so load its text sub-config explicitly
            # instead of relying on AutoModelForCausalLM to infer a partial
            # conversion from the full vision-language configuration.
            if hasattr(model_config, "text_config"):
                model_config = model_config.text_config

            base_model = AutoModelForCausalLM.from_pretrained(
                self._model_path,
                config=model_config,
                device_map={"": self._local_device} if self._local_device else "auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            if self._adapter_path:
                from peft import PeftModel

                log.info("Loading LoRA adapter from %s", self._adapter_path)
                model = PeftModel.from_pretrained(base_model, self._adapter_path)
            else:
                log.info("Using merged local lab model from %s", self._model_path)
                model = base_model
            model.eval()

            self._tokenizer = tokenizer
            self._model = model
            _MODEL_CACHE[cache_key] = (tokenizer, model)
            return True
        except Exception as e:
            _FAILED_MODEL_KEYS.add(cache_key)
            log.warning("Failed to load lab prediction model: %s", e)
            return False

    def _get_target_label_and_category(self, item: dict) -> tuple[str, str]:
        """Return (label, category) using training-consistent id_map when available."""
        for iid in item.get("itemids", []):
            iid_str = str(iid)
            if iid_str in self._id_map:
                return (
                    self._id_map[iid_str],
                    self._item_category_map.get(iid_str, item.get("category", "")),
                )
        return item.get("display_name", item.get("canonical_name", "")), item.get("category", "")

    def _build_context_labs(self, labs_by_itemid: dict[str, list[dict]]) -> list[dict]:
        """Build the context lab list from available real results (matches training format)."""
        context: list[dict] = []
        for iid_str, rows in labs_by_itemid.items():
            if not rows or iid_str not in self._id_map:
                continue
            row = rows[0]
            if _is_missing(row.get("valuestr")):
                continue
            context.append({
                "label":    self._id_map[iid_str],
                "valuestr": str(row["valuestr"]),
                "flag":     row.get("flag"),
            })
        return context

    def _item_prediction_key(self, item: dict) -> str:
        for iid in item.get("itemids", []):
            iid_str = str(iid)
            if iid_str in self._id_map:
                return iid_str
        return item.get("canonical_name", item.get("display_name", ""))

    def _predicted_value_for_item(self, item: dict, predicted_by_key: dict[str, str | None]) -> str | None:
        return predicted_by_key.get(self._item_prediction_key(item))

    def _has_real_lab_result(self, item: dict, labs_by_itemid: dict[str, list[dict]]) -> bool:
        return any(labs_by_itemid.get(str(iid), []) for iid in item.get("itemids", []))

    def _build_prediction_messages(
        self,
        item: dict,
        labs_by_itemid: dict[str, list[dict]],
        case: dict,
    ) -> tuple[str, list[dict]] | None:
        # ── Gather patient context from the case (omniscient view) ──
        core = case.get("core_admission", {})
        moderator = case.get("simulation_views", {}).get("moderator_view", {})

        demo = core.get("demographics", {})
        hpi = core.get("hpi", "")
        pe = core.get("physical_exam", "")
        medications = core.get("medications", [])
        microbiology = case.get("simulation_views", {}).get("lab_provider_view", {}).get("microbiology", [])
        hospital_course = moderator.get("hospital_course", "")
        discharge_diagnosis = moderator.get("discharge_diagnosis", "")
        icd_diagnoses = moderator.get("icd_diagnoses", [])

        target_label, target_category = self._get_target_label_and_category(item)
        if not target_label:
            return None

        context_labs = self._build_context_labs(labs_by_itemid)
        user_prompt = _build_user_prompt(
            demo=demo,
            hpi=hpi,
            pe=pe,
            diagnoses=icd_diagnoses,
            medications=medications,
            microbiology=microbiology,
            hospital_course=hospital_course,
            discharge_diagnosis=discharge_diagnosis,
            context_labs=context_labs,
            target_label=target_label,
            target_category=target_category,
            ref_range_lower=None,
            ref_range_upper=None,
            unit="",
        )
        return target_label, [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    def _predict_missing_labs_batch(
        self,
        items: list[dict],
        labs_by_itemid: dict[str, list[dict]],
        case: dict,
    ) -> dict[str, str | None]:
        missing_items = [
            item for item in items
            if item.get("source_table") != "microbiology" and not self._has_real_lab_result(item, labs_by_itemid)
        ]
        if not missing_items:
            return {}

        local_model_configured = bool(self._model_path and self._adapter_path)
        local_model_loaded = self._ensure_model_loaded() if local_model_configured else False
        if local_model_configured and not local_model_loaded:
            raise RuntimeError(
                "Local lab prediction model failed to load; refusing to fall back "
                "to missing results. Fix the configured model/adapter or GPU allocation."
            )
        use_local_model = bool(self._id_map) and local_model_loaded
        if not use_local_model and self._llm_client is None:
            return {}

        prompts: list[tuple[str, str, list[dict]]] = []
        for item in missing_items:
            built = self._build_prediction_messages(item, labs_by_itemid, case)
            if built is None:
                continue
            target_label, messages = built
            prompts.append((self._item_prediction_key(item), target_label, messages))
        if not prompts:
            return {}

        if use_local_model:
            predictions = self._run_model_inference_batch([messages for _, _, messages in prompts])
            failed_labels = [
                target_label
                for (_, target_label, _), prediction in zip(prompts, predictions)
                if prediction is None
            ]
            if failed_labels:
                raise RuntimeError(
                    "Batch lab prediction returned empty output for: "
                    + ", ".join(failed_labels)
                )
            return {
                key: prediction
                for (key, _, _), prediction in zip(prompts, predictions)
            }

        predictions: dict[str, str | None] = {}
        for key, target_label, messages in prompts:
            try:
                predictions[key] = self._run_llm_inference(messages)
            except Exception as e:
                raise RuntimeError(f"Lab prediction inference failed for {target_label}: {e}") from e
            if predictions[key] is None:
                raise RuntimeError(f"Lab prediction returned empty output for {target_label}")
        return predictions

    def _predict_lab(
        self,
        item: dict,
        labs_by_itemid: dict[str, list[dict]],
        case: dict,
    ) -> str | None:
        """Build an omniscient prompt and run the fine-tuned model (or, when no
        local adapter is configured, the chat-API fallback) to predict the
        result."""
        local_model_configured = bool(self._model_path and self._adapter_path)
        local_model_loaded = self._ensure_model_loaded() if local_model_configured else False
        if local_model_configured and not local_model_loaded:
            raise RuntimeError(
                "Local lab prediction model failed to load; refusing to fall back "
                "to missing results. Fix the configured model/adapter or GPU allocation."
            )
        use_local_model = bool(self._id_map) and local_model_loaded
        if not use_local_model and self._llm_client is None:
            return None

        # ── Gather patient context from the case (omniscient view) ──
        core = case.get("core_admission", {})
        moderator = case.get("simulation_views", {}).get("moderator_view", {})

        demo        = core.get("demographics", {})
        hpi         = core.get("hpi", "")
        pe          = core.get("physical_exam", "")
        medications = core.get("medications", [])
        microbiology = case.get("simulation_views", {}).get("lab_provider_view", {}).get("microbiology", [])
        hospital_course     = moderator.get("hospital_course", "")
        discharge_diagnosis = moderator.get("discharge_diagnosis", "")
        icd_diagnoses       = moderator.get("icd_diagnoses", [])

        # ── Target lab metadata ──
        target_label, target_category = self._get_target_label_and_category(item)
        if not target_label:
            return None

        # ref_range and unit are unavailable for missing tests → left empty
        context_labs = self._build_context_labs(labs_by_itemid)

        user_prompt = _build_user_prompt(
            demo=demo,
            hpi=hpi,
            pe=pe,
            diagnoses=icd_diagnoses,
            medications=medications,
            microbiology=microbiology,
            hospital_course=hospital_course,
            discharge_diagnosis=discharge_diagnosis,
            context_labs=context_labs,
            target_label=target_label,
            target_category=target_category,
            ref_range_lower=None,
            ref_range_upper=None,
            unit="",
        )

        messages = [
            {"role": "system",  "content": _SYSTEM_PROMPT},
            {"role": "user",    "content": user_prompt},
        ]

        try:
            if use_local_model:
                return self._run_model_inference(messages)
            return self._run_llm_inference(messages)
        except Exception as e:
            log.warning("Lab prediction inference failed for %s: %s", target_label, e)
            return None

    def _run_model_inference(self, messages: list[dict]) -> str | None:
        """Run model inference and return the predicted value string."""
        return self._run_model_inference_batch([messages])[0]

    def _run_model_inference_batch(self, messages_batch: list[list[dict]]) -> list[str | None]:
        """Run a batch of local model inferences and return one prediction per prompt."""
        import torch

        if not messages_batch:
            return []
        # A single broad test order can expand into dozens of missing lab
        # items. Each prompt is up to 4096 tokens, so processing all items in
        # one generate() call can exhaust GPU memory even for a 2B model.
        if len(messages_batch) > self._local_batch_size:
            predictions: list[str | None] = []
            for start in range(0, len(messages_batch), self._local_batch_size):
                predictions.extend(
                    self._run_model_inference_batch(
                        messages_batch[start:start + self._local_batch_size]
                    )
                )
            return predictions
        prompt_texts = [
            _apply_chat_template(
                self._tokenizer,
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for messages in messages_batch
        ]
        enc = self._tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(next(self._model.parameters()).device)
        prompt_len = enc["input_ids"].shape[1]

        with torch.no_grad():
            generate_kwargs = {
                "max_new_tokens": 32,
                "do_sample": False,
                "pad_token_id": self._tokenizer.pad_token_id,
                "eos_token_id": _resolve_eos_ids(self._tokenizer),
            }
            if self._local_generation_timeout is not None:
                generate_kwargs["max_time"] = self._local_generation_timeout
            out = self._model.generate(**enc, **generate_kwargs)

        predictions: list[str | None] = []
        for row_idx in range(len(messages_batch)):
            new_tokens = out[row_idx][prompt_len:]
            text = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            predictions.append(_parse_model_output(text))
        return predictions

    def _run_llm_inference(self, messages: list[dict]) -> str | None:
        """Chat-API fallback for missing-lab prediction (no local adapter
        configured) — same prompt contract as _run_model_inference()."""
        completion = self._llm_client.chat.completions.create(
            model=self._llm_model,
            messages=messages,
            temperature=0,
            max_tokens=32,
            extra_body=self._llm_extra_body,
        )
        text = (completion.choices[0].message.content or "").strip()
        return _parse_model_output(text)
