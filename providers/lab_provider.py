"""Lab Provider: resolves test orders to real lab results via a dataset
adapter's lab/radiology catalog (core.ports.DatasetAdapter) — dataset-agnostic
by construction, since it never reads a catalog file or itemid scheme itself.

When a requested test has no real result in the case, the fine-tuned
lab-provider model (LoRA adapter on Qwen) can predict the value from the
patient's full omniscient clinical context — matching the prompt format used
during training (prepare_data.py). Set ``clinlab_enabled=False`` for the
missingness-control condition: source-record results are still returned, but
missing results become an explicit "not performed" response without invoking
the prediction model.
"""

from __future__ import annotations
from core.lab_prompt import _build_user_prompt, _is_missing, _fmt_flag, _fmt_ref_range, _truncate, SYSTEM_PROMPT as _SYSTEM_PROMPT
from core.microbiology import is_abnormal as micro_abnormal, matches_specimen

import logging
import inspect
import hashlib
import math
import re
from pathlib import Path

from core.json_catalog import LAB_NAME_ZH as _LAB_NAME_ZH
from core.ports import DatasetAdapter
from core.api_retry import call_with_retry

from .base import BaseProvider

log = logging.getLogger(__name__)

_MODEL_CACHE: dict[tuple[str, str | None, str | None], tuple[object, object]] = {}
_FAILED_MODEL_KEYS: set[tuple[str, str | None, str | None]] = set()

# ── Training-consistent constants ──────────────────────────────────────────────


# Truncation limits matching config.yaml training settings

_ABNORMAL_LABEL_ZH = {"high": "↑偏高", "low": "↓偏低", "abnormal": "↑异常", "delta": "↓偏低", "": ""}
_NOT_PERFORMED_ZH = "本病例未执行此检查"
_NOT_AVAILABLE_ZH = "本病例不支持此项检查"
_NOT_RECOGNIZED_ZH = "检查项目无法识别"
_NO_REPORT_ZH = "本病例无相关影像报告"
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
    """Strip tags and truncate hallucinated follow-up chat turns."""
    text = re.sub(r"\[ABNORMAL\]", "", text, flags=re.IGNORECASE).strip()
    text = _THINK_BLOCK_RE.sub("", text).strip()
    match = _ROLE_MARKER_RE.search(text)
    if match:
        text = text[:match.start()].strip()
    text = text.replace("<think>", "").replace("</think>", "").strip()
    return text or None


# ── Prompt builder (replicates prepare_data.py::_build_user_prompt) ────────────



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
        mask_ratio: float = 0.0,
        mask_seed: int = 0,
        clinlab_enabled: bool = True,
    ):
        if not 0.0 <= mask_ratio <= 1.0:
            raise ValueError("mask_ratio must be between 0 and 1")
        self._adapter = adapter
        self._language = language
        self._mask_ratio = float(mask_ratio)
        self._mask_seed = int(mask_seed)
        self._clinlab_enabled = bool(clinlab_enabled)

        # Model prediction config (lazy-loaded)
        self._adapter_path = str(adapter_path) if adapter_path else None
        self._model_path   = str(model_path)   if model_path   else None
        self._local_device = local_device
        self._local_batch_size = max(1, int(local_batch_size))
        self._model        = None
        self._tokenizer    = None
        self._local_generation_timeout = local_generation_timeout

        self._llm_client = llm_client
        self._llm_model = llm_model
        self._llm_extra_body = llm_extra_body or {}

        # Lab mapping: itemid → training label / category (needed for prompt & context)
        self._id_map: dict[str, str] = {}
        self._item_category_map: dict[str, str] = {}
        if self._clinlab_enabled and lab_mapping_path and (self._adapter_path or self._model_path):
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
        masked_items = [
            item for item in resolved_items
            if item.get("source_table") != "microbiology"
            and self._has_real_lab_result(item, labs_by_itemid)
            and self._should_mask_real_lab(item, case)
        ]
        masked_keys = {self._item_prediction_key(item) for item in masked_items}
        predicted_by_key = {}
        if self._clinlab_enabled:
            predicted_by_key = self._predict_missing_labs_batch(
                resolved_items,
                labs_by_itemid,
                case,
                force_predict_keys=masked_keys,
            )

        for item in resolved_items:
            if item.get("source_table") == "microbiology":
                lines, eids, abns = self._format_micro(item["canonical_name"], micro_list, case)
            else:
                lines, eids, abns = self._format_lab(
                    item,
                    labs_by_itemid,
                    predicted=self._predicted_value_for_item(item, predicted_by_key),
                    mask_real=self._item_prediction_key(item) in masked_keys,
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
            "clinlab_enabled": self._clinlab_enabled,
            "predicted_test_keys": sorted(key for key, value in predicted_by_key.items() if value is not None),
            "failed_prediction_keys": sorted(key for key, value in predicted_by_key.items() if value is None),
            "lab_mask_ratio": self._mask_ratio,
            "masked_test_keys": sorted(masked_keys),
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
        mask_real: bool = False,
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

        if not mask_real:
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
            # Predictions must be indistinguishable from recorded values in
            # the doctor-visible response.  Their provenance remains
            # available only in hidden metadata (masked_test_keys).
            lines.append(f"{name}{sep}{predicted}")
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

        abnormal_ids = []
        for i, record in enumerate(micro_list):
            if not matches_specimen(record, test_type):
                continue
            result = record.get("valuestr")
            if not result:
                raise ValueError("Microbiology records require a result")
            eid = f"mimic_micro_{i}"
            lines.append(f"Microbiology ({record.get('spec_type_desc', test_type)}): {record.get('test_name', '')} — {result}")
            eids.append(eid)
            if micro_abnormal(record):
                abnormal_ids.append(eid)
        if not lines:
            lines.append(f"[{test_type}: no result available]")
        return lines, eids, abnormal_ids

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

    def _should_mask_real_lab(self, item: dict, case: dict) -> bool:
        """Make an episode-stable Bernoulli choice for each case/lab pair."""
        if self._mask_ratio <= 0.0:
            return False
        if self._mask_ratio >= 1.0:
            return True
        case_id = str(case.get("case_id", ""))
        item_key = self._item_prediction_key(item)
        payload = f"{self._mask_seed}:{case_id}:{item_key}".encode("utf-8")
        sample = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / 2**64
        return sample < self._mask_ratio

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

        # A masked target must not appear in its own ClinLab prompt.  This is
        # essential because, unlike a naturally missing test, its ground-truth
        # row is present in labs_by_itemid.
        target_itemids = {str(iid) for iid in item.get("itemids", [])}
        context_labs = self._build_context_labs({
            iid: rows for iid, rows in labs_by_itemid.items()
            if str(iid) not in target_itemids
        })
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
        force_predict_keys: set[str] | None = None,
    ) -> dict[str, str | None]:
        force_predict_keys = force_predict_keys or set()
        prediction_items = [
            item for item in items
            if item.get("source_table") != "microbiology"
            and (
                not self._has_real_lab_result(item, labs_by_itemid)
                or self._item_prediction_key(item) in force_predict_keys
            )
        ]
        if not prediction_items:
            return {}

        local_model_configured = bool(self._model_path)
        local_model_loaded = self._ensure_model_loaded() if local_model_configured else False
        if local_model_configured and not local_model_loaded:
            raise RuntimeError(
                "Local lab prediction model failed to load; refusing to fall back "
                "to missing results. Fix the configured model/adapter or GPU allocation."
            )
        use_local_model = bool(self._id_map) and local_model_loaded
        if not use_local_model and self._llm_client is None:
            raise RuntimeError("Configure a ClinLab model, mapping and endpoint")

        prompts: list[tuple[str, str, list[dict]]] = []
        for item in prediction_items:
            built = self._build_prediction_messages(item, labs_by_itemid, case)
            if built is None:
                raise ValueError("Requested test lacks ClinLab target metadata")
            target_label, messages = built
            prompts.append((self._item_prediction_key(item), target_label, messages))
        if not prompts:
            return {}

        if use_local_model:
            predictions = self._run_model_inference_batch([messages for _, _, messages in prompts])
        else:
            predictions = [self._run_llm_inference(messages) for _, _, messages in prompts]
        if len(predictions) != len(prompts) or any(value is None for value in predictions):
            raise RuntimeError("ClinLab must return a result for each requested prediction")
        return {key: value for (key, _, _), value in zip(prompts, predictions)}

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
        """Request a prediction from the configured ClinLab endpoint."""
        completion = call_with_retry(
            self._llm_client.chat.completions.create,
            model=self._llm_model,
            messages=messages,
            temperature=0,
            max_tokens=32,
            extra_body=self._llm_extra_body,
        )
        text = (completion.choices[0].message.content or "").strip()
        return _parse_model_output(text)
