"""Exam Provider: returns physical examination findings on request (no LLM)."""

from __future__ import annotations

import re

from .base import BaseProvider

_COMPONENT_LABEL_ZH = {
    "general": "一般情况",
    "vital_signs": "生命体征",
    "heent": "头颈/五官",
    "cardiovascular": "心血管",
    "pulmonary": "肺部",
    "abdomen": "腹部",
    "extremities": "四肢",
    "neurological": "神经系统",
    "skin": "皮肤",
    "musculoskeletal": "肌肉骨骼",
}


class ExamProvider(BaseProvider):
    def __init__(self, language: str = "en"):
        self._language = language

    def respond(self, action_args: dict, case: dict) -> tuple[str, dict]:
        view = case["simulation_views"]["exam_provider_view"]
        pe_text = view.get("physical_exam_full", "")

        components = action_args.get("components", [])
        if components:
            text = self._filter_components(pe_text, components)
        else:
            if self._language == "zh":
                text = pe_text or "体格检查记录暂无。"
            else:
                text = pe_text or "Physical examination findings not available."

        if self._language == "zh" and text:
            labels = [_COMPONENT_LABEL_ZH.get(c.lower(), c) for c in components] if components else ["完整体检"]
            header = f"【体格检查结果 — {' / '.join(labels)}】\n"
            text = header + text

        evidence_ids = self._collect_evidence_ids(case, components)
        return text, {"provider": "exam_provider", "disclosed_evidence_ids": evidence_ids}

    def _filter_components(self, pe_text: str, components: list[str]) -> str:
        """Return PE sections relevant to requested components."""
        if not pe_text:
            return "体格检查未记录。" if self._language == "zh" else "Physical examination not documented."

        component_map = {
            "general": ["gen", "general", "appearance", "nad"],
            "vital_signs": ["vs", "vitals", "vital signs", "t ", "hr ", "bp ", "rr ", "o2"],
            "heent": ["heent", "head", "eyes", "ears", "nose", "throat"],
            "cardiovascular": ["card", "heart", "cardiovascular", "cv"],
            "pulmonary": ["pulm", "lungs", "respiratory", "breath"],
            "abdomen": ["abd", "abdomen", "abdominal", "belly"],
            "extremities": ["ext", "extremities", "limbs", "legs", "arms"],
            "neurological": ["neuro", "neurologic", "cns"],
            "skin": ["skin", "derm"],
            "musculoskeletal": ["msk", "musculoskeletal", "joints"],
        }

        requested_keywords: set[str] = set()
        for comp in components:
            comp_lower = comp.lower()
            for key, kws in component_map.items():
                if comp_lower in key or any(kw in comp_lower for kw in kws):
                    requested_keywords.update(kws)
            requested_keywords.add(comp_lower)

        sentences = re.split(r"(?<=[.:])\s+", pe_text)
        relevant = [s for s in sentences if any(kw in s.lower() for kw in requested_keywords)]
        return " ".join(relevant) if relevant else pe_text

    def _collect_evidence_ids(self, case: dict, components: list[str]) -> list[str]:
        return [
            item["evidence_id"]
            for item in case.get("evidence_inventory", [])
            if item.get("type") == "vital_signs" or item.get("type") == "pe"
        ]
