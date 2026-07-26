"""TaskStream: an ordered, splittable list of cases for an external harness to
drive offline (fixed train/test split) or online (single streaming pass)
evaluation against ClinicalEnv. This module never talks to an LLM — it only
manages case ordering and dataset-level metadata.

CaseRef.category is derived from each case's ground-truth ICD-10 code via
core.icd.block(). It exists purely for the harness's own bookkeeping (train/
test stratification, per-category score breakdown) — see clinical_env.py's
module docstring for why this must never be surfaced inside an agent's
in-episode observation.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from core import icd as icd_mod


@dataclass(frozen=True)
class CaseRef:
    case_id: str
    path: Path
    category: Optional[str]
    source: Optional[str]


def _case_source(case: dict) -> Optional[str]:
    source = case.get("core_admission", {}).get("source")
    # Legacy pre-adapter cases recorded source as "mimic-iv"; mirrors the same
    # normalization in ClinicalEnv.from_case_file.
    if source == "mimic-iv":
        source = "mimic-iv-ext"
    return source


def _derive_category(case: dict) -> Optional[str]:
    gt = case.get("simulation_views", {}).get("moderator_view", {}).get("ground_truth", {})
    icd10 = gt.get("icd10_code")
    if not icd10:
        icd_code = gt.get("icd_code")
        icd_version = gt.get("icd_version")
        if icd_code and icd_version:
            icd10 = icd_mod.to_icd10(icd_code, icd_version)
    if not icd10:
        return None
    return icd_mod.block(icd10)


class TaskStream:
    def __init__(
        self,
        cases_dir: str | Path,
        only_usable: bool = True,
        source: Optional[str] = None,
        _refs: Optional[list[CaseRef]] = None,
    ):
        self.cases_dir = Path(cases_dir)
        self._refs = _refs if _refs is not None else self._load_refs(only_usable, source)

    def _load_refs(self, only_usable: bool, source: Optional[str]) -> list[CaseRef]:
        index_path = self.cases_dir / "index.jsonl"
        entries: list[dict] = []
        if index_path.exists():
            with open(index_path) as f:
                for line in f:
                    entry = json.loads(line)
                    if only_usable and not entry.get("usable", True):
                        continue
                    entries.append(entry)
        else:
            entries = [{"case_id": p.stem} for p in sorted(self.cases_dir.glob("*.json"))]

        refs: list[CaseRef] = []
        for entry in entries:
            case_id = entry["case_id"]
            path = self.cases_dir / f"{case_id}.json"
            if not path.exists():
                continue
            with open(path) as f:
                case = json.load(f)
            case_source = _case_source(case)
            if source is not None and case_source != source:
                continue
            refs.append(CaseRef(case_id=case_id, path=path, category=_derive_category(case), source=case_source))
        return refs

    def __iter__(self) -> Iterator[CaseRef]:
        return iter(self._refs)

    def __len__(self) -> int:
        return len(self._refs)

    def split(self, train_ratio: float = 0.5, seed: int = 42) -> tuple["TaskStream", "TaskStream"]:
        refs = list(self._refs)
        random.Random(seed).shuffle(refs)
        cut = round(len(refs) * train_ratio)
        train_refs, test_refs = refs[:cut], refs[cut:]
        return (
            TaskStream(self.cases_dir, _refs=train_refs),
            TaskStream(self.cases_dir, _refs=test_refs),
        )

    def by_category(self) -> dict[str, list[CaseRef]]:
        out: dict[str, list[CaseRef]] = {}
        for ref in self._refs:
            out.setdefault(ref.category or "unknown", []).append(ref)
        return out
