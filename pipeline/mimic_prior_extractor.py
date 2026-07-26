"""DEPRECATED shim — kept only so old scripts/tests importing
`pipeline.mimic_prior_extractor` directly keep working.

The real implementation is now split:
  - dataset-agnostic hard_priors/forbidden_clues/orchestration -> pipeline/synthea_priors.py
  - MIMIC's coded-ICD condition-flag extraction -> adapters/mimic_iv_ext/condition_priors.py

Prefer `from pipeline.synthea_priors import extract_all_priors` (passing an
explicit adapter) over this module for any new code.
"""
from __future__ import annotations

from adapters.mimic_iv_ext.condition_priors import extract_condition_priors
from core.registry import get_adapter
from core.schema import Admission
from .synthea_priors import extract_all_priors as _extract_all_priors
from .synthea_priors import extract_forbidden_clues, extract_hard_priors


def extract_all_priors(admission: Admission) -> dict:
    """No-adapter-arg form, defaulting to the MIMIC adapter (this module's
    historical scope) — see pipeline.synthea_priors.extract_all_priors for
    the dataset-agnostic version that takes an explicit adapter."""
    return _extract_all_priors(admission, get_adapter("mimic-iv-ext"))


__all__ = [
    "extract_all_priors",
    "extract_condition_priors",
    "extract_hard_priors",
    "extract_forbidden_clues",
]
