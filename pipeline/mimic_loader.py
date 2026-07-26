"""DEPRECATED shim — kept only so old scripts/tests importing
`pipeline.mimic_loader` directly keep working. The real implementation now
lives in adapters/mimic_iv_ext/adapter.py (MimicAdapter), reachable through
the dataset-agnostic core.registry.get_adapter("mimic-iv-ext") for any new
code. Prefer that over importing this module.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from adapters.mimic_iv_ext.adapter import MimicAdapter


def load_admissions(data_dir: str | Path, limit: int | None = None) -> Iterator[dict]:
    """Yield structured admission dicts for all accepted cases."""
    return MimicAdapter(data_dir=data_dir).load_admissions(limit=limit)


def load_admission(hadm_id: str, data_dir: str | Path) -> dict:
    """Load a single admission by hadm_id."""
    return MimicAdapter(data_dir=data_dir).load_admission(hadm_id)
