"""Run all MIMIC-IV-Ext statistics scripts."""
from __future__ import annotations

import importlib
import sys
import time

MODULES = [
    "stats_diagnosis_category",
    "stats_age_distribution",
    "stats_lab_count",
    "stats_demographics",
    "stats_modality_count",
    "stats_additional_demographics",
    "stats_hpi_length",
    "stats_lab_abnormal",
    "stats_radiology_breakdown",
]


def main() -> None:
    t0 = time.time()
    for name in MODULES:
        print(f"\n{'=' * 60}\n>>> {name}\n{'=' * 60}")
        mod = importlib.import_module(name)
        mod.main()
    print(f"\n全部完成，耗时 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
