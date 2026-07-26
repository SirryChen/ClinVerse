#!/usr/bin/env python3
"""Compute patient information-unit counts from locally built case JSONs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "info_unit_statistics.json"
CASE_DIRS = {
    "MIMIC-IV-Ext": REPO_ROOT / "ClinVerse" / "data" / "cases",
    "SICdb-Ext": REPO_ROOT / "ClinVerse" / "data" / "cases_sicdb",
}


def summarize_case_dir(case_dir: Path) -> dict[str, float | int]:
    counts: list[int] = []
    missing = 0
    for path in sorted(case_dir.glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            case = json.load(handle)
        units = case.get("patient_information_units")
        if not isinstance(units, list):
            missing += 1
            continue
        # These are the pipeline's patient-visible Atomic Information Units
        # (AIUs), not a proxy derived from text length or event-table rows.
        counts.append(len(units))
    if not counts:
        raise RuntimeError(f"No patient_information_units found in {case_dir}")
    return {
        "n_cases": len(counts),
        "missing_cases": missing,
        "mean_info_units_per_case": statistics.mean(counts),
        "median_info_units_per_case": statistics.median(counts),
        "min_info_units_per_case": min(counts),
        "max_info_units_per_case": max(counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = {
        "definition": "Count of patient_information_units stored in each locally built case JSON.",
        "datasets": {
            name: summarize_case_dir(case_dir)
            for name, case_dir in CASE_DIRS.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
