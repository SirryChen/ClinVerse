#!/usr/bin/env python3
"""Re-extract patient_information_units for existing case JSON files.

This decouples conditioned case generation from AIU extraction so cases can be
built locally, uploaded elsewhere, and enriched with a different LLM later.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

from core.config import load_yaml_config
from pipeline.patient_info_units import (
    OpenAIChatPatientInfoExtractor,
    build_patient_information_units,
    _admission_from_case,
    _persona_from_case_background,
)


def load_config(path: str) -> dict:
    return load_yaml_config(path)


def build_extractor(cfg: dict, provider_name: str) -> OpenAIChatPatientInfoExtractor:
    llm_cfg = cfg["llm_providers"][provider_name]
    return OpenAIChatPatientInfoExtractor(
        api_key=llm_cfg["api_key"],
        base_url=llm_cfg["base_url"],
        model=llm_cfg.get("info_extractor_model")
        or llm_cfg.get("patient_actor_model")
        or llm_cfg.get("doctor_model")
        or llm_cfg.get("lab_model"),
        timeout=llm_cfg.get("timeout", 120),
        extra_body=llm_cfg.get("extra_body"),
    )


def backfill_case(path: Path, extractor: OpenAIChatPatientInfoExtractor) -> bool:
    case = json.loads(path.read_text(encoding="utf-8"))
    patient_view = case.setdefault("simulation_views", {}).setdefault("patient_actor_view", {})
    units = build_patient_information_units(
        _admission_from_case(case),
        _persona_from_case_background(case, patient_view),
        extractor=extractor,
    )
    old_units = case.get("patient_information_units")
    case["patient_information_units"] = units
    patient_view["patient_information_units"] = units
    changed = old_units != units
    if changed:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(case, f, ensure_ascii=False, separators=(",", ":"))
    return changed


def case_has_backfilled_units(path: Path) -> bool:
    case = json.loads(path.read_text(encoding="utf-8"))
    units = case.get("patient_information_units")
    view_units = case.get("simulation_views", {}).get("patient_actor_view", {}).get("patient_information_units")
    return isinstance(units, list) and len(units) > 0 and units == view_units


def main():
    parser = argparse.ArgumentParser(description="Backfill patient_information_units for existing cases.")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--cases-dir", default="data/cases")
    parser.add_argument("--provider", required=True, help="llm_providers key used for extraction")
    parser.add_argument("--limit", type=int, default=None, help="Optional max case count")
    parser.add_argument("--start-index", type=int, default=None,
                        help="Optional 1-based start position after sorting case filenames.")
    parser.add_argument("--start-after", default=None,
                        help="Optional case filename or case_id to start after.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip cases that already have synchronized patient_information_units.")
    parser.add_argument("--fail-log", default=None,
                        help="Optional path to append failed case filenames and errors.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    extractor = build_extractor(cfg, args.provider)
    cases_dir = Path(args.cases_dir)
    paths = sorted(cases_dir.glob("*.json"))
    if args.start_after:
        target = args.start_after
        if not target.endswith(".json"):
            target = f"{target}.json"
        names = [path.name for path in paths]
        if target not in names:
            raise ValueError(f"--start-after target not found: {target}")
        start_pos = names.index(target) + 1
        paths = paths[start_pos:]
    if args.start_index is not None:
        if args.start_index < 1:
            raise ValueError("--start-index must be >= 1")
        paths = paths[args.start_index - 1:]
    if args.limit is not None:
        paths = paths[: args.limit]
    if args.resume:
        paths = [path for path in paths if not case_has_backfilled_units(path)]

    updated = 0
    failed = 0
    progress = tqdm(paths, desc="backfilling patient_information_units", unit="case")
    for i, path in enumerate(progress, start=1):
        progress.set_postfix_str(path.name)
        try:
            changed = backfill_case(path, extractor)
            updated += int(changed)
            print(f"[{i}/{len(paths)}] {'updated' if changed else 'unchanged'} {path.name}")
        except Exception as e:
            failed += 1
            print(f"[{i}/{len(paths)}] failed {path.name}: {e}")
            if args.fail_log:
                with open(args.fail_log, "a", encoding="utf-8") as f:
                    f.write(f"{path.name}\t{type(e).__name__}\t{e}\n")

    print(f"\nDone: updated {updated}/{len(paths)} cases, failed {failed}")


if __name__ == "__main__":
    main()
