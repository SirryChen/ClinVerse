#!/usr/bin/env python3
"""One-off backfill: rewrite evidence_inventory lab entries' `content` in
already-built data/cases/*.json files from the raw itemid=.../value=.../flag=...
format to a human-readable bilingual label (see pipeline/evidence_builder.py::
lab_evidence_content). Existing cases predate the lab catalog itemid
reverse-lookup, so they need a one-time rewrite rather than a full pipeline
re-run (which would require Synthea/Java and re-roll persona generation).

Usage: python scripts/backfill_lab_evidence_labels.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.registry import get_adapter
from pipeline.evidence_builder import lab_evidence_content

CASES_DIR = Path(__file__).parent.parent / "data" / "cases"

# Some pre-refactor case files still carry the old pre-adapter source name.
_SOURCE_ALIASES = {"mimic-iv": "mimic-iv-ext"}


def backfill_file(path: Path) -> bool:
    case = json.loads(path.read_text(encoding="utf-8"))
    source = case["core_admission"]["source"]
    adapter = get_adapter(_SOURCE_ALIASES.get(source, source))

    labs_by_itemid = {str(lab["itemid"]): lab for lab in case["core_admission"].get("labs", [])}

    changed = False
    for ev in case.get("evidence_inventory", []):
        if ev.get("type") != "lab":
            continue
        lab = labs_by_itemid.get(str(ev.get("itemid")))
        if lab is None:
            continue
        new_content = lab_evidence_content(
            adapter, ev["itemid"], lab.get("valuestr", ""), lab.get("flag", ""), ev.get("is_abnormal", False)
        )
        if new_content != ev["content"]:
            ev["content"] = new_content
            changed = True

    if changed:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(case, f, ensure_ascii=False, separators=(",", ":"))
    return changed


def main():
    paths = sorted(CASES_DIR.glob("*.json"))
    n_changed = 0
    for path in paths:
        changed = backfill_file(path)
        n_changed += changed
        print(f"{'[updated]' if changed else '[unchanged]'} {path.name}")
    print(f"\n{n_changed}/{len(paths)} case files updated")


if __name__ == "__main__":
    main()
