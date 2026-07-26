#!/usr/bin/env python3
"""Quality audit: report conflict and leakage statistics across built cases."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import load_yaml_config


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Audit built cases for quality issues.")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--cases-dir", default=None)
    parser.add_argument("--top-n", type=int, default=10, help="Show top-N conflict cases")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)

    cases_dir = Path(args.cases_dir or cfg["paths"]["cases"])
    index_path = cases_dir / "index.jsonl"

    entries = []
    if index_path.exists():
        with open(index_path) as f:
            entries = [json.loads(line) for line in f]
    else:
        for p in sorted(cases_dir.glob("*.json")):
            with open(p) as f:
                case = json.load(f)
            entries.append(case.get("consistency_metadata", {"case_id": p.stem}))

    n = len(entries)
    usable = sum(1 for e in entries if e.get("usable", True))
    high_leakage = [e for e in entries if e.get("leakage_risk_score", 0) > 0.3]
    conflicted = [e for e in entries if e.get("conflict_count", 0) > 0]
    match_scores = [e.get("match_score", 0) for e in entries if "match_score" in e]

    print(f"Total cases: {n}")
    print(f"Usable: {usable} ({100*usable/n:.1f}%)")
    print(f"High leakage risk (>0.3): {len(high_leakage)} ({100*len(high_leakage)/n:.1f}%)")
    print(f"Has conflicts: {len(conflicted)} ({100*len(conflicted)/n:.1f}%)")
    if match_scores:
        import statistics
        print(f"Match score: mean={statistics.mean(match_scores):.3f} "
              f"min={min(match_scores):.3f} max={max(match_scores):.3f}")

    if high_leakage:
        print(f"\nTop {args.top_n} high-leakage cases:")
        for e in sorted(high_leakage, key=lambda x: -x.get("leakage_risk_score", 0))[:args.top_n]:
            print(f"  {e.get('case_id', '?')} leakage={e.get('leakage_risk_score'):.2f} "
                  f"conflicts={e.get('conflict_count', 0)}")


if __name__ == "__main__":
    main()
