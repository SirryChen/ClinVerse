#!/usr/bin/env python3
"""Export doctor-agent rollouts into Trace2Skill-style analysis directories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from doctor_train.trace2skill import write_error_report, write_success_report


def load_jsonl(path: str | Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Trace2Skill-style analysis inputs from doctor-agent rollouts.")
    parser.add_argument("--input-jsonl", required=True, help="Path to rollouts.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["error", "success", "combined"], default="combined")
    parser.add_argument("--diagnosis-threshold", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N rollout records.")
    args = parser.parse_args()

    records = load_jsonl(args.input_jsonl)
    if args.limit is not None:
        records = records[: args.limit]
    output_dir = Path(args.output_dir)
    error_root = output_dir / "error_analysis"
    success_root = output_dir / "success_analysis"
    error_root.mkdir(parents=True, exist_ok=True)
    success_root.mkdir(parents=True, exist_ok=True)

    exported = {"error": 0, "success": 0}
    for record in records:
        diagnosis_score = float(record["scores"].get("diagnosis_score", 0))
        case_id = record["case_id"]
        if args.mode in {"error", "combined"} and diagnosis_score < args.diagnosis_threshold:
            case_dir = error_root / case_id
            case_dir.mkdir(parents=True, exist_ok=True)
            write_error_report(record, case_dir)
            (case_dir / "evaluate_passed.flag").write_text("", encoding="utf-8")
            exported["error"] += 1
        if args.mode in {"success", "combined"} and diagnosis_score >= args.diagnosis_threshold:
            case_dir = success_root / case_id
            case_dir.mkdir(parents=True, exist_ok=True)
            write_success_report(record, case_dir)
            exported["success"] += 1

    (output_dir / "export_summary.json").write_text(json.dumps(exported, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "limit": args.limit, **exported}, indent=2))


if __name__ == "__main__":
    main()
