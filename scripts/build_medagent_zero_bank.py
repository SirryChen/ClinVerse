#!/usr/bin/env python3
"""Build a MedAgent-Zero-style dual library from doctor rollouts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from doctor_train.medagent_zero import build_bank, load_jsonl, save_bank


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MedAgent-Zero record and experience libraries from rollouts.")
    parser.add_argument("--input-jsonl", required=True, help="rollouts.jsonl from run_doctor_rollouts.py")
    parser.add_argument("--output", required=True, help="Output MedAgent-Zero bank JSON path")
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N rollout records.")
    args = parser.parse_args()

    records = load_jsonl(args.input_jsonl)
    if args.limit is not None:
        records = records[: args.limit]
    bank = build_bank(records)
    save_bank(bank, args.output)
    counts = {
        stage: {
            "records": len(libs.get("record_library", [])),
            "experiences": len(libs.get("experience_base", [])),
        }
        for stage, libs in bank.get("stages", {}).items()
    }
    print({"output": args.output, "limit": args.limit, "n_rollouts": len(records), "counts": counts})


if __name__ == "__main__":
    main()
