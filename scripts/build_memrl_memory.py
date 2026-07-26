#!/usr/bin/env python3
"""Build a lightweight MemRL-compatible episodic memory bank from rollouts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from doctor_train.memrl import build_memory_bank, load_jsonl, save_memory_bank


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Hospital-MemRL episodic memory from doctor rollouts.")
    parser.add_argument("--input-jsonl", required=True, help="Path to rollouts.jsonl")
    parser.add_argument("--output", required=True, help="Output memory bank JSON path")
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N rollout records.")
    args = parser.parse_args()

    records = load_jsonl(args.input_jsonl)
    if args.limit is not None:
        records = records[: args.limit]
    bank = build_memory_bank(records)
    save_memory_bank(bank, args.output)
    print(json.dumps({"output": args.output, "limit": args.limit, "memories": len(bank["items"])}, indent=2))


if __name__ == "__main__":
    main()
