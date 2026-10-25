#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from env.task_stream import TaskStream


def split_cases(cases_dir, output, seed=2026, train_ratio=0.8, validation_ratio=0.1):
    if not 0 < train_ratio < 1 or not 0 < validation_ratio < 1 - train_ratio:
        raise ValueError("Train and validation ratios must leave a nonempty test fraction")
    groups = defaultdict(list)
    for ref in TaskStream(cases_dir):
        case = json.loads(ref.path.read_text())
        admission = case["core_admission"]
        subject = admission.get("subject_id")
        if subject is None:
            raise ValueError("Each case requires a subject_id for splitting")
        groups[(admission.get("source"), str(subject))].append(ref.case_id)
    subjects = sorted(groups)
    if len(subjects) < 3:
        raise ValueError("At least three patients are required for splitting")
    random.Random(seed).shuffle(subjects)
    train_end = max(1, min(len(subjects) - 2, int(len(subjects) * train_ratio)))
    validation_end = max(train_end + 1, min(len(subjects) - 1, int(len(subjects) * (train_ratio + validation_ratio))))
    parts = {"train": subjects[:train_end], "validation": subjects[train_end:validation_end], "test": subjects[validation_end:]}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split, members in parts.items():
        ids = [case_id for member in members for case_id in groups[member]]
        (output / f"{split}.txt").write_text("\n".join(ids) + "\n")
        counts[split] = {"patients": len(members), "cases": len(ids)}
    (output / "manifest.json").write_text(json.dumps({"seed": seed, "splits": counts}, indent=2))
    return counts


def main():
    parser = argparse.ArgumentParser(description="Create patient-disjoint train, validation and test manifests")
    parser.add_argument("--cases-dir", default="data/cases")
    parser.add_argument("--output", default="data/splits")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    args = parser.parse_args()
    print(json.dumps(split_cases(args.cases_dir, args.output, args.seed, args.train_ratio, args.validation_ratio), indent=2))


if __name__ == "__main__":
    main()
