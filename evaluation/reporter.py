"""Aggregate multiple evaluation traces into a summary report."""

from __future__ import annotations

import json
import statistics
from pathlib import Path


def generate_report(traces_dir: str | Path, output_path: str | Path | None = None) -> dict:
    traces_dir = Path(traces_dir)
    all_scores: list[dict] = []

    for path in sorted(traces_dir.glob("*.json")):
        with open(path) as f:
            trace = json.load(f)
        scores = trace.get("scores", {})
        scores["case_id"] = trace.get("case_id", path.stem)
        all_scores.append(scores)

    if not all_scores:
        return {"n": 0, "message": "No traces found."}

    keys = [
        "information_collection_score",
        "test_order_coverage",
        "abnormal_test_coverage",
        "diagnosis_score",
        "diagnosis_category_accuracy",
        "diagnosis_subtype_accuracy",
    ]

    summary = {"n": len(all_scores), "metrics": {}}
    for k in keys:
        values = [s[k] for s in all_scores if k in s]
        if not values:
            continue
        summary["metrics"][k] = {
            "mean": round(statistics.mean(values), 4),
            "median": round(statistics.median(values), 4),
            "std": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
            "min": round(min(values), 4),
            "max": round(max(values), 4),
        }

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)

    return summary
