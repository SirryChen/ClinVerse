#!/usr/bin/env python3
"""Plot diagnosis-probe agreement rates across dialogue steps.

Each probe result is compared with (1) the final diagnosis emitted in its
source evaluation trajectory and (2) the trajectory ground truth.  ICD-10
codes are compared after normalization; when either side has no ICD code, the
script falls back to normalized diagnosis text.  Source final diagnoses and
ground truth are read from the trajectory, never from probe prompts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CLINVERSE_ROOT = REPO_ROOT / "ClinVerse"
DEFAULT_INPUT_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_EVAL_ROOT = CLINVERSE_ROOT / "data" / "eval_results"
sys.path.insert(0, str(CLINVERSE_ROOT))

from evaluation.metrics import diagnosis_category_accuracy  # noqa: E402

# Keep the same color system as information_collection_rate, so model identity
# is consistent across the two experiment figures.
MODEL_COLORS = {
    "Qwen3-8B": "#0072B2",
    "Qwen3-32B": "#009E73",
    "GPT-4o-mini": "#D55E00",
    "GPT-5.6-luna": "#7B4FA3",
    "Claude Sonnet 4.6": "#8C564B",
    "HealthGPT-Pro-4B": "#CC79A7",
    "HealthGPT-Pro-8B": "#56B4E9",
    "Lingshu-32B": "#999999",
    "Baichuan-M2-32B": "#CCB000",
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_code(value: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def normalize_text(value: object) -> str:
    return re.sub(r"[\W_]", "", str(value or "").casefold())


def label_for(record: dict) -> str:
    model = str(record.get("model") or "").strip()
    provider = str(record.get("provider") or "").strip()
    known = {
        "Qwen3-8B": "Qwen3-8B", "Qwen3-14B": "Qwen3-14B", "Qwen3-32B": "Qwen3-32B",
        "Baichuan-M2-32B": "Baichuan-M2-32B", "HealthGPT-Pro-4B": "HealthGPT-Pro-4B",
        "HealthGPT-Pro-8B": "HealthGPT-Pro-8B", "Lingshu-32b": "Lingshu-32B",
        "gpt-4o-mini": "GPT-4o-mini", "gpt-5.6-luna": "GPT-5.6-luna",
        "claude-sonnet-4-6": "Claude Sonnet 4.6",
    }
    return known.get(model) or known.get(provider) or model or provider or "Unknown model"


def source_trace_path(record: dict, eval_root: Path) -> Path:
    """Resolve both copied results and remote absolute source paths locally."""
    trajectory_id = Path(str(record["trajectory_id"]))
    return eval_root / str(record["source_run_id"]) / trajectory_id


def diagnosis_match(probe_code: object, reference_code: object) -> tuple[bool | None, str]:
    """Return ICD-block agreement, or None when either ICD code is absent."""
    left_code, right_code = normalize_code(probe_code), normalize_code(reference_code)
    if left_code and right_code:
        return bool(diagnosis_category_accuracy(left_code, right_code)), "icd10_block"
    return None, "missing_icd10"


def load_reference(trace_path: Path, cache: dict[Path, dict]) -> dict:
    if trace_path not in cache:
        cache[trace_path] = json.loads(trace_path.read_text(encoding="utf-8"))
    trace = cache[trace_path]
    transcript = trace.get("transcript") or {}
    final = transcript.get("final_answer") or {}
    scores = trace.get("scores") or {}
    return {
        "model_final_icd10": scores.get("predicted_icd10") or final.get("icd10_code", ""),
        "model_final_diagnosis": scores.get("predicted_diagnosis") or final.get("final_diagnosis", ""),
        "ground_truth_icd10": scores.get("ground_truth_icd10", ""),
        "ground_truth": scores.get("ground_truth", ""),
    }


def collect(input_paths: list[Path], eval_root: Path) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    cache: dict[Path, dict] = {}
    stats = {"input_files": [str(path) for path in input_paths], "records": 0, "errors": 0, "missing_source_traces": 0}
    for input_path in input_paths:
        for record in read_jsonl(input_path):
            stats["records"] += 1
            # Earlier outputs used --step as one cutoff. Keep that incompatible
            # convention out of interval-sampled curves.
            if record.get("probe_mode") != "interval_carry_forward":
                continue
            if record.get("error"):
                stats["errors"] += 1
                continue
            try:
                reference = load_reference(source_trace_path(record, eval_root), cache)
            except (KeyError, OSError, json.JSONDecodeError):
                stats["missing_source_traces"] += 1
                # Downloaded probe results retain ground truth for offline
                # analysis.  When a source trace is unavailable locally we
                # can still score the right-axis truth curve, while leaving
                # the left-axis final-model comparison unscorable.
                reference = {
                    "model_final_icd10": "",
                    "model_final_diagnosis": "",
                    "ground_truth_icd10": record.get("ground_truth_icd10_for_analysis_only", ""),
                    "ground_truth": record.get("ground_truth_for_analysis_only", ""),
                }
            final_match, final_method = diagnosis_match(record.get("predicted_icd10"), reference["model_final_icd10"])
            truth_match, truth_method = diagnosis_match(record.get("predicted_icd10"), reference["ground_truth_icd10"])
            rows.append({
                "input_file": input_path.name,
                "model": label_for(record),
                "provider": record.get("provider", ""),
                "step": int(record["step"]),
                "trajectory_id": record["trajectory_id"],
                "match_final_model_diagnosis": final_match,
                "match_ground_truth": truth_match,
                "final_comparison_method": final_method,
                "truth_comparison_method": truth_method,
            })
    excluded = sum(row["model"] == "Qwen3-14B" for row in rows)
    stats["excluded_qwen3_14b_records"] = excluded
    return [row for row in rows if row["model"] != "Qwen3-14B"], stats


def summarize(rows: list[dict]) -> list[dict]:
    """Aggregate all trajectories, carrying each early-stop result forward."""
    all_steps = sorted({row["step"] for row in rows})
    by_model_trajectory: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_model_trajectory[(row["model"], row["trajectory_id"])].append(row)
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for (model, _trajectory_id), trajectory_rows in by_model_trajectory.items():
        trajectory_rows.sort(key=lambda row: row["step"])
        last = None
        by_step = {row["step"]: row for row in trajectory_rows}
        for step in all_steps:
            if step in by_step:
                last = by_step[step]
            if last is not None:
                groups[(model, step)].append(last)
    summary = []
    for (model, step), values in sorted(groups.items()):
        final_values = [row["match_final_model_diagnosis"] for row in values if row["match_final_model_diagnosis"] is not None]
        truth_values = [row["match_ground_truth"] for row in values if row["match_ground_truth"] is not None]
        summary.append({
            "model": model,
            "step": step,
            "n_trajectories": len(values),
            "n_comparable_to_final": len(final_values),
            "final_diagnosis_icd_block_agreement_rate": sum(final_values) / len(final_values) if final_values else None,
            "n_comparable_to_truth": len(truth_values),
            "ground_truth_icd_block_agreement_rate": sum(truth_values) / len(truth_values) if truth_values else None,
        })
    return summary


def plot(rows: list[dict], output_path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_path.parent / ".matplotlib"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 13, "axes.labelsize": 17, "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 10})
    by_model: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)
    fig, left = plt.subplots(figsize=(11.5, 6.8), constrained_layout=True)
    for model, points in by_model.items():
        points.sort(key=lambda item: item["step"])
        base = MODEL_COLORS.get(model, "#4D4D4D")
        dark = mcolors.to_hex(mcolors.to_rgba(base, alpha=1.0))
        # A pale companion curve makes the shared model color visible without
        # sacrificing the visual distinction between the two target metrics.
        final = [point["final_diagnosis_icd_block_agreement_rate"] for point in points]
        steps = [point["step"] for point in points]
        if any(value is not None for value in final):
            left.plot(steps, final, color=dark, marker="o", markersize=6.5, linewidth=2.5, label=model)
    left.set_xlabel("Dialogue turn", fontsize=22, fontweight="bold", labelpad=10)
    left.set_ylabel("Diagnosis agreement with final diagnosis", fontsize=22, fontweight="bold", labelpad=12)
    all_steps = sorted({row["step"] for row in rows})
    left.set_xlim(min(all_steps) - 0.3, max(all_steps) + 0.3) if len(all_steps) == 1 else left.set_xlim(left=min(all_steps))
    left.set_xticks(range(0, max(all_steps) + 1, 2))
    plotted_values = [
        row["final_diagnosis_icd_block_agreement_rate"]
        for row in rows
        if row["final_diagnosis_icd_block_agreement_rate"] is not None
    ]
    # Use a data-aware y-scale with a small 5-point margin, so differences
    # between models remain legible instead of being compressed into 0–100%.
    lower = max(0.0, math.floor((min(plotted_values) - 0.03) * 20) / 20)
    upper = min(1.0, math.ceil((max(plotted_values) + 0.03) * 20) / 20)
    if upper - lower < 0.10:
        upper = min(1.0, lower + 0.10)
    left.set_ylim(lower, upper)
    left.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    left.grid(axis="both", color="#D9D9D9", alpha=0.8, linewidth=0.9, linestyle="--")
    left.tick_params(axis="both", which="major", labelsize=16, width=1.3, length=6)
    for label in [*left.get_xticklabels(), *left.get_yticklabels()]:
        label.set_fontweight("bold")
    handles1, labels1 = left.get_legend_handles_labels()
    # Rank models by their last available agreement score.  A single column
    # makes the descending order unambiguous: the best model is shown first.
    final_scores = {
        model: next(
            point["final_diagnosis_icd_block_agreement_rate"]
            for point in reversed(sorted(points, key=lambda item: item["step"]))
            if point["final_diagnosis_icd_block_agreement_rate"] is not None
        )
        for model, points in by_model.items()
        if any(point["final_diagnosis_icd_block_agreement_rate"] is not None for point in points)
    }
    ordered = sorted(zip(handles1, labels1), key=lambda item: final_scores[item[1]], reverse=True)
    legend = left.legend(
        [handle for handle, _ in ordered],
        [label for _, label in ordered],
        loc="best",
        frameon=False,
        ncol=1,
    )
    for text in legend.get_texts():
        text.set_fontsize(15)
        text.set_fontweight("bold")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", help="Probe JSONL input; repeat this argument. Default: all results/*.jsonl.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--eval-results-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_INPUT_DIR)
    args = parser.parse_args()
    # Default to the current random-sampling experiment outputs. This avoids
    # mixing legacy single-checkpoint and earlier filtered-result JSONL files.
    inputs = args.input or sorted(args.input_dir.glob("*_random_carryforward.jsonl"))
    if not inputs:
        raise FileNotFoundError("No probe JSONL files found. Pass --input or check --input-dir.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    record_rows, stats = collect(inputs, args.eval_results_root)
    if not record_rows:
        raise RuntimeError("No probe rows with locally resolvable source trajectories.")
    summary_rows = summarize(record_rows)
    csv_path = args.out_dir / "diagnosis_probe_agreement_by_step.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(summary_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    plot_path = args.out_dir / "diagnosis_probe_agreement_by_step.pdf"
    plot(summary_rows, plot_path)
    report = {"comparison": "ICD-10 block agreement (ClinVerse diagnosis_category_accuracy). Trajectories that end early carry their final available probe result forward at later checkpoints.", "stats": stats, "outputs": {"plot": str(plot_path), "table": str(csv_path)}}
    (args.out_dir / "diagnosis_probe_plot_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
