#!/usr/bin/env python3
"""Plot cumulative patient-information collection rate by dialogue turn.

The script pools every trace found under each configured evaluation run. A
trace is one independent case trajectory, so repeated evaluations contribute
additional trajectories. The score is recomputed from the trace's disclosed
information IDs and the corresponding case's
``patient_information_units`` using ClinVerse's current scoring definition.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev

REPO_ROOT = Path(__file__).resolve().parents[2]
CLINVERSE_ROOT = REPO_ROOT / "ClinVerse"
sys.path.insert(0, str(CLINVERSE_ROOT))

from evaluation.metrics import history_collection_score  # noqa: E402
from evaluation.scorer import _patient_information_units  # noqa: E402


DEFAULT_MODELS = (
    ("Qwen3-8B", "mimic100_qwen3_8b"),
    ("Qwen3-32B", "mimic100_qwen3_32b"),
    ("GPT-4o-mini", "mimic100_gpt-4o-mini"),
    ("GPT-5.6-luna", "mimic100_gpt-5.6-luna"),
    ("Claude Sonnet 4.6", "mimic100_claude-sonnet-4-6"),
    ("HealthGPT-Pro-4B", "mimic100_healthgpt-pro-4b"),
    ("HealthGPT-Pro-8B", "mimic100_healthgpt-pro-8b"),
)

# Deliberately chosen for strong separation on white backgrounds and for
# color-vision accessibility; do not rely on Matplotlib's default cycle.
MODEL_COLORS = {
    "Qwen3-8B": "#0072B2",          # blue
    "Qwen3-32B": "#009E73",         # bluish green
    "GPT-4o-mini": "#D55E00",       # vermilion
    "GPT-5.6-luna": "#7B4FA3",      # deep violet
    "Claude Sonnet 4.6": "#8C564B", # warm brown
    "HealthGPT-Pro-4B": "#CC79A7",  # magenta
    "HealthGPT-Pro-8B": "#56B4E9",  # sky blue
}


@dataclass(frozen=True)
class ModelSpec:
    label: str
    run_id: str


def parse_model(value: str) -> ModelSpec:
    try:
        label, run_id = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected LABEL=RUN_ID.") from exc
    if not label.strip() or not run_id.strip():
        raise argparse.ArgumentTypeError("LABEL and RUN_ID must both be non-empty.")
    return ModelSpec(label.strip(), run_id.strip())


def load_case(case_id: str, cases_dir: Path, cache: dict[str, dict]) -> dict | None:
    if case_id not in cache:
        path = cases_dir / f"{case_id}.json"
        if not path.exists():
            return None
        cache[case_id] = json.loads(path.read_text(encoding="utf-8"))
    return cache[case_id]


def disclosed_ids(turn: dict) -> list[str]:
    meta = turn.get("hidden_metadata") or {}
    return list(meta.get("disclosed_info_ids") or []) + list(
        meta.get("disclosed_evidence_ids") or []
    )


def trajectory(trace: dict, case: dict) -> list[tuple[int, float]]:
    """Return the cumulative official history-collection score at every turn."""
    turns = (trace.get("transcript") or {}).get("turns") or []
    units = _patient_information_units(case)
    collected: list[str] = []
    values: list[tuple[int, float]] = []
    for fallback_turn, turn in enumerate(turns, start=1):
        collected.extend(disclosed_ids(turn))
        turn_number = turn.get("turn", fallback_turn)
        try:
            turn_number = int(turn_number)
        except (TypeError, ValueError):
            turn_number = fallback_turn
        values.append((turn_number, history_collection_score(collected, units)))
    return values


def collect_model(
    spec: ModelSpec, eval_results: Path, cases_dir: Path
) -> tuple[list[list[tuple[int, float]]], dict[str, int]]:
    run_root = eval_results / spec.run_id
    trace_paths = sorted(run_root.rglob("traces/*.json")) if run_root.exists() else []
    trajectories: list[list[tuple[int, float]]] = []
    stats = {"trace_files": len(trace_paths), "scored_traces": 0, "missing_cases": 0, "invalid_traces": 0}
    cases: dict[str, dict] = {}

    for path in trace_paths:
        try:
            trace = json.loads(path.read_text(encoding="utf-8"))
            case_id = str(trace.get("case_id") or (trace.get("transcript") or {}).get("case_id") or "")
            case = load_case(case_id, cases_dir, cases)
            if case is None:
                stats["missing_cases"] += 1
                continue
            values = trajectory(trace, case)
            if not values:
                stats["invalid_traces"] += 1
                continue
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            stats["invalid_traces"] += 1
            continue
        trajectories.append(values)
        stats["scored_traces"] += 1
    return trajectories, stats


def summarize(
    trajectories: list[list[tuple[int, float]]], label: str, terminal_policy: str
) -> list[dict]:
    if not trajectories:
        return []
    max_turn = max(turn for trajectory_values in trajectories for turn, _ in trajectory_values)
    per_turn: dict[int, list[float]] = defaultdict(list)
    # Before the first doctor action, no non-initially-visible patient
    # information has been elicited. Include this common baseline explicitly.
    per_turn[0] = [0.0] * len(trajectories)
    for trajectory_values in trajectories:
        scores_by_turn = dict(trajectory_values)
        last_score: float | None = None
        for turn in range(1, max_turn + 1):
            if turn in scores_by_turn:
                last_score = scores_by_turn[turn]
            if last_score is not None and (terminal_policy == "carry-forward" or turn in scores_by_turn):
                per_turn[turn].append(last_score)
    rows = []
    for turn in sorted(per_turn):
        values = per_turn[turn]
        rows.append({
            "model": label,
            "turn": turn,
            "information_collection_rate_mean": round(mean(values), 6),
            "information_collection_rate_std": round(stdev(values), 6) if len(values) > 1 else 0.0,
            "n_trajectories": len(values),
        })
    return rows


def plot(rows: list[dict], output_path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_path.parent / ".matplotlib"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 14,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
        "legend.title_fontsize": 14,
    })

    by_model: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)

    fig, ax = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    for label, points in by_model.items():
        points.sort(key=lambda row: row["turn"])
        ax.plot(
            [row["turn"] for row in points],
            [row["information_collection_rate_mean"] for row in points],
            color=MODEL_COLORS.get(label, "#4D4D4D"),
            marker="o", markersize=6.2, linewidth=2.4, label=label,
        )
    ax.set_xlabel("Dialogue turn", fontsize=22, fontweight="bold", labelpad=10)
    ax.set_ylabel("Average information collection rate", fontsize=22, fontweight="bold", labelpad=12)
    ax.set_xlim(left=0)
    # Rates are fractions, but use a data-aware upper limit so early-turn
    # differences remain legible when all models collect far below 100%.
    max_rate = max(row["information_collection_rate_mean"] for row in rows)
    ax.set_ylim(0, min(1.0, max(0.1, math.ceil((max_rate + 0.02) * 20) / 20)))
    max_turn = max(row["turn"] for row in rows)
    ax.set_xticks(range(0, max_turn + 1, 2))
    ax.tick_params(axis="both", which="major", labelsize=16, width=1.3, length=6)
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontweight("bold")
    ax.grid(axis="both", color="#D9D9D9", alpha=0.8, linewidth=0.9, linestyle="--")
    # Order the legend by the final-turn score, with the best model at top.
    handles, labels = ax.get_legend_handles_labels()
    final_scores = {
        label: max(points, key=lambda row: row["turn"])["information_collection_rate_mean"]
        for label, points in by_model.items()
    }
    ordered = sorted(zip(handles, labels), key=lambda item: final_scores[item[1]], reverse=True)
    legend = ax.legend(
        [handle for handle, _ in ordered],
        [label for _, label in ordered],
        loc="best",
        frameon=False,
    )
    for text in legend.get_texts():
        text.set_fontsize(15)
        text.set_fontweight("bold")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-results", type=Path, default=CLINVERSE_ROOT / "data/eval_results")
    parser.add_argument("--cases-dir", type=Path, default=CLINVERSE_ROOT / "data/cases")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument(
        "--terminal-policy", choices=("carry-forward", "observed-only"), default="carry-forward",
        help="How to aggregate after a trace terminates. carry-forward keeps its final rate (default).",
    )
    parser.add_argument(
        "--model", action="append", type=parse_model,
        help="Optional LABEL=RUN_ID mapping. Repeat this option to override the default seven models.",
    )
    args = parser.parse_args()
    specs = args.model or [ModelSpec(*item) for item in DEFAULT_MODELS]
    specs = [spec for spec in specs if spec.label != "Qwen3-14B"]
    if not specs:
        raise ValueError("Qwen3-14B is excluded from this figure; select at least one other model.")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    model_stats: dict[str, dict[str, int | str]] = {}
    for spec in specs:
        trajectories, stats = collect_model(spec, args.eval_results, args.cases_dir)
        all_rows.extend(summarize(trajectories, spec.label, args.terminal_policy))
        model_stats[spec.label] = {"run_id": spec.run_id, **stats}

    if not all_rows:
        raise RuntimeError("No scorable traces found. Check --eval-results, --cases-dir, and --model mappings.")

    csv_path = args.out_dir / "information_collection_rate_by_turn.csv"
    fields = ["model", "turn", "information_collection_rate_mean", "information_collection_rate_std", "n_trajectories"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    plot_path = args.out_dir / "information_collection_rate_by_turn.pdf"
    plot(all_rows, plot_path)
    summary = {
        "metric": "Cumulative weighted patient-history information collection rate",
        "aggregation": (
            "Repeated runs are independent trajectories. "
            + ("After a trace terminates, retain its final rate at later turns." if args.terminal_policy == "carry-forward"
               else "At each turn, use only traces that reached that turn.")
        ),
        "terminal_policy": args.terminal_policy,
        "model_stats": model_stats,
        "outputs": {"plot": str(plot_path), "table": str(csv_path)},
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
