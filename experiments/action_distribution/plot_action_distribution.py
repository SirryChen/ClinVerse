#!/usr/bin/env python3
"""Plot action-selection ratios at each dialogue turn for three doctor models.

Every trace that has an action at a given turn contributes one observation to
that turn.  Hence, trajectories that have already terminated are not carried
forward: each vertical stack is the conditional action distribution among the
trajectories that actually reached that dialogue turn.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_RESULTS = REPO_ROOT / "ClinVerse" / "data" / "eval_results"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "results"

DEFAULT_MODELS = (
    ("Qwen3-32B", "mimic100_qwen3_32b"),
    ("Baichuan-M2-32B", "mimic100_baichuan-m2-32b"),
    ("GPT-5.6-Luna", "mimic100_gpt-5.6-luna"),
)

ACTION_ORDER = (
    "ask_patient",
    "request_physical_exam",
    "order_tests",
    "diagnosis",
)
ACTION_LABELS = {
    "ask_patient": "Ask patient",
    "request_physical_exam": "Physical exam",
    "order_tests": "Order lab test",
    "diagnosis": "Diagnosis",
}
ACTION_COLORS = {
    "ask_patient": "#ebf8e7",
    "request_physical_exam": "#cee5f7",
    "order_tests": "#fff1c6",
    "diagnosis": "#fee7df",
}
BOUNDARY_COLORS = {
    "ask_patient": "#5D8D57",
    "request_physical_exam": "#4F7FA3",
    "order_tests": "#B28825",
    "diagnosis": "#B46B59",
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


def collect_model(spec: ModelSpec, eval_results: Path, max_turn: int) -> tuple[dict[int, Counter], dict[str, int]]:
    """Count the four requested decision categories at every observed dialogue turn."""
    trace_paths = sorted((eval_results / spec.run_id).rglob("traces/*.json"))
    counts: dict[int, Counter] = defaultdict(Counter)
    stats = {
        "trace_files": len(trace_paths),
        "valid_traces": 0,
        "invalid_traces": 0,
        "unknown_actions": 0,
        "imaging_folded_into_lab_test": 0,
        "diagnosis_actions": 0,
    }
    for path in trace_paths:
        try:
            transcript = json.loads(path.read_text(encoding="utf-8")).get("transcript") or {}
            turns = transcript.get("turns") or []
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            stats["invalid_traces"] += 1
            continue
        if not isinstance(turns, list):
            stats["invalid_traces"] += 1
            continue
        stats["valid_traces"] += 1
        for fallback_turn, item in enumerate(turns, start=1):
            try:
                turn = int(item.get("turn", fallback_turn))
            except (AttributeError, TypeError, ValueError):
                turn = fallback_turn
            if not 1 <= turn <= max_turn:
                continue
            action = item.get("function") if isinstance(item, dict) else None
            # The requested view has one test category, so imaging orders are
            # included with lab-test orders rather than being discarded.
            if action == "order_imaging":
                action = "order_tests"
                stats["imaging_folded_into_lab_test"] += 1
            if action not in ACTION_ORDER:
                stats["unknown_actions"] += 1
                continue
            counts[turn][action] += 1
        # `finalize_case` is stored as final_answer rather than a turn. It is
        # the next decision after the final non-terminal tool call.
        if transcript.get("termination_reason") == "finalize_case":
            diagnosis_turn = len(turns) + 1
            if diagnosis_turn <= max_turn:
                counts[diagnosis_turn]["diagnosis"] += 1
                stats["diagnosis_actions"] += 1
    return counts, stats


def summarize(specs: list[ModelSpec], eval_results: Path, max_turn: int) -> tuple[list[dict], dict[str, dict[str, int]]]:
    rows: list[dict] = []
    stats: dict[str, dict[str, int]] = {}
    for spec in specs:
        counts, model_stats = collect_model(spec, eval_results, max_turn)
        stats[spec.label] = model_stats
        for turn in sorted(counts):
            total = sum(counts[turn].values())
            for action in ACTION_ORDER:
                count = counts[turn][action]
                rows.append({
                    "model": spec.label,
                    "turn": turn,
                    "action": action,
                    "count": count,
                    "active_trajectories": total,
                    "ratio": count / total if total else 0.0,
                })
    return rows, stats


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("model", "turn", "action", "count", "active_trajectories", "ratio"))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict], specs: list[ModelSpec], output_path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_path.parent / ".matplotlib"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import PercentFormatter

    by_model: dict[str, dict[int, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        by_model[row["model"]][row["turn"]][row["action"]] = row["ratio"]

    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 16,
    })
    # Match the 11.5 × 6.8 in canvas used by the trajectory-diagnosis figure;
    # the three vertically stacked panels deliberately use a compact height.
    fig, axes = plt.subplots(len(specs), 1, figsize=(11.5, 6.8), sharex=True, sharey=True, constrained_layout=True)
    if len(specs) == 1:
        axes = [axes]
    for axis, spec in zip(axes, specs):
        turns = sorted(by_model[spec.label])
        if not turns:
            axis.set_visible(False)
            continue
        ratios = [[by_model[spec.label][turn].get(action, 0.0) for turn in turns] for action in ACTION_ORDER]
        fills = axis.stackplot(
            turns,
            *ratios,
            colors=[ACTION_COLORS[action] for action in ACTION_ORDER],
            alpha=0.95,
        )
        cumulative = [0.0] * len(turns)
        # Mark the cumulative boundaries (e.g. 0.6, 0.8, 0.9 and 1.0 for 6:2:1:1).
        for action, values in zip(ACTION_ORDER, ratios):
            cumulative = [total + value for total, value in zip(cumulative, values)]
            # Diagnosis is always the final stack and therefore lies exactly
            # on the 100% axis; its marker conveys no additional information.
            if action == "diagnosis":
                continue
            # A white halo isolates each cumulative boundary from both
            # adjacent fills; the dark inner stroke and hollow marker keep
            # individual turn-level ratios readable in dense regions.
            axis.plot(turns, cumulative, color="white", linewidth=4.4, zorder=3)
            axis.plot(
                turns,
                cumulative,
                color=BOUNDARY_COLORS[action],
                marker="o",
                markerfacecolor="white",
                markeredgecolor=BOUNDARY_COLORS[action],
                markeredgewidth=1.6,
                markersize=5.8,
                linewidth=1.75,
                zorder=4,
            )
        axis.set_title(spec.label, loc="center", fontsize=22, fontweight="bold", pad=8)
        axis.set_ylabel("Ratio", fontsize=24, fontweight="bold", labelpad=12)
        axis.set_ylim(0, 1)
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.grid(axis="both", color="#D9D9D9", alpha=0.85, linewidth=0.9, linestyle="--")
        axis.set_axisbelow(True)
        axis.tick_params(axis="both", which="major", labelsize=17, width=1.3, length=6)
        for tick in [*axis.get_xticklabels(), *axis.get_yticklabels()]:
            tick.set_fontweight("bold")
    max_turn = max(row["turn"] for row in rows)
    axes[-1].set_xlabel("Dialogue turn", fontsize=25, fontweight="bold", labelpad=12)
    axes[-1].set_xlim(1, max_turn)
    axes[-1].set_xticks(range(1, max_turn + 1, 2))
    if rows:
        # Use dedicated legend patches so the pale fills remain readable on
        # white: the black outline applies only to the legend color blocks.
        legend_handles = [
            Patch(facecolor=ACTION_COLORS[action], edgecolor="black", linewidth=1.1)
            for action in ACTION_ORDER
        ]
        legend = axes[0].legend(
            legend_handles,
            [ACTION_LABELS[action] for action in ACTION_ORDER],
            loc="upper right",
            frameon=False,
            fontsize=14.5,
            ncol=2,
            columnspacing=0.9,
            handletextpad=0.6,
        )
        for text in legend.get_texts():
            text.set_fontweight("bold")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-results", type=Path, default=DEFAULT_EVAL_RESULTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-turn", type=int, default=20, help="Include actions through this dialogue turn (default: 20).")
    parser.add_argument("--model", type=parse_model, action="append", help="Override defaults; repeat LABEL=RUN_ID exactly three times.")
    args = parser.parse_args()
    if args.max_turn < 1:
        parser.error("--max-turn must be positive.")
    specs = args.model or [ModelSpec(*model) for model in DEFAULT_MODELS]
    if len(specs) != 3:
        parser.error("Exactly three models are required for the three vertically arranged subplots.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows, stats = summarize(specs, args.eval_results, args.max_turn)
    if not rows:
        parser.error("No valid actions found in the selected evaluation runs.")
    csv_path = args.out_dir / "action_distribution_by_turn.csv"
    pdf_path = args.out_dir / "action_distribution_by_turn.pdf"
    write_csv(rows, csv_path)
    plot(rows, specs, pdf_path)
    print(json.dumps({"aggregation": "observed-only: each turn uses only trajectories that reached it", "models": stats, "outputs": {"plot": str(pdf_path), "table": str(csv_path)}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
