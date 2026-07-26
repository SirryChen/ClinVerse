#!/usr/bin/env python3
"""Plot action distributions for four Qwen3-32B adaptation methods."""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
DEFAULT_EVAL_RESULTS = REPO_ROOT / "ClinVerse" / "data" / "eval_results"
DEFAULT_OUT_DIR = EXPERIMENT_ROOT / "results" / "training_methods"
sys.path.insert(0, str(EXPERIMENT_ROOT))

from plot_action_distribution import (  # noqa: E402
    ACTION_COLORS,
    ACTION_LABELS,
    ACTION_ORDER,
    BOUNDARY_COLORS,
    ModelSpec,
)


MODEL_SPECS = (
    ModelSpec("Trace2Skill", "mimic100_qwen3_32b_trace2skill_base_heldout"),
    ModelSpec("MemRL-online", "mimic100_qwen3_32b_memrl_online1000_heldout"),
    ModelSpec("SkillRL", "mimic100_qwen3_32b_skillrl_heldout"),
    ModelSpec("MedAgent-Zero", "mimic100_qwen3_32b_medagent_zero"),
)


def official_trace_paths(run_root: Path) -> list[Path]:
    """Read aggregate repeat directories while excluding stale root copies."""
    repeat_dirs = sorted(path for path in run_root.glob("repeat_*") if path.is_dir())
    if repeat_dirs:
        return [
            trace_path
            for repeat_dir in repeat_dirs
            for trace_path in sorted((repeat_dir / "traces").glob("*.json"))
        ]
    return sorted((run_root / "traces").glob("*.json"))


def collect_model(
    spec: ModelSpec, eval_results: Path, max_turn: int
) -> tuple[dict[int, Counter], dict[str, int | str]]:
    trace_paths = official_trace_paths(eval_results / spec.run_id)
    counts: dict[int, Counter] = defaultdict(Counter)
    stats: dict[str, int | str] = {
        "run_id": spec.run_id,
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
            if action == "order_imaging":
                action = "order_tests"
                stats["imaging_folded_into_lab_test"] += 1
            if action not in ACTION_ORDER:
                stats["unknown_actions"] += 1
                continue
            counts[turn][action] += 1
        if transcript.get("termination_reason") == "finalize_case":
            diagnosis_turn = len(turns) + 1
            if diagnosis_turn <= max_turn:
                counts[diagnosis_turn]["diagnosis"] += 1
                stats["diagnosis_actions"] += 1
    return counts, stats


def summarize(
    specs: tuple[ModelSpec, ...], eval_results: Path, max_turn: int
) -> tuple[list[dict], dict[str, dict[str, int | str]]]:
    rows: list[dict] = []
    all_stats: dict[str, dict[str, int | str]] = {}
    for spec in specs:
        counts, stats = collect_model(spec, eval_results, max_turn)
        all_stats[spec.label] = stats
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
    return rows, all_stats


def write_csv(rows: list[dict], path: Path) -> None:
    fields = ("model", "turn", "action", "count", "active_trajectories", "ratio")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict], output_stem: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_stem.parent / ".matplotlib"))
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
        "font.weight": "bold",
        "font.size": 14,
        "axes.titleweight": "bold",
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
    })
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.4), sharex=True, sharey=True)
    flat_axes = axes.ravel()

    for axis, spec in zip(flat_axes, MODEL_SPECS):
        turns = sorted(by_model[spec.label])
        ratios = [
            [by_model[spec.label][turn].get(action, 0.0) for turn in turns]
            for action in ACTION_ORDER
        ]
        axis.stackplot(
            turns,
            *ratios,
            colors=[ACTION_COLORS[action] for action in ACTION_ORDER],
            alpha=0.96,
        )
        cumulative = [0.0] * len(turns)
        for action, values in zip(ACTION_ORDER, ratios):
            cumulative = [total + value for total, value in zip(cumulative, values)]
            if action == "diagnosis":
                continue
            axis.plot(turns, cumulative, color="white", linewidth=3.7, zorder=3)
            axis.plot(
                turns,
                cumulative,
                color=BOUNDARY_COLORS[action],
                marker="o",
                markerfacecolor="white",
                markeredgecolor=BOUNDARY_COLORS[action],
                markeredgewidth=1.35,
                markersize=4.8,
                linewidth=1.55,
                zorder=4,
            )
        axis.set_title(spec.label, fontsize=19, pad=7)
        axis.set_ylim(0, 1)
        axis.set_xlim(1, 20)
        axis.set_xticks(range(1, 21, 2))
        axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        axis.grid(axis="both", color="#D9D9D9", alpha=0.82, linewidth=0.8, linestyle="--")
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_linewidth(1.15)
        axis.spines["bottom"].set_linewidth(1.15)
        axis.tick_params(axis="both", width=1.15, length=5)
        for tick in [*axis.get_xticklabels(), *axis.get_yticklabels()]:
            tick.set_fontweight("bold")

    fig.supxlabel("Dialogue turn", fontsize=21, fontweight="bold", y=0.075)
    fig.supylabel("Action ratio", fontsize=21, fontweight="bold", x=0.02)
    legend_handles = [
        Patch(
            facecolor=ACTION_COLORS[action],
            edgecolor=BOUNDARY_COLORS[action],
            linewidth=1.1,
        )
        for action in ACTION_ORDER
    ]
    legend = fig.legend(
        legend_handles,
        [ACTION_LABELS[action] for action in ACTION_ORDER],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=4,
        frameon=False,
        fontsize=14,
        columnspacing=1.5,
        handletextpad=0.55,
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")
    fig.subplots_adjust(left=0.095, right=0.985, top=0.95, bottom=0.17, wspace=0.10, hspace=0.20)

    for suffix, options in {
        ".pdf": {},
        ".svg": {},
        ".png": {"dpi": 300},
    }.items():
        fig.savefig(output_stem.with_suffix(suffix), bbox_inches="tight", **options)
    plt.close(fig)


def main() -> None:
    max_turn = 20
    DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, stats = summarize(MODEL_SPECS, DEFAULT_EVAL_RESULTS, max_turn)
    if not rows:
        raise RuntimeError("No valid actions found in the selected evaluation runs.")

    csv_path = DEFAULT_OUT_DIR / "training_action_distribution_by_turn.csv"
    output_stem = DEFAULT_OUT_DIR / "training_action_distribution_by_turn"
    write_csv(rows, csv_path)
    plot(rows, output_stem)
    summary = {
        "aggregation": "Observed-only: each turn uses trajectories that reached it.",
        "trace_selection": (
            "Uses repeat_* trace directories when present, excluding stale root-level "
            "trace copies from aggregated runs."
        ),
        "models": stats,
        "outputs": {
            "pdf": str(output_stem.with_suffix(".pdf")),
            "svg": str(output_stem.with_suffix(".svg")),
            "png": str(output_stem.with_suffix(".png")),
            "csv": str(csv_path),
        },
    }
    (DEFAULT_OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
