#!/usr/bin/env python3
"""Plot information-collection dynamics for Qwen3-32B adaptations."""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
CLINVERSE_ROOT = REPO_ROOT / "ClinVerse"
sys.path.insert(0, str(EXPERIMENT_ROOT))

from plot_information_collection_rate import (  # noqa: E402
    ModelSpec,
    load_case,
    summarize,
    trajectory,
)


MODEL_SPECS = (
    ModelSpec("Qwen3-32B", "mimic100_qwen3_32b"),
    ModelSpec("Trace2Skill", "mimic100_qwen3_32b_trace2skill_base_heldout"),
    ModelSpec("SkillRL", "mimic100_qwen3_32b_skillrl_heldout"),
    ModelSpec("MemRL-offline", "mimic500_qwen3_32b_memrl_heldout"),
    ModelSpec("MemRL-online", "mimic100_qwen3_32b_memrl_online1000_heldout"),
    ModelSpec("MedAgent-Zero", "mimic100_qwen3_32b_medagent_zero"),
)

STYLES = {
    "Qwen3-32B": {"color": "#333333", "marker": "o", "linestyle": "-"},
    "Trace2Skill": {"color": "#0072B2", "marker": "s", "linestyle": "-"},
    "SkillRL": {"color": "#56B4E9", "marker": "^", "linestyle": "-"},
    "MemRL-offline": {"color": "#009E73", "marker": "D", "linestyle": "-"},
    "MemRL-online": {"color": "#E69F00", "marker": "P", "linestyle": "-"},
    "MedAgent-Zero": {"color": "#CC79A7", "marker": "X", "linestyle": "-"},
}


def official_trace_paths(run_root: Path) -> list[Path]:
    """Use aggregate repeat directories and ignore stale root-level traces."""
    repeat_dirs = sorted(path for path in run_root.glob("repeat_*") if path.is_dir())
    if repeat_dirs:
        return [
            trace_path
            for repeat_dir in repeat_dirs
            for trace_path in sorted((repeat_dir / "traces").glob("*.json"))
        ]
    return sorted((run_root / "traces").glob("*.json"))


def collect_trajectories(
    spec: ModelSpec, eval_results: Path, cases_dir: Path
) -> tuple[list[list[tuple[int, float]]], dict[str, int | str]]:
    run_root = eval_results / spec.run_id
    trace_paths = official_trace_paths(run_root)
    trajectories: list[list[tuple[int, float]]] = []
    cases: dict[str, dict] = {}
    missing_cases = invalid_traces = 0

    for path in trace_paths:
        try:
            trace = json.loads(path.read_text(encoding="utf-8"))
            case_id = str(
                trace.get("case_id")
                or (trace.get("transcript") or {}).get("case_id")
                or ""
            )
            case = load_case(case_id, cases_dir, cases)
            if case is None:
                missing_cases += 1
                continue
            values = trajectory(trace, case)
            if not values:
                invalid_traces += 1
                continue
            trajectories.append(values)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            invalid_traces += 1

    return trajectories, {
        "run_id": spec.run_id,
        "trace_files": len(trace_paths),
        "scored_traces": len(trajectories),
        "missing_cases": missing_cases,
        "invalid_traces": invalid_traces,
    }


def plot(rows: list[dict], output_stem: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_stem.parent / ".matplotlib"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.weight": "bold",
        "font.size": 14,
        "axes.labelsize": 18,
        "axes.labelweight": "bold",
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 12.5,
    })

    by_model: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)

    fig, ax = plt.subplots(figsize=(9.2, 5.25), constrained_layout=True)
    for spec in MODEL_SPECS:
        points = sorted(by_model[spec.label], key=lambda row: row["turn"])
        style = STYLES[spec.label]
        turns = [row["turn"] for row in points]
        means = [row["information_collection_rate_mean"] for row in points]
        confidence = [
            1.96 * row["information_collection_rate_std"] / math.sqrt(row["n_trajectories"])
            for row in points
        ]
        ax.fill_between(
            turns,
            [max(0.0, value - error) for value, error in zip(means, confidence)],
            [min(1.0, value + error) for value, error in zip(means, confidence)],
            color=style["color"],
            alpha=0.075,
            linewidth=0,
        )
        ax.plot(
            turns,
            means,
            label=spec.label,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.35,
            markersize=5.8,
            markeredgewidth=0.7,
            markeredgecolor="white",
        )

    max_turn = max(row["turn"] for row in rows)
    max_rate = max(row["information_collection_rate_mean"] for row in rows)
    y_max = min(1.0, max(0.2, math.ceil((max_rate + 0.018) * 20) / 20))
    ax.set_xlim(0, max_turn)
    ax.set_ylim(0, y_max)
    ax.set_xticks(range(0, max_turn + 1, 2))
    ax.set_yticks([value / 100 for value in range(0, int(round(y_max * 100)) + 1, 2)])
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlabel("Dialogue turn", labelpad=8)
    ax.set_ylabel("Average information collection rate", labelpad=10)
    ax.grid(axis="both", color="#D8D8D8", linewidth=0.8, linestyle="--", alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.25)
    ax.spines["bottom"].set_linewidth(1.25)
    ax.tick_params(axis="both", width=1.2, length=5)
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontweight("bold")

    legend = ax.legend(
        loc="upper left",
        ncol=2,
        frameon=False,
        columnspacing=1.4,
        handlelength=2.7,
        handletextpad=0.6,
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")

    for suffix, save_options in {
        ".pdf": {},
        ".svg": {},
        ".png": {"dpi": 300},
    }.items():
        fig.savefig(output_stem.with_suffix(suffix), bbox_inches="tight", **save_options)
    plt.close(fig)


def main() -> None:
    eval_results = CLINVERSE_ROOT / "data/eval_results"
    cases_dir = CLINVERSE_ROOT / "data/cases"
    out_dir = EXPERIMENT_ROOT / "results" / "qwen_training_methods"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    stats: dict[str, dict[str, int | str]] = {}
    for spec in MODEL_SPECS:
        trajectories, model_stats = collect_trajectories(spec, eval_results, cases_dir)
        if not trajectories:
            raise RuntimeError(f"No scorable traces found for {spec.label}: {spec.run_id}")
        rows.extend(summarize(trajectories, spec.label, "carry-forward"))
        stats[spec.label] = model_stats

    csv_path = out_dir / "qwen_training_information_collection_rate.csv"
    fields = [
        "model",
        "turn",
        "information_collection_rate_mean",
        "information_collection_rate_std",
        "n_trajectories",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    output_stem = out_dir / "qwen_training_information_collection_rate"
    plot(rows, output_stem)
    summary = {
        "metric": "Cumulative weighted patient-history information collection rate",
        "aggregation": "Mean by dialogue turn; terminal values are carried forward.",
        "uncertainty": "Shaded bands show trajectory-level 95% confidence intervals.",
        "trace_selection": (
            "Uses repeat_* trace directories when present, excluding stale root-level "
            "trace copies from aggregated runs."
        ),
        "model_stats": stats,
        "outputs": {
            "pdf": str(output_stem.with_suffix(".pdf")),
            "svg": str(output_stem.with_suffix(".svg")),
            "png": str(output_stem.with_suffix(".png")),
            "csv": str(csv_path),
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
