#!/usr/bin/env python3
"""Draw a compact two-panel summary of MIMIC-IV-Ext and SICdb-Ext.

Panel (a) is a two-ring age-composition donut enriched with cohort and outcome
statistics. Panel (b) is a grouped diagnosis bar chart enriched with clinical
record-density statistics. The result is designed for a two-column KDD paper.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle
from matplotlib.text import Text


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results"
INFO_UNIT_STATS = DEFAULT_OUTPUT_DIR / "info_unit_statistics.json"

# Pastel clinical palette adapted from the project's overview figure:
# saturated green/orange for data marks, soft yellow-green-blue for context.
DATASET_COLORS = {
    "MIMIC-IV-Ext": "#2F7D5A",
    "SICdb-Ext": "#E5822B",
}
AGE_COLORS = ["#F5EDB8", "#CDE8C4", "#9DD7C1", "#65AFC0", "#397B91"]
TEXT = "#27352F"
MUTED = "#6D786F"
GRID = "#E3E8DE"
MALE_COLOR = "#75B9D4"
FEMALE_COLOR = "#EE8E85"
TABLE_HEADER = "#EAF4E2"
MIMIC_ROW = "#F3F9EF"
SICDB_ROW = "#FFF4E3"

DIAGNOSIS_NAMES = {
    "循环系统疾病": "Circulatory",
    "损伤和中毒": "Injury & poisoning",
    "损伤、中毒和外因的某些其他后果": "Injury & poisoning",
    "消化系统疾病": "Digestive",
    "感染性和寄生虫病": "Infectious",
    "某些传染病和寄生虫病": "Infectious",
    "肿瘤": "Neoplasms",
    "症状、体征和异常临床所见": "Symptoms & signs",
    "症状、体征和异常临床与实验室所见": "Symptoms & signs",
    "呼吸系统疾病": "Respiratory",
    "泌尿生殖系统疾病": "Genitourinary",
    "神经系统和感觉器官疾病": "Nervous system",
    "神经系统疾病": "Nervous system",
    "内分泌、营养和代谢疾病": "Endocrine/metabolic",
    "肌肉骨骼系统和结缔组织疾病": "Musculoskeletal",
    "其他": "Other",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_statistics() -> dict[str, dict[str, Any]]:
    datasets: dict[str, dict[str, Any]] = {}
    for key, name in (
        ("mimic-iv-ext", "MIMIC-IV-Ext"),
        ("sicdb-ext", "SICdb-Ext"),
    ):
        summary_dir = REPO_ROOT / "data" / key / "statistics" / "output"
        datasets[name] = {
            "cohort": load_json(REPO_ROOT / "dataset" / key / "stats.json"),
            "age": load_json(summary_dir / "age_distribution_summary.json"),
            "demographics": load_json(summary_dir / "demographics_summary.json"),
            "labs": load_json(summary_dir / "lab_count_summary.json"),
            "modalities": load_json(summary_dir / "modality_count_summary.json"),
            "diagnoses": load_json(summary_dir / "diagnosis_category_summary.json"),
        }
    info_units = load_json(INFO_UNIT_STATS)["datasets"]
    for name in datasets:
        datasets[name]["info_units"] = info_units[name]
    return datasets


def cohort_total(cohort: dict[str, Any]) -> int:
    return cohort.get("total_notes", cohort.get("total_cases"))


def autopct_visible(values: list[int]):
    """Label meaningful wedges only; tiny slices remain visible but uncluttered."""

    def formatter(percent: float) -> str:
        return f"{percent:.0f}%" if percent >= 5 else ""

    return formatter


def draw_donut(ax: plt.Axes, data: dict[str, dict[str, Any]]) -> None:
    age_keys = ["0-17", "18-44", "45-64", "65-74", "75+"]
    age_labels = ["<18", "18–44", "45–64", "65–74", "≥75"]
    mimic_age = data["MIMIC-IV-Ext"]["age"]
    sic_age = data["SICdb-Ext"]["age"]
    mimic_values = [mimic_age["age_bins"][key] for key in age_keys]
    sic_values = [sic_age["age_bins"][key] for key in age_keys]

    # Outer ring: SICdb-Ext; inner ring: MIMIC-IV-Ext.
    outer_wedges, _, outer_autotexts = ax.pie(
        sic_values,
        radius=1.10,
        colors=AGE_COLORS,
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.28, "edgecolor": "white", "linewidth": 1.2},
        autopct=autopct_visible(sic_values),
        pctdistance=0.89,
        textprops={"fontsize": 7.2, "fontweight": "bold", "color": TEXT},
    )
    inner_wedges, _, inner_autotexts = ax.pie(
        mimic_values,
        radius=0.78,
        colors=AGE_COLORS,
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.28, "edgecolor": "white", "linewidth": 1.2},
        autopct=autopct_visible(mimic_values),
        pctdistance=0.84,
        textprops={"fontsize": 7.2, "fontweight": "bold", "color": TEXT},
    )

    total_cases = sum(dataset["cohort"]["accepted"] for dataset in data.values())
    ax.text(
        0,
        0.08,
        f"{total_cases:,}",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color=TEXT,
    )
    ax.text(
        0,
        -0.10,
        "cases",
        ha="center",
        va="center",
        fontsize=7.8,
        color=MUTED,
    )

    # Ring labels make the encoding explicit without consuming dataset colors.
    ax.annotate(
        "SICdb-Ext\nouter ring",
        xy=(-0.82, -0.70),
        xytext=(-1.18, -1.24),
        arrowprops={"arrowstyle": "-", "color": DATASET_COLORS["SICdb-Ext"], "lw": 1.1},
        fontsize=7.4,
        fontweight="bold",
        color=DATASET_COLORS["SICdb-Ext"],
        ha="center",
        va="center",
    )
    ax.annotate(
        "MIMIC-IV-Ext\ninner ring",
        xy=(0.62, -0.43),
        xytext=(0.88, -1.25),
        arrowprops={"arrowstyle": "-", "color": DATASET_COLORS["MIMIC-IV-Ext"], "lw": 1.1},
        fontsize=7.4,
        fontweight="bold",
        color=DATASET_COLORS["MIMIC-IV-Ext"],
        ha="center",
        va="center",
    )

    legend_handles = [
        Patch(facecolor=color, edgecolor="none", label=label)
        for color, label in zip(AGE_COLORS, age_labels)
    ]
    ax.legend(
        handles=legend_handles,
        title="Age (years)",
        ncol=5,
        frameon=False,
        fontsize=8.0,
        title_fontsize=8.5,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.18),
        handlelength=1.15,
        handleheight=0.9,
        handletextpad=0.35,
        columnspacing=0.9,
    )

    ax.set_xlim(-1.50, 1.50)
    ax.set_ylim(-1.56, 1.20)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_combined_table(ax: plt.Axes, data: dict[str, dict[str, Any]]) -> None:
    """Draw one compact cohort-and-evidence table across the full figure."""
    names = ["MIMIC-IV-Ext", "SICdb-Ext"]
    rows = []
    for name in names:
        cohort = data[name]["cohort"]
        demo = data[name]["demographics"]
        retained = cohort["accepted"] / cohort_total(cohort)
        male = demo["gender"]["M"] / demo["n"]
        rows.append(
            [
                name,
                f"{cohort['accepted']:,} / {retained:.1%}",
                f"M {male:.1%} : F {1 - male:.1%}",
                f"{data[name]['labs']['mean']:.1f}",
                f"{data[name]['info_units']['mean_info_units_per_case']:.1f}",
            ]
        )
    ax.axis("off")
    headers = [
        "Dataset",
        "Retained num/ratio",
        "Male : Female",
        "Avg. labs / case",
        "Avg. info units / case",
    ]
    x0, y0, width, height = 0.07, 0.14, 0.86, 0.66
    header_height = height / 3
    data_height = height / 3
    column_widths = np.array([0.19, 0.22, 0.20, 0.17, 0.22])
    column_centers = x0 + width * (
        np.cumsum(column_widths) - column_widths / 2
    )
    group_boundary = x0 + width * column_widths[:3].sum()
    header_bottom = y0 + 2 * data_height
    table_top = y0 + height

    # Rounded outer card with a single header and low-ink body.
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0),
            width,
            height,
            boxstyle="round,pad=0.004,rounding_size=0.012",
            transform=ax.transAxes,
            facecolor="white",
            edgecolor="#789987",
            linewidth=0.9,
            zorder=1,
        )
    )
    ax.add_patch(
        Rectangle(
            (x0, header_bottom),
            width,
            header_height,
            transform=ax.transAxes,
            facecolor=TABLE_HEADER,
            edgecolor="none",
            zorder=2,
        )
    )
    ax.add_patch(
        Rectangle(
            (x0, y0 + data_height),
            width,
            data_height,
            transform=ax.transAxes,
            facecolor=MIMIC_ROW,
            edgecolor="none",
            zorder=2,
        )
    )
    ax.add_patch(
        Rectangle(
            (x0, y0),
            width,
            data_height,
            transform=ax.transAxes,
            facecolor=SICDB_ROW,
            edgecolor="none",
            zorder=2,
        )
    )
    for boundary in (y0 + data_height, header_bottom):
        ax.plot(
            [x0, x0 + width],
            [boundary, boundary],
            transform=ax.transAxes,
            color="#A8BBB0",
            linewidth=0.7,
            zorder=3,
        )

    # Draw column separators after the row fills so the colored backgrounds
    # cannot obscure them. The cohort/evidence boundary remains slightly bold.
    for boundary in x0 + width * np.cumsum(column_widths[:-1]):
        is_group_boundary = np.isclose(boundary, group_boundary)
        ax.plot(
            [boundary, boundary],
            [y0, table_top],
            transform=ax.transAxes,
            color="#95AA9E" if is_group_boundary else "#B4C4BA",
            linewidth=0.8 if is_group_boundary else 0.55,
            zorder=3,
        )

    header_y = header_bottom + header_height / 2
    for x, label in zip(column_centers, headers):
        ax.text(
            x,
            header_y,
            label,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=6.8,
            color=TEXT,
            fontweight="bold",
            zorder=4,
        )
    for row_index, (name, row) in enumerate(zip(names, rows)):
        row_y = y0 + (1.5 - row_index) * data_height
        for column_index, (x, value) in enumerate(zip(column_centers, row)):
            ax.text(
                x,
                row_y,
                value,
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=6.8,
                color=(
                    DATASET_COLORS[name]
                    if column_index == 0
                    else TEXT
                ),
                fontweight="bold",
                zorder=4,
            )
    # Redraw the rounded outline above all fills so its corners remain crisp.
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0),
            width,
            height,
            boxstyle="round,pad=0.004,rounding_size=0.012",
            transform=ax.transAxes,
            facecolor="none",
            edgecolor="#789987",
            linewidth=0.9,
            zorder=5,
        )
    )


def normalized_diagnoses(summary: dict[str, Any]) -> dict[str, float]:
    counts: dict[str, int] = {}
    for original, count in summary["major_chapter_counts"].items():
        translated = DIAGNOSIS_NAMES.get(original, original)
        counts[translated] = counts.get(translated, 0) + count
    denominator = summary["n_admissions"]
    return {name: 100 * count / denominator for name, count in counts.items()}


def draw_bars(ax: plt.Axes, data: dict[str, dict[str, Any]]) -> None:
    names = ["MIMIC-IV-Ext", "SICdb-Ext"]
    shares = {
        name: normalized_diagnoses(data[name]["diagnoses"])
        for name in names
    }
    candidates = set().union(*(dataset.keys() for dataset in shares.values()))
    candidates.discard("Other")
    selected = sorted(
        candidates,
        key=lambda category: np.mean([shares[name].get(category, 0) for name in names]),
        reverse=True,
    )[:8]
    selected = selected[::-1]

    y = np.arange(len(selected))
    bar_height = 0.34
    offsets = {"MIMIC-IV-Ext": -bar_height / 2, "SICdb-Ext": bar_height / 2}
    for name in names:
        values = [shares[name].get(category, 0.0) for category in selected]
        bars = ax.barh(
            y + offsets[name],
            values,
            height=bar_height,
            color=DATASET_COLORS[name],
            edgecolor="white",
            linewidth=0.5,
            label=name,
        )
        for bar, value in zip(bars, values):
            if value >= 1.0:
                ax.text(
                    value + 0.35,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.1f}",
                    va="center",
                    fontsize=6.8,
                    color=DATASET_COLORS[name],
                    fontweight="bold",
                )

    ax.set_yticks(y, selected)
    ax.set_xlim(0, 39)
    ax.set_ylim(-0.65, 7.6)
    ax.set_xlabel("Cases with primary diagnosis (%)", fontsize=8.3)
    ax.set_ylabel("Primary diagnosis category", fontsize=8.3, labelpad=12)
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7.6, colors=TEXT)
    ax.legend(
        frameon=False,
        fontsize=7.4,
        loc="lower right",
        handlelength=1.4,
        handletextpad=0.4,
    )
def draw_evidence_table(ax: plt.Axes, data: dict[str, dict[str, Any]]) -> None:
    """Draw clinical evidence in an axis separate from the diagnosis bars."""
    names = ["MIMIC-IV-Ext", "SICdb-Ext"]
    ax.axis("off")
    table = ax.table(
        cellText=[
            [
                name,
                f"{data[name]['labs']['mean']:.1f}",
                f"{data[name]['info_units']['mean_info_units_per_case']:.1f}",
            ]
            for name in names
        ],
        colLabels=["Dataset", "Avg. labs / case", "Avg. info units / case"],
        cellLoc="center",
        colLoc="center",
        colWidths=[0.32, 0.29, 0.39],
        bbox=[0.00, 0.08, 0.93, 0.72],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.8)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#9DB4A6")
        cell.set_linewidth(0.65)
        if row == 0:
            cell.set_facecolor(TABLE_HEADER)
        elif row == 1:
            cell.set_facecolor(MIMIC_ROW)
        else:
            cell.set_facecolor(SICDB_ROW)
        cell.get_text().set_fontfamily("Times New Roman")
        cell.get_text().set_color(TEXT)
        if row == 0:
            cell.get_text().set_fontweight("bold")
        elif row <= len(names) and column == 0:
            dataset_name = names[row - 1]
            cell.get_text().set_color(DATASET_COLORS[dataset_name])
            cell.get_text().set_fontweight("bold")


def save_plot(output_dir: Path) -> tuple[Path, Path]:
    data = load_statistics()
    output_dir.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.weight": "bold",
            "font.size": 8,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    fig = plt.figure(figsize=(8.5, 4.7))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=(0.21, 0.79),
        width_ratios=(0.95, 0.85),
        left=0.035,
        right=0.99,
        top=0.96,
        bottom=0.12,
        hspace=0.03,
        wspace=0.00,
    )
    summary_ax = fig.add_subplot(grid[0, :])
    donut_ax = fig.add_subplot(grid[1, 0])
    bars_ax = fig.add_subplot(grid[1, 1])
    draw_combined_table(summary_ax, data)
    draw_donut(donut_ax, data)
    draw_bars(bars_ax, data)
    # The table spans the full label+plot width (0.00–0.93). The plotting
    # region begins after the category-label column and is about half the
    # physical width used in the preceding wide version.
    bars_position = bars_ax.get_position()
    bars_ax.set_position(
        [
            bars_position.x0 + bars_position.width * 0.28,
            bars_position.y0,
            bars_position.width * 0.65,
            bars_position.height,
        ]
    )
    # Enforce one consistent bold typographic treatment across titles, labels,
    # tick labels, legends, annotations, and table text.
    for text_artist in fig.findobj(match=Text):
        text_artist.set_fontfamily("Times New Roman")
        text_artist.set_fontweight("bold")
        text_artist.set_fontsize(text_artist.get_fontsize() * 1.32)

    pdf_path = output_dir / "dataset_statistics_overview.pdf"
    png_path = output_dir / "dataset_statistics_overview.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return pdf_path, png_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    pdf_path, png_path = save_plot(args.output_dir)
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
