"""Radiology report breakdown by modality and region."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd

from config import RADIOLOGY_CSV, ensure_output_dir, filter_accepted
from plot_utils import setup_plot_style, top_n_with_other

MODALITY_MAP = {
    "Radiograph": "X 光",
    "CT": "CT",
    "MRI": "MRI",
    "Ultrasound": "超声",
    "Fluoroscopy": "透视",
    "Nuclear Medicine": "核医学",
    "PET": "PET",
}


def _hbar(counts: pd.Series, title: str, path: str, color: str = "#3b82f6") -> None:
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, max(4, len(counts) * 0.55)))
    bars = ax.barh(counts.index[::-1], counts.values[::-1], color=color, alpha=0.85)
    ax.bar_label(bars, fmt="%,.0f", padding=4, fontsize=13)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("报告数")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    out = ensure_output_dir()
    rad = filter_accepted(
        pd.read_csv(RADIOLOGY_CSV, usecols=["hadm_id", "modality", "region", "exam_name"])
    )
    n_reports = len(rad)
    n_cases = rad["hadm_id"].nunique()

    # Modality distribution
    modality = rad["modality"].fillna("Unknown").map(lambda x: MODALITY_MAP.get(x, x)).value_counts()
    _hbar(modality, f"影像报告检查类型分布 (共 {n_reports:,} 份)", f"{out}/radiology_modality.png", "#2563eb")

    # Region distribution (top 12)
    region = top_n_with_other(rad["region"].fillna("Unknown"), n=12)
    _hbar(region, f"影像检查部位分布 (Top 12)", f"{out}/radiology_region.png", "#7c3aed")

    # Top exam names (top 15)
    exam = top_n_with_other(rad["exam_name"].fillna("Unknown"), n=15)
    _hbar(exam, f"检查项目 Top 15", f"{out}/radiology_exam_top15.png", "#0891b2")

    # Reports per admission
    per_case = rad.groupby("hadm_id").size()
    summary = {
        "n_reports": int(n_reports),
        "n_cases_with_radiology": int(n_cases),
        "modality_counts": rad["modality"].value_counts().to_dict(),
        "region_top15": rad["region"].value_counts().head(15).to_dict(),
        "exam_top15": rad["exam_name"].value_counts().head(15).to_dict(),
        "reports_per_case": {
            "mean": float(per_case.mean()),
            "median": float(per_case.median()),
            "p95": float(per_case.quantile(0.95)),
            "max": int(per_case.max()),
        },
    }
    with open(f"{out}/radiology_breakdown_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[radiology] {n_reports:,} reports, {n_cases:,} cases → modality/region/exam PNGs")


if __name__ == "__main__":
    main()
