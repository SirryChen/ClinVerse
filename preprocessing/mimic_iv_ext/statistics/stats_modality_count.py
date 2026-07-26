"""Microbiology, radiology, and discharge medication counts per admission."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import (
    MEDS_CSV,
    MICRO_CSV,
    OUTPUT_DIR,
    RADIOLOGY_CSV,
    count_per_hadm,
    ensure_output_dir,
    load_accepted_hadm_ids,
)
from plot_utils import setup_plot_style


def main() -> None:
    out = ensure_output_dir()
    accepted = load_accepted_hadm_ids()

    print("[modality] counting microbiology …")
    micro = count_per_hadm(MICRO_CSV)
    print("[modality] counting radiology …")
    rad = count_per_hadm(RADIOLOGY_CSV)
    print("[modality] counting medications …")
    meds = count_per_hadm(MEDS_CSV)

    df = pd.DataFrame(
        {
            "hadm_id": sorted(accepted),
            "microbiology": micro.values,
            "radiology": rad.values,
            "medications": meds.values,
        }
    )

    setup_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    modalities = [
        ("microbiology", "微生物检验次数", "#059669"),
        ("radiology", "影像报告数", "#2563eb"),
        ("medications", "出院带药数", "#d97706"),
    ]
    for ax, (col, label, color) in zip(axes, modalities):
        data = df[col]
        p95 = data.quantile(0.95)
        sns.histplot(data[data <= p95], bins=40, kde=True, ax=ax, color=color)
        ax.set_xlabel(label)
        ax.set_ylabel("病例数")
        ax.set_title(f"{label} (≤P95={p95:.0f}, 均值={data.mean():.1f})")

    fig.suptitle(f"各数据模态每例住院条目数 (N={len(df):,})", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{out}/modality_count_distribution.png")
    plt.close(fig)

    summary = {}
    for col, _, _ in modalities:
        s = df[col]
        summary[col] = {
            "mean": float(s.mean()),
            "median": float(s.median()),
            "max": int(s.max()),
            "p95": float(s.quantile(0.95)),
            "zero_rate": float((s == 0).mean()),
        }
    with open(f"{out}/modality_count_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[modality] → {out}/modality_count_distribution.png")


if __name__ == "__main__":
    main()
