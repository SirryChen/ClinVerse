"""Medication and device/line (icd_procedures proxy) counts per admission.

Adapted from mimic-iv-ext/statistics/stats_modality_count.py. SICdb has no
radiology or microbiology tables (see predata/README.md), so the two
modalities charted here are the ones SICdb-Ext actually has: discharge
medications and device/line events (data_range, written into icd_procedures
as a proxy for procedures — see predata/loader.py:write_icd_procedures).
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import (
    ICD_PROC_CSV,
    MEDS_CSV,
    count_per_hadm,
    ensure_output_dir,
    load_accepted_hadm_ids,
)
from plot_utils import setup_plot_style


def main() -> None:
    out = ensure_output_dir()
    accepted = load_accepted_hadm_ids()

    print("[modality] counting medications …")
    meds = count_per_hadm(MEDS_CSV)
    print("[modality] counting device/line events (icd_procedures proxy) …")
    devices = count_per_hadm(ICD_PROC_CSV)

    df = pd.DataFrame(
        {
            "hadm_id": sorted(accepted),
            "medications": meds.values,
            "devices": devices.values,
        }
    )

    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    modalities = [
        ("medications", "用药条目数", "#d97706"),
        ("devices", "管路/设备事件数", "#059669"),
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
