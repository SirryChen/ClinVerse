"""Marital status, insurance type, and discharge location statistics."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd

from config import DEMOGRAPHICS_CSV, ensure_output_dir, filter_accepted
from plot_utils import setup_plot_style, top_n_with_other

MARITAL_MAP = {
    "MARRIED": "已婚",
    "SINGLE": "未婚",
    "DIVORCED": "离婚",
    "WIDOWED": "丧偶",
    "SEPARATED": "分居",
    "UNKNOWN (DEFAULT)": "未知",
}

INSURANCE_MAP = {
    "Medicare": "医疗保险(老年)",
    "Medicaid": "医疗救助(低收入)",
    "Other": "其他",
    "Private": "商业保险",
}

DISCHARGE_MAP = {
    "HOME": "回家",
    "HOME HEALTH CARE": "居家护理",
    "SKILLED NURSING FACILITY": "护理院",
    "DIED": "院内死亡",
    "REHAB": "康复机构",
    "LONG TERM CARE HOSPITAL": "长期护理医院",
    "ACUTE HOSPITAL": "转院(急性期)",
    "CHRONIC/LONG TERM ACUTE CARE": "长期急性护理",
    "HOSPICE - HOME": "临终关怀(居家)",
    "HOSPICE - MEDICAL FACILITY": "临终关怀(机构)",
    "AGAINST ADVICE": "自动出院",
    "PSYCHIATRIC FACILITY": "精神科机构",
    "OTHER FACILITY": "其他机构",
}


def _hbar(counts: pd.Series, title: str, path: str, color: str = "#3b82f6") -> None:
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(9, max(4, len(counts) * 0.55)))
    bars = ax.barh(counts.index[::-1], counts.values[::-1], color=color, alpha=0.85)
    ax.bar_label(bars, fmt="%,.0f", padding=4, fontsize=13)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("病例数")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    out = ensure_output_dir()
    demo = filter_accepted(
        pd.read_csv(
            DEMOGRAPHICS_CSV,
            usecols=["hadm_id", "marital_status", "insurance", "discharge_location"],
        )
    )
    n = len(demo)

    # Marital status
    marital_raw = demo["marital_status"].fillna("UNKNOWN (DEFAULT)")
    marital = marital_raw.map(lambda x: MARITAL_MAP.get(x, x)).value_counts()
    _hbar(marital, f"婚姻状况分布 (N={n:,})", f"{out}/marital_distribution.png", "#8b5cf6")

    # Insurance
    ins_raw = demo["insurance"].fillna("Other")
    ins = ins_raw.map(lambda x: INSURANCE_MAP.get(x, x)).value_counts()
    _hbar(ins, f"保险类型分布 (N={n:,})", f"{out}/insurance_distribution.png", "#f59e0b")

    # Discharge location (top 10)
    disc_raw = top_n_with_other(demo["discharge_location"].fillna("OTHER FACILITY"), n=10)
    disc = disc_raw.rename(index=lambda x: DISCHARGE_MAP.get(x, x))
    _hbar(disc, f"出院去向分布 (Top 10)", f"{out}/discharge_distribution.png", "#10b981")

    summary = {
        "n": n,
        "marital_status": marital_raw.value_counts().to_dict(),
        "insurance": ins_raw.value_counts().to_dict(),
        "discharge_location": demo["discharge_location"].value_counts().head(15).to_dict(),
    }
    with open(f"{out}/additional_demographics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[additional_demo] N={n:,} → marital/insurance/discharge PNGs")


if __name__ == "__main__":
    main()
