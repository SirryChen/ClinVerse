"""HPI text length distribution."""
from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import DATASET_DIR, ensure_output_dir, load_accepted_hadm_ids
from plot_utils import setup_plot_style

HPI_CSV = os.path.join(DATASET_DIR, "history_of_present_illness.csv")


def main() -> None:
    out = ensure_output_dir()
    accepted = load_accepted_hadm_ids()

    hpi = pd.read_csv(HPI_CSV, usecols=["hadm_id", "hpi"])
    hpi = hpi[hpi["hadm_id"].isin(accepted)].copy()
    hpi["length"] = hpi["hpi"].fillna("").str.len()
    lengths = hpi["length"]

    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(lengths, bins=60, kde=True, ax=axes[0], color="#1d4ed8")
    axes[0].axvline(lengths.mean(), color="#dc2626", ls="--", lw=1.5, label=f"均值 {lengths.mean():.0f}")
    axes[0].axvline(lengths.median(), color="#16a34a", ls="--", lw=1.5, label=f"中位数 {lengths.median():.0f}")
    axes[0].set_xlabel("HPI 字符数")
    axes[0].set_ylabel("病例数")
    axes[0].set_title(f"HPI 文本长度分布 (N={len(lengths):,})")
    axes[0].legend()

    p95 = lengths.quantile(0.95)
    sns.boxplot(x=lengths[lengths <= p95], ax=axes[1], color="#93c5fd")
    axes[1].set_xlabel("HPI 字符数 (≤ P95)")
    axes[1].set_title(f"箱线图 (P95={p95:.0f})")

    fig.tight_layout()
    fig.savefig(f"{out}/hpi_length_distribution.png")
    plt.close(fig)

    bins = {"<500": int((lengths < 500).sum()),
            "500-1000": int(((lengths >= 500) & (lengths < 1000)).sum()),
            "1000-2000": int(((lengths >= 1000) & (lengths < 2000)).sum()),
            "2000-4000": int(((lengths >= 2000) & (lengths < 4000)).sum()),
            "4000+": int((lengths >= 4000).sum())}

    summary = {
        "n": int(len(lengths)),
        "mean": float(lengths.mean()),
        "std": float(lengths.std()),
        "median": float(lengths.median()),
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "p25": float(lengths.quantile(0.25)),
        "p75": float(lengths.quantile(0.75)),
        "p95": float(lengths.quantile(0.95)),
        "length_bins": bins,
    }
    with open(f"{out}/hpi_length_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[hpi_length] mean={lengths.mean():.0f}, median={lengths.median():.0f} → {out}/hpi_length_distribution.png")


if __name__ == "__main__":
    main()
