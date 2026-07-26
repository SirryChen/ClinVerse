"""Laboratory test item count per admission. Mirrors mimic-iv-ext."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import LABS_CSV, count_per_hadm, ensure_output_dir
from plot_utils import setup_plot_style


def main() -> None:
    out = ensure_output_dir()
    lab_counts = count_per_hadm(LABS_CSV)

    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(lab_counts, bins=50, kde=True, ax=axes[0], color="#7c3aed")
    axes[0].set_xlabel("每例住院检验项目数")
    axes[0].set_ylabel("病例数")
    axes[0].set_title(f"检验项目数分布 (N={len(lab_counts):,})")
    axes[0].axvline(lab_counts.mean(), color="#dc2626", ls="--", label=f"均值 {lab_counts.mean():.1f}")
    axes[0].axvline(lab_counts.median(), color="#16a34a", ls="--", label=f"中位数 {lab_counts.median():.0f}")
    axes[0].legend()

    p95 = lab_counts.quantile(0.95)
    sns.boxplot(x=lab_counts[lab_counts <= p95], ax=axes[1], color="#a78bfa")
    axes[1].set_xlabel("检验项目数 (≤ P95)")
    axes[1].set_title(f"箱线图 (P95={p95:.0f})")

    fig.tight_layout()
    fig.savefig(f"{out}/lab_count_distribution.png")
    plt.close(fig)

    summary = {
        "n_admissions": int(len(lab_counts)),
        "mean": float(lab_counts.mean()),
        "std": float(lab_counts.std()),
        "median": float(lab_counts.median()),
        "min": int(lab_counts.min()),
        "max": int(lab_counts.max()),
        "p25": float(lab_counts.quantile(0.25)),
        "p75": float(lab_counts.quantile(0.75)),
        "p95": float(lab_counts.quantile(0.95)),
        "zero_lab_admissions": int((lab_counts == 0).sum()),
    }
    with open(f"{out}/lab_count_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(
        f"[labs] mean={lab_counts.mean():.1f}, median={lab_counts.median():.0f} "
        f"→ {out}/lab_count_distribution.png"
    )


if __name__ == "__main__":
    main()
