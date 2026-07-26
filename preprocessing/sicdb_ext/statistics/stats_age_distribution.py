"""Patient age distribution (accepted admissions). Mirrors mimic-iv-ext."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import DEMOGRAPHICS_CSV, ensure_output_dir, filter_accepted
from plot_utils import setup_plot_style


def main() -> None:
    out = ensure_output_dir()
    demo = filter_accepted(pd.read_csv(DEMOGRAPHICS_CSV, usecols=["hadm_id", "anchor_age"]))
    ages = demo["anchor_age"].dropna().astype(int)

    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.kdeplot(ages, ax=ax, fill=True, alpha=0.35, color="#2563eb", linewidth=2, label="核密度曲线")
    ax.hist(
        ages,
        bins=range(int(ages.min()), int(ages.max()) + 2),
        density=True,
        alpha=0.25,
        color="#64748b",
        edgecolor="white",
        linewidth=0.4,
        label="直方图(密度)",
    )

    ax.axvline(ages.mean(), color="#dc2626", linestyle="--", linewidth=1.5, label=f"均值 {ages.mean():.1f} 岁")
    ax.axvline(ages.median(), color="#16a34a", linestyle="--", linewidth=1.5, label=f"中位数 {ages.median():.0f} 岁")

    ax.set_xlabel("入院年龄 (anchor_age)")
    ax.set_ylabel("密度")
    ax.set_title(f"患者年龄分布 (N={len(ages):,})")
    ax.legend(loc="upper right")
    ax.set_xlim(max(0, ages.min() - 2), ages.max() + 2)
    fig.savefig(f"{out}/age_distribution.png")
    plt.close(fig)

    summary = {
        "n": int(len(ages)),
        "mean": float(ages.mean()),
        "std": float(ages.std()),
        "median": float(ages.median()),
        "min": int(ages.min()),
        "max": int(ages.max()),
        "q25": float(ages.quantile(0.25)),
        "q75": float(ages.quantile(0.75)),
        "age_bins": {
            "0-17": int(((ages >= 0) & (ages < 18)).sum()),
            "18-44": int(((ages >= 18) & (ages < 45)).sum()),
            "45-64": int(((ages >= 45) & (ages < 65)).sum()),
            "65-74": int(((ages >= 65) & (ages < 75)).sum()),
            "75+": int((ages >= 75).sum()),
        },
    }
    with open(f"{out}/age_distribution_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[age] mean={ages.mean():.1f}, median={ages.median():.0f} → {out}/age_distribution.png")


if __name__ == "__main__":
    main()
