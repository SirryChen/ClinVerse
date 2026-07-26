"""Lab test abnormality rate statistics."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import LABS_CSV, ensure_output_dir, load_accepted_hadm_ids
from plot_utils import setup_plot_style


def main() -> None:
    out = ensure_output_dir()
    accepted = load_accepted_hadm_ids()

    total_counts: dict[int, int] = {}
    abnormal_counts: dict[int, int] = {}

    for chunk in pd.read_csv(
        LABS_CSV,
        usecols=["hadm_id", "flag"],
        chunksize=500_000,
    ):
        sub = chunk[chunk["hadm_id"].isin(accepted)]
        for hadm_id, grp in sub.groupby("hadm_id"):
            hid = int(hadm_id)
            total_counts[hid] = total_counts.get(hid, 0) + len(grp)
            abn = grp["flag"].isin(["abnormal", "critical", "delta"]).sum()
            abnormal_counts[hid] = abnormal_counts.get(hid, 0) + int(abn)

    total_s = pd.Series(total_counts)
    abnormal_s = pd.Series(abnormal_counts).reindex(total_s.index, fill_value=0)
    rate_s = (abnormal_s / total_s.replace(0, 1))

    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(abnormal_s, bins=50, kde=True, ax=axes[0], color="#ef4444")
    axes[0].axvline(abnormal_s.mean(), color="#dc2626", ls="--", lw=1.5,
                    label=f"均值 {abnormal_s.mean():.1f}")
    axes[0].axvline(abnormal_s.median(), color="#16a34a", ls="--", lw=1.5,
                    label=f"中位数 {abnormal_s.median():.0f}")
    axes[0].set_xlabel("每例异常检验数")
    axes[0].set_ylabel("病例数")
    axes[0].set_title(f"每例异常检验项目数分布 (N={len(abnormal_s):,})")
    axes[0].legend()

    sns.histplot(rate_s * 100, bins=40, kde=True, ax=axes[1], color="#f97316")
    axes[1].axvline(rate_s.mean() * 100, color="#dc2626", ls="--", lw=1.5,
                    label=f"均值 {rate_s.mean()*100:.1f}%")
    axes[1].set_xlabel("异常率 (%)")
    axes[1].set_ylabel("病例数")
    axes[1].set_title("检验异常率分布")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(f"{out}/lab_abnormal_distribution.png")
    plt.close(fig)

    overall_rate = abnormal_s.sum() / total_s.sum()
    summary = {
        "n_admissions": int(len(total_s)),
        "total_lab_items": int(total_s.sum()),
        "total_abnormal_items": int(abnormal_s.sum()),
        "overall_abnormal_rate": float(overall_rate),
        "mean_abnormal_per_admission": float(abnormal_s.mean()),
        "median_abnormal_per_admission": float(abnormal_s.median()),
        "p95_abnormal_per_admission": float(abnormal_s.quantile(0.95)),
        "mean_abnormal_rate_per_admission": float(rate_s.mean()),
        "median_abnormal_rate_per_admission": float(rate_s.median()),
    }
    with open(f"{out}/lab_abnormal_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(
        f"[lab_abnormal] overall_rate={overall_rate*100:.1f}%, "
        f"mean_abn/case={abnormal_s.mean():.1f} → {out}/lab_abnormal_distribution.png"
    )


if __name__ == "__main__":
    main()
