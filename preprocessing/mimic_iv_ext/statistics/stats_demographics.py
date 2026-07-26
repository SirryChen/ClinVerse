"""Gender, race, admission type, and in-hospital mortality statistics."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd

from config import DEMOGRAPHICS_CSV, OUTPUT_DIR, ensure_output_dir, filter_accepted
from plot_utils import setup_plot_style, top_n_with_other


def _pie(counts: pd.Series, title: str, path: str) -> None:
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.pie(
        counts.values,
        labels=[str(x) for x in counts.index],
        autopct=lambda p: f"{p:.1f}%" if p >= 2 else "",
        startangle=90,
        textprops={"fontsize": 9},
    )
    ax.set_title(title, fontsize=12)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    out = ensure_output_dir()
    demo = filter_accepted(
        pd.read_csv(
            DEMOGRAPHICS_CSV,
            usecols=[
                "hadm_id",
                "gender",
                "race",
                "admission_type",
                "hospital_expire_flag",
            ],
        )
    )

    gender = demo["gender"].value_counts()
    _pie(gender, f"性别分布 (N={len(demo):,})", f"{out}/gender_pie.png")

    race = top_n_with_other(demo["race"].fillna("UNKNOWN"), n=8)
    _pie(race, f"种族分布 (Top 8 + 其他)", f"{out}/race_pie.png")

    adm = top_n_with_other(demo["admission_type"].fillna("UNKNOWN"), n=8)
    _pie(adm, f"入院类型分布", f"{out}/admission_type_pie.png")

    mortality = demo["hospital_expire_flag"].map({0: "存活出院", 1: "院内死亡"}).value_counts()
    _pie(mortality, "院内死亡率", f"{out}/mortality_pie.png")

    summary = {
        "n": len(demo),
        "gender": gender.to_dict(),
        "race_top10": demo["race"].value_counts().head(10).to_dict(),
        "admission_type": demo["admission_type"].value_counts().to_dict(),
        "hospital_expire_rate": float(demo["hospital_expire_flag"].mean()),
    }
    with open(f"{out}/demographics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[demographics] N={len(demo):,} → {out}/gender_pie.png 等")


if __name__ == "__main__":
    main()
