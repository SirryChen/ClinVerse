"""Plot styling shared across statistics scripts."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from config import PLOT_FONT_FAMILY


def setup_plot_style() -> None:
    plt.rcParams["font.sans-serif"] = PLOT_FONT_FAMILY
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 200
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["font.size"] = 14
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["xtick.labelsize"] = 13
    plt.rcParams["ytick.labelsize"] = 13
    plt.rcParams["legend.fontsize"] = 13


def top_n_with_other(series, n: int = 12, other_label: str = "其他"):
    """Keep top-N categories; aggregate the rest."""
    counts = series.value_counts()
    if len(counts) <= n:
        return counts
    top = counts.head(n)
    other = counts.iloc[n:].sum()
    return pd.concat([top, pd.Series({other_label: other})])  # type: ignore[name-defined]
