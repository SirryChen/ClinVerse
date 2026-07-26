"""Plot styling shared across statistics scripts."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from config import PLOT_FONT_FAMILY


def setup_plot_style() -> None:
    plt.rcParams["font.sans-serif"] = PLOT_FONT_FAMILY
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 150
    plt.rcParams["savefig.bbox"] = "tight"


def top_n_with_other(series, n: int = 12, other_label: str = "其他"):
    """Keep top-N categories; aggregate the rest."""
    counts = series.value_counts()
    if len(counts) <= n:
        return counts
    top = counts.head(n)
    other = counts.iloc[n:].sum()
    return pd.concat([top, pd.Series({other_label: other})])  # type: ignore[name-defined]
