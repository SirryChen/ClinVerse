"""Primary diagnosis: Nature-style nested donut (major inner / minor outer).

Adapted from mimic-iv-ext/statistics/stats_diagnosis_category.py. SICdb is
ICD-10-GM only (see icd_utils.py), so there is no ICD-9/ICD-10 chapter-name
merging to do; titles are German (long_title comes straight from
primary_diagnosis_index.json, sourced from cases.ICD10MainText).
"""
from __future__ import annotations

import json
import re
from colorsys import hls_to_rgb, rgb_to_hls

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from config import ensure_output_dir, load_primary_diagnosis_df
from icd_utils import icd_major_chapter, icd_minor_label, load_icd_title_lookup
from plot_utils import setup_plot_style, top_n_with_other

CHAPTER_SHORT: dict[str, str] = {
    "某些传染病和寄生虫病": "感染/寄生虫",
    "肿瘤": "肿瘤",
    "血液及造血器官疾病和某些涉及免疫机制的疾患": "血液/免疫",
    "内分泌、营养和代谢疾病": "内分泌代谢",
    "精神和行为障碍": "精神行为",
    "神经系统疾病": "神经系统",
    "眼和附器疾病": "眼科",
    "耳和乳突疾病": "耳科",
    "循环系统疾病": "循环系统",
    "呼吸系统疾病": "呼吸系统",
    "消化系统疾病": "消化系统",
    "皮肤和皮下组织疾病": "皮肤",
    "肌肉骨骼系统和结缔组织疾病": "肌骨系统",
    "泌尿生殖系统疾病": "泌尿生殖",
    "妊娠、分娩和产褥期": "妊娠产褥",
    "围生期起源的某些情况": "围生期",
    "先天性畸形、变形和染色体异常": "先天畸形",
    "症状、体征和异常临床与实验室所见": "症状体征",
    "损伤、中毒和外因的某些其他后果": "损伤中毒",
    "新发疾病临时分类(如 COVID-19)": "新发疾病(U)",
    "疾病和死亡的外因": "外因(V-Y)",
    "影响健康状态和与保健机构接触的因素": "健康因素(Z)",
    "假体/植入物并发症补充编码": "假体并发症",
    "未知": "未知",
    "其他": "其他",
}

# Nature Publishing Group–inspired categorical palette
NATURE_COLORS = [
    "#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F",
    "#8491B4", "#91D1C2", "#DC0000", "#7E6148", "#B09C85",
    "#E18727", "#20854E",
]

FONT_CJK = "PingFang SC"


def _short_chapter(name: str) -> str:
    return CHAPTER_SHORT.get(name, name[:8] + ("…" if len(name) > 8 else ""))


def _short_minor(label: str, max_len: int = 22) -> str:
    """'[I71] Dissektion der Aorta...' -> 'I71 · Dissektion der Aorta'"""
    m = re.match(r"\[([^\]]+)\]\s*(.+)", label)
    if m:
        code, title = m.group(1), m.group(2)
        title = title.split(",")[0].strip()
        if len(title) > max_len - len(code) - 3:
            title = title[: max_len - len(code) - 6] + "…"
        return f"{code} · {title}"
    return label[:max_len] + ("…" if len(label) > max_len else "")


def _shade(hex_color: str, lightness_delta: float) -> tuple:
    r, g, b = tuple(c / 255 for c in bytes.fromhex(hex_color.lstrip("#")))
    h, l, s = rgb_to_hls(r, g, b)
    l = max(0.12, min(0.92, l + lightness_delta))
    return hls_to_rgb(h, l, s)


def _adjust_positions(y_values: list[float], low: float, high: float, min_gap: float) -> list[float]:
    """Simple collision-avoidance for label y positions on one side."""
    if not y_values:
        return []
    ys = sorted(y_values)
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + min_gap)
    overflow = ys[-1] - high
    if overflow > 0:
        ys = [y - overflow for y in ys]
        for i in range(len(ys) - 2, -1, -1):
            ys[i] = min(ys[i], ys[i + 1] - min_gap)
    underflow = low - ys[0]
    if underflow > 0:
        ys = [y + underflow for y in ys]
    return ys


def _build_hierarchy(df: pd.DataFrame, top_majors: int = 10, top_minors: int = 3):
    df = df.copy()
    df["major"] = df["major_chapter"]

    major_counts = df["major"].value_counts()
    top_major_names = major_counts.head(top_majors).index.tolist()
    df["major_plot"] = df["major"].where(df["major"].isin(top_major_names), "其他")

    inner_labels: list[str] = []
    inner_sizes: list[int] = []
    inner_colors: list[str] = []

    outer_sizes: list[int] = []
    outer_colors: list[tuple] = []
    outer_meta: list[dict] = []

    major_order = (
        df.groupby("major_plot").size().sort_values(ascending=False).index.tolist()
    )

    for mi, major in enumerate(major_order):
        base = NATURE_COLORS[mi % len(NATURE_COLORS)]
        sub = df[df["major_plot"] == major]
        n_major = len(sub)
        inner_labels.append(_short_chapter(major))
        inner_sizes.append(n_major)
        inner_colors.append(base)

        minors = sub["minor_label"].value_counts()
        top = minors.head(top_minors)
        other_minor = n_major - int(top.sum())

        shades = np.linspace(0.22, -0.08, top_minors + (1 if other_minor > 0 else 0))
        for si, (minor, cnt) in enumerate(top.items()):
            outer_sizes.append(int(cnt))
            outer_colors.append(_shade(base, float(shades[si])))
            outer_meta.append(
                {
                    "major": major,
                    "minor": minor,
                    "short": _short_minor(minor),
                    "pct": cnt / n_major * 100,
                    "color": base,
                }
            )
        if other_minor > 0:
            outer_sizes.append(other_minor)
            outer_colors.append(_shade(base, float(shades[-1])))
            outer_meta.append(
                {
                    "major": major,
                    "minor": None,
                    "short": "其他小类",
                    "pct": other_minor / n_major * 100,
                    "color": base,
                }
            )

    return inner_labels, inner_sizes, inner_colors, outer_sizes, outer_colors, outer_meta


def _plot_nature_nested_donut(
    inner_labels: list[str],
    inner_sizes: list[int],
    inner_colors: list[str],
    outer_sizes: list[int],
    outer_colors: list[tuple],
    outer_meta: list[dict],
    n_total: int,
    out_path: str,
) -> None:
    setup_plot_style()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [FONT_CJK, "Arial Unicode MS", "Heiti SC", "Microsoft YaHei", "DejaVu Sans"],
            "font.size": 9,
            "axes.facecolor": "#FAFAFA",
            "figure.facecolor": "#FFFFFF",
        }
    )

    fig = plt.figure(figsize=(14, 9), facecolor="#FFFFFF")
    ax = fig.add_axes([0.02, 0.06, 0.58, 0.88])
    ax.set_facecolor("#FAFAFA")

    inner_radius = 0.52
    ring_width = 0.34
    gap = 0.018

    wedges_in, _ = ax.pie(
        inner_sizes,
        radius=inner_radius,
        colors=inner_colors,
        startangle=90,
        counterclock=False,
        wedgeprops={
            "width": inner_radius * 0.62,
            "edgecolor": "#FFFFFF",
            "linewidth": 1.8,
            "antialiased": True,
        },
    )

    ax.pie(
        outer_sizes,
        radius=inner_radius + ring_width,
        colors=outer_colors,
        startangle=90,
        counterclock=False,
        wedgeprops={
            "width": ring_width - gap,
            "edgecolor": "#FFFFFF",
            "linewidth": 1.0,
            "antialiased": True,
        },
    )

    total = sum(inner_sizes)
    major_infos = []
    for wedge, size, label, color in zip(wedges_in, inner_sizes, inner_labels, inner_colors):
        pct = size / total * 100
        if pct < 3.0:
            continue
        mid = (wedge.theta1 + wedge.theta2) / 2
        rad = np.deg2rad(mid)
        major_infos.append(
            {
                "mid": mid,
                "rad": rad,
                "pct": pct,
                "label": label,
                "color": color,
                "anchor_x": (inner_radius + ring_width) * np.cos(rad),
                "anchor_y": (inner_radius + ring_width) * np.sin(rad),
                "side": "right" if np.cos(rad) >= 0 else "left",
            }
        )

    left = [m for m in major_infos if m["side"] == "left"]
    right = [m for m in major_infos if m["side"] == "right"]
    y_low, y_high = -1.08, 1.08
    left_targets = _adjust_positions([m["anchor_y"] for m in left], y_low, y_high, 0.12)
    right_targets = _adjust_positions([m["anchor_y"] for m in right], y_low, y_high, 0.12)

    for group, ys, x_text, ha in [
        (left, left_targets, -1.02, "right"),
        (right, right_targets, 1.00, "left"),
    ]:
        for m, y_t in zip(sorted(group, key=lambda x: x["anchor_y"]), ys):
            x0, y0 = m["anchor_x"], m["anchor_y"]
            x1 = 1.03 if x_text > 0 else -1.03
            ax.plot([x0, x1, x_text - (0.02 if x_text > 0 else -0.02)], [y0, y_t, y_t], color="#9CA3AF", lw=0.9)
            ax.text(
                x_text,
                y_t,
                f"{m['label']}  {m['pct']:.1f}%",
                ha=ha,
                va="center",
                fontsize=(11.6 if m["pct"] >= 8 else 11.9),
                fontweight="semibold" if m["pct"] >= 10 else "normal",
                color="#111827",
            )

    hub = plt.Circle((0, 0), inner_radius * 0.28, fc="#FFFFFF", ec="#E5E7EB", lw=1.2, zorder=5)
    ax.add_patch(hub)
    ax.text(0, 0.07, f"{n_total:,}", ha="center", va="center", fontsize=19, fontweight="bold", color="#111827", zorder=6)
    ax.text(0, -0.10, "住院例数", ha="center", va="center", fontsize=9.2, color="#4B5563", zorder=6)
    ax.text(0, -0.22, "主诊断层级", ha="center", va="center", fontsize=8.2, color="#9CA3AF", zorder=6)

    ax.set_aspect("equal")
    ax.set_xlim(-1.30, 1.30)
    ax.set_ylim(-1.35, 1.35)
    ax.axis("off")

    ax_leg = fig.add_axes([0.62, 0.08, 0.36, 0.84])
    ax_leg.set_xlim(0, 1)
    ax_leg.set_ylim(0, 1)
    ax_leg.axis("off")

    ax_leg.text(0, 0.98, "ICD-10-GM 主诊断层级", fontsize=13.5, fontweight="bold", color="#111827", va="top")
    ax_leg.text(
        0, 0.945,
        "内环：疾病大类  ·  外环：三位码小类（每大类 Top 3 + 其他）",
        fontsize=8.4, color="#6B7280", va="top",
    )

    majors_seen: list[str] = []
    for m in outer_meta:
        if m["major"] not in majors_seen:
            majors_seen.append(m["major"])

    col_w = 0.48
    rows_per_col = (len(majors_seen) + 1) // 2
    row_step = 0.88 / max(rows_per_col, 1)

    major_blocks: dict[str, list[dict]] = {}
    for meta in outer_meta:
        major_blocks.setdefault(meta["major"], []).append(meta)

    for idx, major in enumerate(majors_seen):
        col = idx // rows_per_col
        row = idx % rows_per_col
        x0 = col * col_w
        y_top = 0.90 - row * row_step
        base = major_blocks[major][0]["color"]
        major_total = sum(m["pct"] for m in major_blocks[major])

        patch = mpatches.FancyBboxPatch(
            (x0, y_top - 0.028), 0.022, 0.022,
            boxstyle="round,pad=0.003", facecolor=base, edgecolor="#FFFFFF",
            linewidth=1.0, transform=ax_leg.transData, clip_on=False,
        )
        ax_leg.add_patch(patch)
        ax_leg.text(
            x0 + 0.028, y_top - 0.017,
            f"{_short_chapter(major)} ({major_total:.1f}%)",
            fontsize=8.4, fontweight="bold", color="#111827", va="center",
        )

        y_sub = y_top - 0.042
        for meta in major_blocks[major]:
            minor_patch = mpatches.Rectangle(
                (x0 + 0.012, y_sub - 0.011), 0.012, 0.012,
                facecolor=_shade(meta["color"], 0.15 if meta["minor"] else -0.05),
                edgecolor="none", transform=ax_leg.transData,
            )
            ax_leg.add_patch(minor_patch)
            ax_leg.text(
                x0 + 0.028, y_sub - 0.005,
                f"{meta['short']} {meta['pct']:.1f}%",
                fontsize=7.0, color="#374151", va="center",
            )
            y_sub -= 0.028

    fig.text(
        0.02, 0.015,
        "SICdb-Ext 已接受队列  ·  Top 10 疾病大类 + 其他  ·  各大类 Top 3 小类",
        fontsize=7.5, color="#9CA3AF",
    )

    fig.savefig(out_path, facecolor="#FFFFFF", edgecolor="none")
    plt.close(fig)


def main() -> None:
    out = ensure_output_dir()
    df = load_primary_diagnosis_df()
    lookup = load_icd_title_lookup()

    df["major_chapter"] = df.apply(
        lambda r: icd_major_chapter(r["icd_code"], int(r["icd_version"])), axis=1
    )
    df["minor_label"] = df.apply(
        lambda r: icd_minor_label(r["icd_code"], int(r["icd_version"]), lookup), axis=1
    )

    inner_labels, inner_sizes, inner_colors, outer_sizes, outer_colors, outer_meta = (
        _build_hierarchy(df, top_majors=10, top_minors=3)
    )

    combined_path = f"{out}/diagnosis_category_nature.png"
    _plot_nature_nested_donut(
        inner_labels, inner_sizes, inner_colors,
        outer_sizes, outer_colors, outer_meta,
        len(df), combined_path,
    )

    major_counts = top_n_with_other(df["major_chapter"], n=15)
    minor_counts = top_n_with_other(df["minor_label"], n=15)

    summary = {
        "n_admissions": len(df),
        "n_major_chapters": int(df["major_chapter"].nunique()),
        "n_minor_categories": int(df["minor_label"].nunique()),
        "major_chapter_counts": major_counts.to_dict(),
        "minor_category_top20": df["minor_label"].value_counts().head(20).to_dict(),
        "icd_version_mix": df["icd_version"].value_counts().astype(int).to_dict(),
        "figure": "diagnosis_category_nature.png",
        "figure_description": "Nested donut: inner=major chapter (ICD-10-GM), outer=top-3 minor per chapter",
    }
    with open(f"{out}/diagnosis_category_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[diagnosis] {len(df):,} admissions → {combined_path}")


if __name__ == "__main__":
    main()
