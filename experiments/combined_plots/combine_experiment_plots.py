#!/usr/bin/env python3
"""Combine the three experiment figures into one left-to-right PDF panel."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUTS = (
    REPO_ROOT / "experiment" / "information_collection_rate" / "results" / "information_collection_rate_by_turn.pdf",
    REPO_ROOT / "experiment" / "trajectory_diagnosis_probe" / "results" / "diagnosis_probe_agreement_by_step.pdf",
    REPO_ROOT / "experiment" / "action_distribution_by_turn" / "results" / "action_distribution_by_turn.pdf",
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "combined_experiment_figures.pdf"


def render_pdf(pdf_path: Path, temp_dir: Path, index: int):
    """Render a single PDF page at high resolution for a compact composite."""
    target = temp_dir / f"panel_{index}"
    subprocess.run(
        ["pdftoppm", "-png", "-f", "1", "-singlefile", "-r", "360", str(pdf_path), str(target)],
        check=True,
    )
    from PIL import Image

    with Image.open(f"{target}.png") as image:
        return image.convert("RGB").copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", help="Input PDFs in left-to-right order; supply exactly three to override defaults.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    inputs = tuple(args.input) if args.input else DEFAULT_INPUTS
    if len(inputs) != 3:
        parser.error("Provide exactly three --input PDFs.")
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        parser.error(f"Input PDF not found: {', '.join(missing)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Preserve panel aspect ratios while reducing the three figures to a
    # compact, publication-style single row.
    panel_height = 5.0
    panel_gap = 0.38
    label_band_height = 0.38
    with tempfile.TemporaryDirectory(prefix="combined_experiment_figures_") as directory:
        images = [render_pdf(path, Path(directory), index) for index, path in enumerate(inputs)]
    widths = [panel_height * image.width / image.height for image in images]
    figure_width = sum(widths) + panel_gap * (len(images) - 1)
    figure_height = panel_height + label_band_height
    fig = plt.figure(figsize=(figure_width, figure_height), dpi=300)
    cursor = 0.0
    for label, image, width in zip(("(a)", "(b)", "(c)"), images, widths):
        axis = fig.add_axes((
            cursor / figure_width,
            0,
            width / figure_width,
            panel_height / figure_height,
        ))
        axis.imshow(image)
        axis.axis("off")
        # Keep the label immediately above the panel's upper-left corner so
        # it identifies the panel without obscuring any plotted content.
        fig.text(
            cursor / figure_width,
            0.975,
            label,
            ha="left",
            va="top",
            fontsize=22,
            fontweight="bold",
            fontfamily="Times New Roman",
        )
        cursor += width + panel_gap
    fig.savefig(args.output, format="pdf", dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
