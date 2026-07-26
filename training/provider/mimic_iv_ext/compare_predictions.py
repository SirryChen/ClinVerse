"""
Compare base vs fine-tuned predictions with robust metrics.

Metrics (after excluding outlier test items):
  - MAPE          : mean |pred-true|/|true| on numeric pairs (true != 0)
  - MAE           : mean |pred-true| on numeric pairs
  - NormRefErr    : mean |pred-true| / (ref_upper - ref_lower) when range valid
  - AbnormalityAcc: balanced (macro) accuracy — equal weight per class
                    (abnormal recall + normal recall) / 2

Usage:
    python compare_predictions.py \\
        --finetuned output/predictions.jsonl \\
        --base output/predictions_base_5k.jsonl \\
        --max-samples 5000 \\
        --test-file output/test.jsonl \\
        --output output/baseline_comparison.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterator, Optional


def _is_missing(val) -> bool:
    if val is None:
        return True
    text = str(val).strip()
    return text in ("", "nan", "NaN", "None")


def _is_numeric(val: str) -> bool:
    try:
        float(val)
        return True
    except (ValueError, TypeError):
        return False


def _iter_records(path: Path, limit: Optional[int] = None) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            yield json.loads(line)


def _iter_meta(path: Optional[Path], limit: Optional[int] = None) -> Iterator[Optional[dict]]:
    if path is None or not path.exists():
        for _ in range(limit or 0):
            yield None
        return
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            ex = json.loads(line)
            yield ex.get("_meta", {})


def _ref_width(meta: Optional[dict], rec: Optional[dict] = None) -> Optional[float]:
    lo = hi = None
    if meta:
        lo, hi = meta.get("ref_range_lower"), meta.get("ref_range_upper")
    if rec and (_is_missing(lo) or _is_missing(hi)):
        lo = rec.get("ref_range_lower", lo)
        hi = rec.get("ref_range_upper", hi)
    if _is_missing(lo) or _is_missing(hi):
        return None
    width = float(hi) - float(lo)
    return width if width > 0 else None


def _detect_outlier_tests(
    records: list[dict],
    threshold_pct: float,
) -> tuple[set[str], dict[str, float]]:
    """Exclude tests whose per-test MAPE exceeds threshold."""
    by_test: dict[str, list[float]] = defaultdict(list)
    for rec in records:
        if rec.get("metric_type") != "numeric":
            continue
        ape = rec.get("abs_pct_error")
        if ape is None:
            t, p = float(rec["true_val"]), float(rec["pred_val"])
            if t == 0:
                continue
            ape = abs(t - p) / abs(t) * 100
        by_test[rec["test_name"]].append(float(ape))

    per_test_mape: dict[str, float] = {}
    excluded: set[str] = set()
    for name, errors in by_test.items():
        mape = sum(errors) / len(errors)
        per_test_mape[name] = round(mape, 2)
        if mape > threshold_pct:
            excluded.add(name)
    return excluded, per_test_mape


def _compute_metrics(
    records: list[dict],
    metas: list[Optional[dict]],
    excluded_tests: set[str],
    sample_mape_cap: Optional[float] = 200.0,
) -> dict:
    mape_vals: list[float] = []
    mae_vals: list[float] = []
    norm_ref_vals: list[float] = []
    abn_tp = abn_fn = abn_fp = abn_tn = 0

    for rec, meta in zip(records, metas):
        name = rec["test_name"]
        if name in excluded_tests:
            continue

        true_abn = bool(rec.get("is_abnormal_true", False))
        pred_abn = bool(rec.get("is_abnormal_pred", False))
        if true_abn and pred_abn:
            abn_tp += 1
        elif true_abn and not pred_abn:
            abn_fn += 1
        elif not true_abn and pred_abn:
            abn_fp += 1
        else:
            abn_tn += 1

        if rec.get("metric_type") != "numeric":
            continue
        if not (rec.get("true_is_numeric") and rec.get("pred_is_numeric")):
            continue

        t = float(rec["true_val"])
        p = float(rec["pred_val"])
        if t == 0:
            continue

        mae_vals.append(abs(t - p))
        ape = abs(t - p) / abs(t) * 100
        if sample_mape_cap is not None:
            ape = min(ape, sample_mape_cap)
        mape_vals.append(ape)

        width = _ref_width(meta, rec)
        if width is not None:
            norm_ref_vals.append(abs(t - p) / width)

    def _mean(xs: list[float]) -> Optional[float]:
        return round(sum(xs) / len(xs), 4) if xs else None

    def _median(xs: list[float]) -> Optional[float]:
        if not xs:
            return None
        s = sorted(xs)
        mid = len(s) // 2
        if len(s) % 2:
            return round(s[mid], 4)
        return round((s[mid - 1] + s[mid]) / 2, 4)

    abnormal_total = abn_tp + abn_fn
    normal_total = abn_fp + abn_tn
    abn_total = abnormal_total + normal_total
    abn_correct = abn_tp + abn_tn
    abnormal_recall = abn_tp / abnormal_total if abnormal_total else None
    normal_recall = abn_tn / normal_total if normal_total else None
    balanced_abn = (
        (abnormal_recall + normal_recall) / 2
        if abnormal_recall is not None and normal_recall is not None
        else None
    )

    return {
        "numeric_total": len(mape_vals),
        "norm_ref_total": len(norm_ref_vals),
        "abnormality_total": abn_total,
        "abnormality_abnormal_total": abnormal_total,
        "abnormality_normal_total": normal_total,
        "mape_pct": _mean(mape_vals),
        "mape_median_pct": _median(mape_vals),
        "mae": _mean(mae_vals),
        "normalized_ref_range_error": _mean(norm_ref_vals),
        "abnormality_accuracy": round(abn_correct / abn_total, 4) if abn_total else None,
        "abnormality_balanced_accuracy": round(balanced_abn, 4) if balanced_abn is not None else None,
        "abnormality_abnormal_recall": round(abnormal_recall, 4) if abnormal_recall is not None else None,
        "abnormality_normal_recall": round(normal_recall, 4) if normal_recall is not None else None,
        "abnormality_correct": abn_correct,
    }


def _parse_baseline_specs(args) -> list[tuple[str, Path]]:
    specs: list[tuple[str, Path]] = []
    if args.baselines:
        for item in args.baselines:
            if ":" not in item:
                raise ValueError(f"Invalid --baselines entry {item!r}; use name:path")
            name, path = item.split(":", 1)
            specs.append((name.strip(), Path(path.strip())))
    elif args.base:
        specs.append(("qwen3.5-2b", Path(args.base)))
    else:
        raise ValueError("Provide --baselines name:path ... or --base path")
    return specs


def _load_reference(
    finetuned_path: Path,
    test_path: Optional[Path],
    max_samples: Optional[int],
) -> tuple[list[dict], list[Optional[dict]]]:
    ft_records = list(_iter_records(finetuned_path, max_samples))
    metas = list(_iter_meta(test_path, len(ft_records)))
    if test_path and len(metas) < len(ft_records):
        metas.extend([None] * (len(ft_records) - len(metas)))
    elif not test_path or not test_path.exists():
        metas = [None] * len(ft_records)
    return ft_records, metas


def _load_paired_base(
    base_path: Path,
    reference: list[dict],
) -> list[dict]:
    base_records = list(_iter_records(base_path, len(reference)))
    if len(base_records) != len(reference):
        raise ValueError(
            f"Base file {base_path} has {len(base_records)} lines, "
            f"expected {len(reference)}"
        )
    for i, (ref, base) in enumerate(zip(reference, base_records)):
        if ref["test_name"] != base["test_name"] or ref["true_val"] != base["true_val"]:
            raise ValueError(
                f"Prediction mismatch at line {i} in {base_path}: "
                f"{ref['test_name']}/{ref['true_val']} vs {base['test_name']}/{base['true_val']}"
            )
    return base_records


def _summarize_side(
    label: str,
    records: list[dict],
    metas: list[Optional[dict]],
    excluded: set[str],
    sample_mape_cap: Optional[float],
) -> dict:
    metrics = _compute_metrics(records, metas, excluded, sample_mape_cap)
    return {"label": label, "excluded_tests_count": len(excluded), **metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare base vs fine-tuned lab predictions")
    parser.add_argument("--finetuned", required=True, help="Fine-tuned predictions JSONL")
    parser.add_argument("--base", default=None, help="[legacy] Qwen3.5-2B base predictions JSONL")
    parser.add_argument(
        "--baselines", nargs="*", default=None,
        help="Baseline specs: name:path (e.g. qwen3.5-2b:out/base_2b.jsonl qwen3.5-27b:out/base_27b.jsonl)",
    )
    parser.add_argument("--test-file", default=None, help="test.jsonl for reference ranges")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Compare on first N paired samples (default: all base lines)")
    parser.add_argument("--outlier-threshold", type=float, default=200.0,
                        help="Drop test items with per-test MAPE above this (default: 200)")
    parser.add_argument("--outlier-from", default=None,
                        help="Detect outlier tests from this JSONL (default: finetuned file)")
    parser.add_argument("--sample-mape-cap", type=float, default=200.0,
                        help="Cap per-sample MAPE for micro average (default: 200, 0=disable)")
    parser.add_argument("--output", default="output/baseline_comparison.json")
    parser.add_argument("--plot", default="output/metrics_comparison.png")
    args = parser.parse_args()

    ft_path = Path(args.finetuned)
    test_path = Path(args.test_file) if args.test_file else None
    baseline_specs = _parse_baseline_specs(args)

    limit = args.max_samples
    if limit is None:
        limit = min(sum(1 for _ in open(p, encoding="utf-8")) for _, p in baseline_specs)

    ft_records, metas = _load_reference(ft_path, test_path, limit)

    outlier_path = Path(args.outlier_from) if args.outlier_from else ft_path
    outlier_records = list(_iter_records(outlier_path, None))
    excluded, per_test_mape = _detect_outlier_tests(outlier_records, args.outlier_threshold)
    excluded_sorted = sorted(excluded, key=lambda n: per_test_mape.get(n, 0), reverse=True)

    sample_cap = args.sample_mape_cap if args.sample_mape_cap > 0 else None

    baselines: dict[str, dict] = {}
    for name, path in baseline_specs:
        base_records = _load_paired_base(path, ft_records)
        baselines[name] = _summarize_side(name, base_records, metas, excluded, sample_cap)

    result = {
        "n_samples": len(ft_records),
        "outlier_threshold_pct": args.outlier_threshold,
        "sample_mape_cap_pct": args.sample_mape_cap if sample_cap else None,
        "excluded_tests": excluded_sorted,
        "excluded_tests_count": len(excluded_sorted),
        "test_file_used": str(test_path) if test_path and test_path.exists() else None,
        "baselines": baselines,
        "finetuned": _summarize_side("finetuned", ft_records, metas, excluded, sample_cap),
    }

    # Backward-compatible single-base alias
    if len(baseline_specs) == 1:
        result["base"] = baselines[baseline_specs[0][0]]

    # Full finetuned set (same exclusion rules) when comparing a subset
    if len(outlier_records) > len(ft_records):
        full_metas = list(_iter_meta(test_path, len(outlier_records)))
        if len(full_metas) < len(outlier_records):
            full_metas.extend([None] * (len(outlier_records) - len(full_metas)))
        result["finetuned_full"] = _summarize_side(
            "finetuned_full", outlier_records, full_metas, excluded, sample_cap,
        )

    result["delta_finetuned_minus"] = {}
    f = result["finetuned"]
    for name, b in baselines.items():
        result["delta_finetuned_minus"][name] = {
            "mape_pct": _delta(f["mape_pct"], b["mape_pct"]),
            "mae": _delta(f["mae"], b["mae"]),
            "normalized_ref_range_error": _delta(
                f["normalized_ref_range_error"], b["normalized_ref_range_error"],
            ),
            "abnormality_balanced_accuracy": _delta(
                f["abnormality_balanced_accuracy"], b["abnormality_balanced_accuracy"],
                higher_better=True,
            ),
            "abnormality_accuracy": _delta(
                f["abnormality_accuracy"], b["abnormality_accuracy"], higher_better=True,
            ),
        }
    if len(baseline_specs) == 1:
        result["delta_finetuned_minus_base"] = result["delta_finetuned_minus"][baseline_specs[0][0]]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    _print_report(result)
    print(f"\nSaved → {out_path}")

    if args.plot:
        _plot_comparison(result, Path(args.plot))


def _delta(new, old, higher_better: bool = False) -> Optional[float]:
    if new is None or old is None:
        return None
    d = round(new - old, 4)
    return d if higher_better else round(-d, 4)  # positive = improvement for error metrics


def _print_report(result: dict) -> None:
    f = result["finetuned"]
    baselines = result["baselines"]
    deltas = result["delta_finetuned_minus"]
    print("\n" + "=" * 72)
    print(f"  Baselines vs Fine-tuned  (n={result['n_samples']:,}, "
          f"excluded {result['excluded_tests_count']} outlier tests @ MAPE>{result['outlier_threshold_pct']}%, "
          f"sample MAPE cap={result['sample_mape_cap_pct']}%)")
    print("=" * 72)

    for name, b in baselines.items():
        d = deltas[name]
        print(f"\n  --- {name} vs finetuned ---")
        print(f"  {'Metric':<32} {name:>14} {'Fine-tuned':>12} {'Δ improve':>12}")
        print("  " + "-" * 72)
        _row("MAPE (%)", b["mape_pct"], f["mape_pct"], d["mape_pct"], lower=True, col_w=14)
        _row("MAE", b["mae"], f["mae"], d["mae"], lower=True, col_w=14)
        _row("Norm. ref-range error", b["normalized_ref_range_error"],
             f["normalized_ref_range_error"], d["normalized_ref_range_error"], lower=True, col_w=14)
        _row("Abn. balanced acc", b["abnormality_balanced_accuracy"], f["abnormality_balanced_accuracy"],
             d["abnormality_balanced_accuracy"], lower=False, pct=True, col_w=14)
        _row("  └ abnormal recall", b["abnormality_abnormal_recall"], f["abnormality_abnormal_recall"],
             None, lower=False, pct=True, col_w=14)
        _row("  └ normal recall", b["abnormality_normal_recall"], f["abnormality_normal_recall"],
             None, lower=False, pct=True, col_w=14)

    print()
    if result["excluded_tests"]:
        print(f"  Excluded tests ({len(result['excluded_tests'])}):")
        for test_name in result["excluded_tests"][:10]:
            print(f"    - {test_name}")
        if len(result["excluded_tests"]) > 10:
            print(f"    ... +{len(result['excluded_tests']) - 10} more")
    if f.get("norm_ref_total", 0) == 0:
        print("\n  Note: NormRefErr needs --test-file with ref_range_lower/upper in _meta.")
    print("=" * 72)


def _row(name, base, ft, delta, lower: bool, pct: bool = False, col_w: int = 12):
    fmt = lambda x: "N/A" if x is None else (f"{x*100:.2f}%" if pct else f"{x:.4f}")
    d_fmt = "N/A" if delta is None else f"{delta:+.4f}"
    print(f"  {name:<32} {fmt(base):>{col_w}} {fmt(ft):>12} {d_fmt:>12}")


def _plot_comparison(result: dict, path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed; skipping plot")
        return

    f = result["finetuned"]
    series: list[tuple[str, dict]] = list(result["baselines"].items())
    series.append(("finetuned", f))
    labels = ["MAPE (%)", "MAE", "NormRefErr", "Abn.BalAcc"]
    colors = ["#94a3b8", "#64748b", "#2563eb"]

    def _vals(side: dict) -> list[float]:
        return [
            side["mape_pct"] or 0,
            side["mae"] or 0,
            side["normalized_ref_range_error"] or 0,
            (side["abnormality_balanced_accuracy"] or 0) * 100,
        ]

    x = np.arange(len(labels))
    n = len(series)
    w = 0.8 / n
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (name, side) in enumerate(series):
        offset = (i - (n - 1) / 2) * w
        vals = _vals(side)
        bars = ax.bar(x + offset, vals, w, label=name, color=colors[i % len(colors)])
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.1f}",
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(
        f"Metrics (outlier tests excluded, n={result['n_samples']:,})",
        fontsize=13,
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {path}")


if __name__ == "__main__":
    main()
