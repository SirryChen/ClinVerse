"""
Evaluate fine-tuned lab-provider model on test.jsonl.

Usage:
    python evaluate.py --adapter output/checkpoints/final_adapter [--config config.yaml]
                        [--predictions-output output/predictions.jsonl]
    python evaluate.py --base-only [--baseline qwen3.5-2b|qwen3.5-27b|deepseek-v4-flash] [--max-samples 5000]
    python evaluate.py --base-only --model-path ../../../weight/Qwen3.5-27B
    export INF_API_KEY=...   # API 基线 deepseek-v4-flash 需要

Lab results fall into three types, each with its own metric:
  1. Numeric   : per-test MAPE (unit-agnostic, computed within each test name)
  2. Categorical: exact-match accuracy  (NEG. / POS. / TRACE / MANY …)
  3. Abnormality: inferred by comparing predicted value against reference range
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
from urllib import error, request

import yaml

log = logging.getLogger(__name__)


# ── Output parser ──────────────────────────────────────────────────────────────

_ABNORMAL_RE = re.compile(r"\[ABNORMAL\]", re.IGNORECASE)
_NUMERIC_RE  = re.compile(r"^[+-]?\d+\.?\d*$")


def _is_missing(val) -> bool:
    if val is None:
        return True
    text = str(val).strip()
    return text in ("", "nan", "NaN", "None")


def parse_output(text: str) -> str:
    """Return the predicted/label value string (strips legacy [ABNORMAL] tags)."""
    text = text.strip()
    return _ABNORMAL_RE.sub("", text).strip()


def infer_abnormal(pred_val: str, meta: dict) -> bool:
    """Infer abnormality from numeric prediction vs reference range."""
    if not is_numeric(pred_val):
        return False
    lo, hi = meta.get("ref_range_lower"), meta.get("ref_range_upper")
    if _is_missing(lo) and _is_missing(hi):
        return bool(meta.get("is_abnormal", False))
    val = float(pred_val)
    if not _is_missing(lo) and val < float(lo):
        return True
    if not _is_missing(hi) and val > float(hi):
        return True
    return False


def is_numeric(value: str) -> bool:
    """True if value can be parsed as a plain float."""
    try:
        float(value)
        return True
    except ValueError:
        return False


# ── Inference ──────────────────────────────────────────────────────────────────

_CHAT_TURN_END_TOKENS = ("<|im_end|>", "<|eot_id|>", "<|end|>")


def apply_chat_template(tokenizer, messages, *, enable_thinking: bool = False, **kwargs) -> str:
    """Apply chat template; disable Qwen3 thinking mode by default for lab-value generation."""
    sig = inspect.signature(tokenizer.apply_chat_template)
    accepts_template_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in sig.parameters.values()
    )
    if "enable_thinking" in sig.parameters or accepts_template_kwargs:
        kwargs["enable_thinking"] = enable_thinking
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        # Older tokenizers may not accept chat-template kwargs.
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def _resolve_eos_ids(tokenizer) -> list[int]:
    """Collect every plausible end-of-turn token id.

    Chat templates (e.g. ChatML's <|im_end|>) often use a turn-end token
    that differs from tokenizer.eos_token_id. If generate() only stops on
    the latter, it will run to max_new_tokens and start hallucinating
    further turns once the real answer is done.
    """
    ids = set()
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    vocab = tokenizer.get_vocab()
    for tok in _CHAT_TURN_END_TOKENS:
        if tok in vocab:
            ids.add(vocab[tok])
    return sorted(ids)


def run_inference(
    examples: list[dict],
    model,
    tokenizer,
    max_new_tokens: int = 32,
    batch_size: int = 16,
) -> list[str]:
    import torch

    predictions: list[str] = []
    device = next(model.parameters()).device
    eos_ids = _resolve_eos_ids(tokenizer)
    log.info("Generation will stop on token id(s): %s", eos_ids)

    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        prompts = [
            apply_chat_template(
                tokenizer,
                [m for m in ex["messages"] if m["role"] != "assistant"],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for ex in batch
        ]
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,        # greedy
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_ids,
            )

        input_len = enc["input_ids"].shape[1]
        for seq in out:
            new_tokens = seq[input_len:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            predictions.append(text)

        if (start // batch_size) % 10 == 0:
            log.info("Inference: %d / %d", start + len(batch), len(examples))

    return predictions


def _chat_completion(
    *,
    api_base: str,
    api_key: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> str:
    url = api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError(f"Empty API response: {body}")
    return (choices[0].get("message") or {}).get("content") or ""


def run_api_inference(
    examples: list[dict],
    *,
    api_base: str,
    api_key: str,
    model: str,
    max_tokens: int = 32,
    temperature: float = 0.0,
    concurrency: int = 16,
    timeout: float = 120.0,
    max_retries: int = 5,
) -> list[str]:
    """Call OpenAI-compatible chat/completions API for each example."""
    predictions: list[Optional[str]] = [None] * len(examples)

    def _infer_one(idx: int, ex: dict) -> tuple[int, str]:
        messages = [m for m in ex["messages"] if m["role"] != "assistant"]
        last_err: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                text = _chat_completion(
                    api_base=api_base,
                    api_key=api_key,
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=timeout,
                )
                return idx, text.strip()
            except (error.HTTPError, error.URLError, TimeoutError, RuntimeError) as exc:
                last_err = exc
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"API failed after {max_retries} retries: {last_err}")

    log.info(
        "API inference: model=%s concurrency=%d examples=%d",
        model, concurrency, len(examples),
    )
    done = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(_infer_one, i, ex): i for i, ex in enumerate(examples)
        }
        for fut in as_completed(futures):
            idx, text = fut.result()
            predictions[idx] = text
            done += 1
            if done % 100 == 0 or done == len(examples):
                log.info("API inference: %d / %d", done, len(examples))

    return [p if p is not None else "" for p in predictions]


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(
    examples: list[dict],
    predictions: list[str],
) -> tuple[dict, list[dict]]:
    """
    Compute three sets of metrics:
      - abnormality_accuracy : binary flag prediction (all examples)
      - categorical_accuracy : exact match on non-numeric values
      - per_test_mape        : mean absolute percentage error, grouped by test name

    Also returns a per-example `records` list (true/pred values + how each
    example was classified) so individual predictions can be inspected later.
    """
    # Per-group accumulators
    abn_correct = abn_total = 0
    abn_tp = abn_fn = abn_fp = abn_tn = 0
    cat_correct = cat_total = 0

    # numeric: {test_name: [(true_val, pred_val), ...]}
    numeric_by_test: dict[str, list[tuple[float, float]]] = defaultdict(list)
    # all numeric for global MAPE
    all_numeric: list[tuple[float, float]] = []

    records: list[dict] = []

    for ex, pred_text in zip(examples, predictions):
        meta       = ex.get("_meta", {})
        test_name  = meta.get("test_name", "unknown")
        true_text  = ex["messages"][-1]["content"]   # assistant turn = ground truth

        true_val = parse_output(true_text)
        pred_val = parse_output(pred_text)

        # ── 1. Abnormality (numeric: compare against ref range) ───────────────
        true_abn = bool(meta.get("is_abnormal", False))
        pred_abn = infer_abnormal(pred_val, meta)
        abn_total += 1
        if true_abn == pred_abn:
            abn_correct += 1
        if true_abn and pred_abn:
            abn_tp += 1
        elif true_abn and not pred_abn:
            abn_fn += 1
        elif not true_abn and pred_abn:
            abn_fp += 1
        else:
            abn_tn += 1

        record = {
            "test_name":         test_name,
            "true_text":         true_text,
            "pred_text":         pred_text,
            "true_val":          true_val,
            "pred_val":          pred_val,
            "true_is_numeric":   is_numeric(true_val),
            "pred_is_numeric":   is_numeric(pred_val),
            "is_abnormal_true":  true_abn,
            "is_abnormal_pred":  pred_abn,
            "abnormal_correct":  true_abn == pred_abn,
            "ref_range_lower":   meta.get("ref_range_lower"),
            "ref_range_upper":   meta.get("ref_range_upper"),
        }

        # ── 2. Numeric vs categorical ─────────────────────────────────────────
        if record["true_is_numeric"] and record["pred_is_numeric"]:
            t = float(true_val)
            p = float(pred_val)
            if t != 0:
                numeric_by_test[test_name].append((t, p))
                all_numeric.append((t, p))
                record["metric_type"]   = "numeric"
                record["abs_pct_error"] = round(abs(t - p) / abs(t) * 100, 2)
            else:
                # true value is exactly 0 → % error undefined, excluded from both
                # the numeric MAPE and the categorical accuracy denominators
                record["metric_type"] = "numeric_zero_excluded"
        else:
            # Categorical: compare normalised strings
            cat_total += 1
            is_match = true_val.upper().strip(".") == pred_val.upper().strip(".")
            if is_match:
                cat_correct += 1
            record["metric_type"]          = "categorical"
            record["categorical_correct"]  = is_match

        records.append(record)

    # ── Per-test MAPE ─────────────────────────────────────────────────────────
    per_test_mape: dict[str, float] = {}
    per_test_n: dict[str, int] = {}
    for test, pairs in sorted(numeric_by_test.items()):
        mape = sum(abs(t - p) / abs(t) for t, p in pairs) / len(pairs) * 100
        per_test_mape[test] = round(mape, 2)
        per_test_n[test] = len(pairs)

    global_mape = (
        sum(abs(t - p) / abs(t) for t, p in all_numeric) / len(all_numeric) * 100
        if all_numeric else None
    )

    abnormal_total = abn_tp + abn_fn
    normal_total = abn_fp + abn_tn
    abnormal_recall = abn_tp / abnormal_total if abnormal_total else None
    normal_recall = abn_tn / normal_total if normal_total else None
    balanced_abn = (
        (abnormal_recall + normal_recall) / 2
        if abnormal_recall is not None and normal_recall is not None
        else None
    )

    metrics = {
        "n_total": len(examples),
        # ── Primary metric ──────────────────────────────────────────────────
        "abnormality_accuracy": round(abn_correct / abn_total, 4) if abn_total else None,
        "abnormality_balanced_accuracy": round(balanced_abn, 4) if balanced_abn is not None else None,
        "abnormality_abnormal_recall": round(abnormal_recall, 4) if abnormal_recall is not None else None,
        "abnormality_normal_recall": round(normal_recall, 4) if normal_recall is not None else None,
        "abnormality_abnormal_total": abnormal_total,
        "abnormality_normal_total": normal_total,
        "abnormality_correct":  abn_correct,
        "abnormality_total":    abn_total,
        # ── Categorical ─────────────────────────────────────────────────────
        "categorical_accuracy": round(cat_correct / cat_total, 4) if cat_total else None,
        "categorical_correct":  cat_correct,
        "categorical_total":    cat_total,
        # ── Numeric ─────────────────────────────────────────────────────────
        "numeric_global_mape":  round(global_mape, 2) if global_mape is not None else None,
        "numeric_total":        len(all_numeric),
        "per_test_mape":        per_test_mape,       # unit-agnostic, per test type
        "per_test_n":           per_test_n,          # sample count backing each MAPE above
    }
    return metrics, records


def print_report(metrics: dict) -> None:
    print("\n" + "=" * 60)
    print("  Lab Prediction Evaluation Report")
    print("=" * 60)
    print(f"  Total examples     : {metrics['n_total']}")
    print()
    print(f"  [Primary] Abnormality balanced acc : {metrics['abnormality_balanced_accuracy']:.4f}"
          f"  (abn recall {metrics['abnormality_abnormal_recall']:.4f}"
          f" n={metrics['abnormality_abnormal_total']},"
          f"  normal recall {metrics['abnormality_normal_recall']:.4f}"
          f" n={metrics['abnormality_normal_total']})")
    print(f"  Abnormality overall acc          : {metrics['abnormality_accuracy']:.4f}"
          f"  ({metrics['abnormality_correct']} / {metrics['abnormality_total']})")
    print()
    if metrics["categorical_total"]:
        print(f"  Categorical accuracy : {metrics['categorical_accuracy']:.4f}"
              f"  ({metrics['categorical_correct']} / {metrics['categorical_total']})")
    if metrics["numeric_global_mape"] is not None:
        print(f"  Numeric global MAPE  : {metrics['numeric_global_mape']:.2f}%"
              f"  (n={metrics['numeric_total']})")
    print()
    if metrics["per_test_mape"]:
        per_test_n = metrics.get("per_test_n", {})
        print("  Per-test MAPE (top 20 by sample count):")
        # Sort by number of samples (descending)
        sorted_tests = sorted(
            metrics["per_test_mape"].items(),
            key=lambda kv: per_test_n.get(kv[0], 0),
            reverse=True,
        )[:20]
        for name, mape in sorted_tests:
            n = per_test_n.get(name, 0)
            print(f"    {name:<40s} {mape:>7.2f}%   (n={n})")
    print("=" * 60)


# ── Model config ───────────────────────────────────────────────────────────────

def resolve_eval_model(cfg: dict, args) -> tuple[dict, str, int]:
    """Return (model_cfg, run_label, batch_size) for this evaluation run."""
    default_batch = args.batch_size

    if args.adapter:
        model_cfg = dict(cfg["model"])
        label = "finetuned"
        batch_size = default_batch
        return model_cfg, label, batch_size

    if args.model_path:
        model_cfg = dict(cfg["model"])
        model_cfg["name_or_path"] = args.model_path
        label = Path(args.model_path).name
        batch_size = default_batch
        return model_cfg, label, batch_size

    baseline_name = args.baseline or "qwen3.5-2b"
    baselines = cfg.get("baselines") or {}
    if baseline_name in baselines:
        bl = baselines[baseline_name]
        if bl.get("backend") == "api":
            model_cfg = dict(bl)
            batch_size = default_batch
            return model_cfg, baseline_name, batch_size
        model_cfg = dict(cfg["model"])
        model_cfg.update({k: v for k, v in bl.items() if k != "batch_size"})
        batch_size = bl.get("batch_size", default_batch)
        return model_cfg, baseline_name, batch_size

    if args.base_only:
        model_cfg = dict(cfg["model"])
        return model_cfg, "base", default_batch

    raise ValueError("Unreachable: adapter or base-only must be set")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Evaluate lab-provider model on test set")
    parser.add_argument("--adapter",     default=None, help="Path to LoRA adapter directory")
    parser.add_argument("--base-only",   action="store_true",
                        help="Evaluate base model only (no LoRA adapter)")
    parser.add_argument("--baseline",    default=None,
                        help="Baseline name from config.baselines (default: qwen3.5-2b)")
    parser.add_argument("--model-path",  default=None,
                        help="Override model weight path (implies --base-only)")
    parser.add_argument("--config",      default="config.yaml")
    parser.add_argument("--test-file",   default=None, help="Override test JSONL path")
    parser.add_argument("--batch-size",  type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of test examples (for quick smoke test)")
    parser.add_argument("--output",      default=None,
                        help="Save full metrics JSON to this path")
    parser.add_argument("--predictions-output", default=None,
                        help="Save per-example true/predicted values to this JSONL path")
    args = parser.parse_args()
    if args.model_path:
        args.base_only = True
    if args.base_only and args.adapter:
        parser.error("Use either --base-only or --adapter, not both")
    if not args.base_only and not args.adapter:
        parser.error("Specify --adapter, --base-only, or --model-path")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    test_path = Path(args.test_file or cfg["data"]["test_file"])
    if not test_path.exists():
        sys.exit(f"Test file not found: {test_path}\nRun: python prepare_data.py")

    log.info("Loading test data from %s", test_path)
    examples: list[dict] = []
    with open(test_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.max_samples and i >= args.max_samples:
                break
            examples.append(json.loads(line))
    log.info("Loaded %d test examples", len(examples))

    model_cfg, run_label, batch_size = resolve_eval_model(cfg, args)

    # ── Inference ──────────────────────────────────────────────────────────────
    if model_cfg.get("backend") == "api":
        api_key_env = model_cfg.get("api_key_env", "INF_API_KEY")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            sys.exit(
                f"API baseline [{run_label}] requires env var {api_key_env}\n"
                f"Example: export {api_key_env}=your_key"
            )
        log.info("Running API inference [%s] …", run_label)
        predictions = run_api_inference(
            examples,
            api_base=model_cfg["api_base"],
            api_key=api_key,
            model=model_cfg["model"],
            max_tokens=model_cfg.get("max_tokens", 32),
            temperature=model_cfg.get("temperature", 0.0),
            concurrency=model_cfg.get("concurrency", args.batch_size),
        )
        model_path = model_cfg.get("api_base", "")
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = Path(model_cfg["name_or_path"])
        if not model_path.exists():
            sys.exit(
                f"Model weights not found: {model_path.resolve()}\n"
                "Check config.baselines or pass --model-path"
            )
        dtype = getattr(torch, model_cfg.get("torch_dtype", "bfloat16"))
        trust = model_cfg.get("trust_remote_code", True)

        log.info("Loading model [%s] from %s (batch_size=%d)", run_label, model_path, batch_size)
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=trust,
                                                  padding_side="left")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            str(model_path), device_map="auto", torch_dtype=dtype, trust_remote_code=trust,
        )
        if args.adapter:
            from peft import PeftModel
            log.info("Loading LoRA adapter from %s", args.adapter)
            model = PeftModel.from_pretrained(model, str(args.adapter))
        else:
            log.info("Evaluating base model only (no LoRA adapter)")
        model.eval()

        log.info("Running inference …")
        predictions = run_inference(examples, model, tokenizer, batch_size=batch_size)
        model_path = str(model_path)

    # ── Metrics ────────────────────────────────────────────────────────────────
    metrics, records = compute_metrics(examples, predictions)
    metrics["model_label"] = run_label
    metrics["model_path"] = str(model_path)
    if model_cfg.get("backend") == "api":
        metrics["backend"] = "api"
        metrics["api_model"] = model_cfg.get("model")
    print_report(metrics)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        log.info("Metrics saved → %s", out_path)

    if args.predictions_output:
        pred_path = Path(args.predictions_output)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pred_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log.info("Per-example predictions saved → %s", pred_path)


if __name__ == "__main__":
    main()
