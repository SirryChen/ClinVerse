#!/usr/bin/env python3
"""Run the full-information diagnosis baseline for a standard model suite."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_PROVIDERS = (
    "gpt-5.6-sol",
    "qwen3-8b-vllm",
    "qwen3-32b-vllm",
    "baichuan-m2-32b-vllm",
    "healthgpt-pro-4b-vllm",
    "healthgpt-pro-8b-vllm",
)

RUN_NAME = {
    "gpt-5.6-sol": "gpt-5.6-sol",
    "qwen3-8b-vllm": "qwen3-8b",
    "qwen3-32b-vllm": "qwen3-32b",
    "baichuan-m2-32b-vllm": "baichuan-m2-32b",
    "healthgpt-pro-4b-vllm": "healthgpt-pro-4b",
    "healthgpt-pro-8b-vllm": "healthgpt-pro-8b",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--cases-dir", default="data/cases")
    parser.add_argument("--source", default="mimic-iv-ext")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--sample-seed", type=int, default=20260717)
    parser.add_argument(
        "--providers",
        nargs="+",
        default=list(DEFAULT_PROVIDERS),
        help="llm_providers keys to evaluate (default: GPT plus the five-model local suite).",
    )
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue to the next provider if one model server is unavailable.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    evaluator = Path(__file__).with_name("run_full_info_diagnosis_eval.py")
    dataset_tag = args.source.removesuffix("-iv-ext").removesuffix("-ext")

    for provider in args.providers:
        model_tag = RUN_NAME.get(provider, provider.removesuffix("-vllm"))
        run_id = f"{dataset_tag}{args.n}_{model_tag}_full_info_diagnosis"
        command = [
            sys.executable,
            str(evaluator),
            "--config",
            args.config,
            "--cases-dir",
            args.cases_dir,
            "--source",
            args.source,
            "--n",
            str(args.n),
            "--shuffle",
            "--sample-seed",
            str(args.sample_seed),
            "--provider",
            provider,
            "--evidence-mode",
            "full-workup",
            "--run-id",
            run_id,
        ]
        if args.output_root:
            command.extend(["--output-root", str(Path(args.output_root) / run_id)])
        if args.resume:
            command.append("--resume")
        if args.retry_failed:
            command.append("--retry-failed")

        print(f"\n[{provider}] {' '.join(command)}", flush=True)
        if args.dry_run:
            continue
        result = subprocess.run(command, check=False)
        if result.returncode and not args.keep_going:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
