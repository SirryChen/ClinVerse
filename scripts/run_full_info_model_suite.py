#!/usr/bin/env python3
"""Run the full-information diagnosis baseline for a standard model suite."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_PROVIDERS = ("doctor",)
RUN_NAME = {}

DIAGNOSIS_TASKS = ("primary-diagnosis", "subtype-given-category")


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
        help="Configured doctor providers to evaluate.",
    )
    parser.add_argument(
        "--diagnosis-tasks",
        nargs="+",
        choices=DIAGNOSIS_TASKS,
        default=list(DIAGNOSIS_TASKS),
        help="Diagnosis experiment groups to run (default: baseline plus subtype given the known category).",
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

    for diagnosis_task in args.diagnosis_tasks:
        for provider in args.providers:
            model_tag = RUN_NAME.get(provider, provider.removesuffix("-vllm"))
            experiment_tag = (
                "full_info_diagnosis"
                if diagnosis_task == "primary-diagnosis"
                else "full_info_category_hint_subtype"
            )
            run_id = f"{dataset_tag}{args.n}_{model_tag}_{experiment_tag}"
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
                "--diagnosis-task",
                diagnosis_task,
                "--run-id",
                run_id,
            ]
            if args.output_root:
                command.extend(["--output-root", str(Path(args.output_root) / run_id)])
            if args.resume:
                command.append("--resume")
            if args.retry_failed:
                command.append("--retry-failed")

            print(f"\n[{diagnosis_task} / {provider}] {' '.join(command)}", flush=True)
            if args.dry_run:
                continue
            result = subprocess.run(command, check=False)
            if result.returncode and not args.keep_going:
                raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
