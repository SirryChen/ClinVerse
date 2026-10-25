#!/usr/bin/env python3
"""Run a single simulation session and print results."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import load_yaml_config
from simulation.session import SimulationSession


def load_config(path: str) -> dict:
    return load_yaml_config(path)


def resolve_config_path(config_path: str | Path, value: str) -> Path:
    base_dir = Path(config_path).resolve().parent.parent
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path)


def main():
    parser = argparse.ArgumentParser(description="Run a simulation on one case.")
    parser.add_argument("--case", required=True, help="Path to case JSON file or case_id")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--output", default=None, help="Save result to this JSON file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    repo_root = Path(__file__).parent.parent

    case_path = Path(args.case)
    if not case_path.exists():
        # try interpreting as case_id under cases dir
        cases_dir = resolve_config_path(args.config, cfg["paths"]["cases"])
        case_path = cases_dir / f"{args.case}.json"

    datasets_config = {
        name: {"data_dir": str((repo_root / cfg_entry["data_dir"]).resolve())}
        for name, cfg_entry in cfg.get("datasets", {}).items()
        if cfg_entry.get("data_dir")
    }

    session = SimulationSession.from_case_file(
        case_path=case_path,
        patient_openai_config=cfg["llm_providers"][cfg["default_patient_provider"]],
        doctor_openai_config=cfg["llm_providers"][cfg["default_doctor_provider"]],
        lab_openai_config=cfg["llm_providers"][cfg["default_lab_provider"]],
        datasets_config=datasets_config,
        max_turns=cfg["simulation"]["max_turns"],
        language=cfg["simulation"].get("language", "en"),
    )

    print(f"Running simulation for case {case_path.stem} ...")
    result = session.run()

    scores = result["scores"]
    print(f"\n=== SCORES ===")
    for k, v in scores.items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v}")
    print(f"  ground_truth: {scores.get('ground_truth')}")
    print(f"  predicted:    {scores.get('predicted_diagnosis')}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nResult saved to {out_path}")


if __name__ == "__main__":
    main()
