#!/usr/bin/env python3
"""Build enhanced cases: dataset adapter + Synthea matching pipeline.

Works with any registered dataset (see core/registry.py) via --dataset;
defaults to config/default.yaml's `default_dataset`.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

from core.config import load_yaml_config
from core.registry import get_adapter, known_datasets
from pipeline.synthea_parser import load_synthea_pool
from pipeline.case_builder import build_cases_batch
from pipeline.patient_info_units import OpenAIChatPatientInfoExtractor


def load_config(path: str) -> dict:
    return load_yaml_config(path)


def main():
    parser = argparse.ArgumentParser(description="Build enhanced dataset + Synthea cases.")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--dataset", default=None,
                        help=f"Dataset adapter to use, e.g. mimic-iv-ext, sicdb-ext. "
                             f"Known: {known_datasets()}. Defaults to config's default_dataset.")
    parser.add_argument("--limit", type=int, default=None, help="Max admissions to process")
    parser.add_argument("--output", default=None, help="Override cases output dir")
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--info-extractor-provider", default=None,
                        help="Optional llm_providers key used to extract patient_information_units.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    repo_root = Path(__file__).parent.parent

    dataset_name = args.dataset or cfg.get("default_dataset", "mimic-iv-ext")
    dataset_cfg  = cfg.get("datasets", {}).get(dataset_name, {})
    data_dir     = repo_root / dataset_cfg["data_dir"] if dataset_cfg.get("data_dir") else None

    synthea_dir  = repo_root / cfg["paths"]["synthea_pool"]
    cases_dir    = Path(args.output or cfg["paths"]["cases"])
    top_k        = cfg["pipeline"]["top_k"]
    min_score    = args.min_score if args.min_score is not None else cfg["pipeline"]["min_match_score"]
    info_extractor = None
    if args.info_extractor_provider:
        llm_cfg = cfg["llm_providers"][args.info_extractor_provider]
        info_extractor = OpenAIChatPatientInfoExtractor(
            api_key=llm_cfg["api_key"],
            base_url=llm_cfg["base_url"],
            model=llm_cfg.get("info_extractor_model")
            or llm_cfg.get("patient_actor_model")
            or llm_cfg["doctor_model"],
            timeout=llm_cfg.get("timeout", 120),
            extra_body=llm_cfg.get("extra_body"),
        )

    print(f"Loading Synthea pool from {synthea_dir} ...")
    personas = load_synthea_pool(synthea_dir)
    print(f"  {len(personas)} personas loaded")

    adapter = get_adapter(dataset_name, **({"data_dir": data_dir} if data_dir else {}))
    print(f"Processing {dataset_name} admissions (limit={args.limit}) ...")
    admissions = adapter.load_admissions(limit=args.limit)

    stats = build_cases_batch(
        admissions=tqdm(admissions, desc="cases"),
        personas=personas,
        cases_dir=cases_dir,
        top_k=top_k,
        min_score=min_score,
        verbose=False,
        adapter=adapter,
        info_extractor=info_extractor,
    )

    print(f"\nDone: {stats}")


if __name__ == "__main__":
    main()
