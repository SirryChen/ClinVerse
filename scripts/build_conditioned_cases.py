#!/usr/bin/env python3
"""Construct conditioned ClinPatient cases from normalized admissions."""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.config import load_yaml_config
from core.registry import get_adapter, known_datasets
from pipeline.case_builder import build_cases_conditioned_batch
from pipeline.patient_info_units import OpenAIChatPatientInfoExtractor

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/default.yaml')
    parser.add_argument('--dataset', default='mimic-iv-ext', choices=known_datasets())
    parser.add_argument('--limit', type=int)
    parser.add_argument('--copies', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--info-extractor-provider', required=True)
    parser.add_argument('--keep-runtime-artifacts', action='store_true')
    parser.add_argument('--keep-table-cache', action='store_true')
    parser.add_argument('--synthea-root', type=Path, default=ROOT.parent / 'synthea')
    args = parser.parse_args()
    if min(args.copies, args.batch_size, args.limit or 1) < 1:
        parser.error('Limits, copies and batch size must be positive')
    if not args.keep_table_cache:
        os.environ['CLINVERSE_DISABLE_TABLE_CACHE'] = '1'
    cfg = load_yaml_config(args.config)
    if not list(args.synthea_root.rglob('synthea-with-dependencies.jar')):
        raise FileNotFoundError('Build Synthea with scripts/setup_synthea.py')
    provider = cfg['llm_providers'][args.info_extractor_provider]
    extractor = OpenAIChatPatientInfoExtractor(
        api_key=provider['api_key'], base_url=provider['base_url'],
        model=provider['info_extractor_model'], timeout=provider.get('timeout', 120),
        extra_body=provider.get('extra_body'))
    adapter = get_adapter(args.dataset, **cfg.get('datasets', {}).get(args.dataset, {}))
    admissions = list(adapter.load_admissions(limit=args.limit))
    if not admissions:
        raise ValueError('No admissions found')
    stats = build_cases_conditioned_batch(
        admissions=admissions, adapter=adapter,
        cases_dir=args.output or Path(cfg['paths']['cases']),
        synthea_root=args.synthea_root.resolve(), runtime_dir=ROOT / 'data/runtime',
        extra_module_dir=ROOT / 'synthea_extensions/modules', top_k=3,
        n_copies=args.copies, verbose=True, info_extractor=extractor,
        cleanup_runtime_artifacts=not args.keep_runtime_artifacts,
        batch_size=args.batch_size, resume=args.resume)
    print(json.dumps(stats, indent=2))
    if stats['skipped_no_match']:
        raise RuntimeError(f"Case generation failed for {stats['skipped_no_match']} admissions")


if __name__ == '__main__':
    main()
