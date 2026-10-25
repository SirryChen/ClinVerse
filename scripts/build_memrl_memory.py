#!/usr/bin/env python3
"""Build a semantic episodic memory bank from training trajectories."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.config import load_yaml_config
from doctor_train.embeddings import EmbeddingEncoder
from doctor_train.memrl import build_memory_bank, load_jsonl, save_memory_bank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/default.yaml')
    parser.add_argument('--input-jsonl', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    records = load_jsonl(args.input_jsonl)
    if not records:
        raise ValueError('Supply completed training trajectories')
    config = load_yaml_config(args.config)
    bank = build_memory_bank(records, EmbeddingEncoder(config['llm_providers']['embedding']))
    save_memory_bank(bank, args.output)
    print(f"Saved {len(bank['items'])} memories: {args.output}")


if __name__ == '__main__':
    main()
