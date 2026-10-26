#!/usr/bin/env python3
from pathlib import Path
import argparse
import yaml


def main():
    parser = argparse.ArgumentParser(description="Configure a local ClinLab checkpoint")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", help="Optional LoRA adapter directory")
    parser.add_argument("--mapping", default="data/normalized/mimic-iv-ext/lab_test_mapping.csv")
    parser.add_argument("--output", default="config/local.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    lab = cfg["llm_providers"]["lab"]
    paths = {"model_path": args.model, "lab_mapping_path": args.mapping}
    if args.adapter:
        paths["adapter_path"] = args.adapter
    else:
        lab.pop("adapter_path", None)
    for key, value in paths.items():
        path = Path(value).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        lab[key] = str(path)
    lab["local_device"] = "cuda:0"
    lab["local_batch_size"] = 4
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(target)


if __name__ == "__main__":
    main()
