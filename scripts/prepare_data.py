#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.config import load_yaml_config
import yaml

ROOT = Path(__file__).resolve().parents[1]
HOSP_FILES = ["admissions", "patients", "prescriptions", "diagnoses_icd", "d_icd_diagnoses",
              "procedures_icd", "d_icd_procedures", "labevents", "d_labitems", "microbiologyevents"]
NOTE_FILES = ["discharge", "radiology", "radiology_detail"]
NORMALIZED_FILES = ["patient_demographics.csv", "history_of_present_illness.csv", "physical_examination.csv",
                    "laboratory_tests.csv", "microbiology.csv", "radiology_reports.csv", "icd_diagnosis.csv",
                    "icd_procedures.csv", "discharge_medications.csv", "hospital_course.csv",
                    "discharge_diagnosis.csv", "discharge_procedures.csv", "lab_test_mapping.csv", "primary_diagnosis_index.json"]


def plan(args):
    commands = [
        [sys.executable, str(ROOT / "preprocessing/mimic_iv_ext/predata/run.py"), "--limit", str(args.limit or 0)],
        [sys.executable, str(ROOT / "preprocessing/mimic_iv_ext/predata/build_catalog.py")],
        [sys.executable, str(ROOT / "scripts/build_conditioned_cases.py"), "--config", str(args.prepared_config),
         "--dataset", "mimic-iv-ext", "--synthea-root", str(args.synthea_root), "--output", str(args.output_dir),
         "--info-extractor-provider", args.extractor_provider, "--copies", str(args.copies), "--batch-size", str(args.batch_size)],
    ]
    if args.limit:
        commands[2] += ["--limit", str(args.limit)]
    if args.resume:
        commands[2] += ["--resume"]
    return commands


def main(argv=None):
    parser = argparse.ArgumentParser(description="Normalize MIMIC-IV, build catalogs, and construct ClinPatient cases")
    parser.add_argument("--mimic-dir", type=Path, required=True)
    parser.add_argument("--note-dir", type=Path, required=True)
    parser.add_argument("--synthea-root", type=Path, default=ROOT.parent / "synthea")
    parser.add_argument("--normalized-dir", type=Path, default=ROOT / "data/normalized/mimic-iv-ext")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/cases")
    parser.add_argument("--config", type=Path, default=ROOT / "config/default.yaml")
    parser.add_argument("--extractor-provider", default="extractor")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--copies", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--setup-synthea", action="store_true")
    parser.add_argument("--synthea-ref", default="29a7d63ce80e9ed2ff83a6060696035538612ccf")
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args(argv)
    if any(v < 1 for v in [args.copies, args.batch_size] + ([args.limit] if args.limit is not None else [])):
        parser.error("Limits, copies and batch size must be positive")
    for name in ["mimic_dir", "note_dir", "synthea_root", "normalized_dir", "output_dir", "config"]:
        setattr(args, name, getattr(args, name).resolve())
    if (args.note_dir / "note").is_dir():
        args.note_dir = args.note_dir / "note"
    args.prepared_config = ROOT / "data/preparation/config.yaml"
    commands = plan(args)
    setup_command = [sys.executable, str(ROOT / "scripts/setup_synthea.py"), "--synthea-root", str(args.synthea_root), "--ref", args.synthea_ref]
    if args.plan:
        print(json.dumps({"stages": ["normalize", "catalogs", "cases"], "commands": ([setup_command] if args.setup_synthea else []) + commands}, indent=2))
        return
    source_files = [args.mimic_dir / "hosp" / f"{name}.csv.gz" for name in HOSP_FILES]
    source_files += [args.note_dir / f"{name}.csv.gz" for name in NOTE_FILES]
    missing = [str(p) for p in source_files if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing source files:\n" + "\n".join(missing))
    if args.setup_synthea:
        subprocess.run(setup_command, cwd=ROOT, check=True)
    if not list(args.synthea_root.rglob("synthea-with-dependencies.jar")):
        raise FileNotFoundError("Build the Synthea JAR before preparing cases")
    cfg = load_yaml_config(args.config)
    if args.extractor_provider not in cfg.get("llm_providers", {}):
        raise ValueError("Select a configured information-unit extractor")
    # Save variable references, not expanded credentials.
    template = yaml.safe_load(args.config.read_text())
    template["datasets"]["mimic-iv-ext"]["data_dir"] = str(args.normalized_dir)
    for name, value in template["datasets"].items():
        if name != "mimic-iv-ext":
            value["data_dir"] = str((ROOT / value["data_dir"]).resolve())
    template["paths"]["cases"] = str(args.output_dir)
    template["paths"]["eval_results"] = str(ROOT / "outputs/evaluation")
    args.prepared_config.parent.mkdir(parents=True, exist_ok=True)
    args.prepared_config.write_text(yaml.safe_dump(template, sort_keys=False))
    args.normalized_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(json.dumps({
        "version": 2, "sources": [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in source_files],
        "normalized_dir": str(args.normalized_dir), "limit": args.limit,
        "copies": args.copies, "synthea_ref": args.synthea_ref,
        "extractor_model": cfg["llm_providers"][args.extractor_provider].get("info_extractor_model"),
    }, sort_keys=True).encode()).hexdigest()
    manifest = args.prepared_config.parent / "manifest.json"
    previous = json.loads(manifest.read_text()) if args.resume and manifest.exists() else {}
    if previous and previous.get("fingerprint") != fingerprint:
        raise ValueError("Resume configuration differs from the saved preparation run")
    completed = previous.get("completed", [])
    expected = [args.normalized_dir / name for name in NORMALIZED_FILES]
    if "normalize" in completed and not all(p.is_file() for p in expected):
        raise FileNotFoundError("The normalized stage is incomplete")
    env = {**os.environ, "CLINVERSE_MIMIC_DIR": str(args.mimic_dir),
           "CLINVERSE_NOTE_DIR": str(args.note_dir), "CLINVERSE_NORMALIZED_DIR": str(args.normalized_dir)}
    for stage, command in zip(["normalize", "catalogs", "cases"], commands):
        if stage in completed and stage != "cases":
            continue
        print(f"Running {stage}", flush=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        artifacts = expected if stage == "normalize" else ([args.normalized_dir / "lab_provider_catalog.json", args.normalized_dir / "radiology_catalog.json"] if stage == "catalogs" else [args.output_dir / "index.jsonl"])
        if not all(p.is_file() and p.stat().st_size > 0 for p in artifacts):
            raise RuntimeError(f"Stage {stage} did not produce its required artifacts")
        if stage not in completed:
            completed.append(stage)
        manifest.write_text(json.dumps({"fingerprint": fingerprint, "completed": completed}, indent=2))
    print(f"Cases: {args.output_dir}\nConfiguration: {args.prepared_config}")


if __name__ == "__main__":
    main()
