import json
from pathlib import Path
import yaml
import pytest
from scripts import prepare_data
from scripts.split_cases import split_cases


def test_patient_disjoint_split(tmp_path):
    cases = tmp_path / "cases"
    cases.mkdir()
    membership = {}
    for patient in range(10):
        for admission in range(2):
            case_id = f"synthetic_{patient}_{admission}"
            membership[case_id] = patient
            (cases / f"{case_id}.json").write_text(json.dumps({"core_admission": {"source": "test", "subject_id": str(patient)}, "simulation_views": {"moderator_view": {"ground_truth": {}}}}))
    output = tmp_path / "splits"
    split_cases(cases, output)
    groups = [{membership[c] for c in (output / f"{split}.txt").read_text().split()} for split in ["train", "validation", "test"]]
    assert all(groups)
    assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]


def test_prepare_stages_and_credentials(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(prepare_data, "ROOT", root)
    raw, note, synthea, normalized, cases = [tmp_path / name for name in ["raw", "note", "synthea", "normalized", "cases"]]
    (raw / "hosp").mkdir(parents=True)
    note.mkdir()
    for name in prepare_data.HOSP_FILES:
        (raw / "hosp" / f"{name}.csv.gz").touch()
    for name in prepare_data.NOTE_FILES:
        (note / f"{name}.csv.gz").touch()
    synthea.mkdir()
    (synthea / "synthea-with-dependencies.jar").touch()
    config = root / "config.yaml"
    config.write_text(yaml.safe_dump({"llm_providers": {"extractor": {"api_key": "${TEST_PREPARE_KEY}", "doctor_model": "test"}}, "datasets": {"mimic-iv-ext": {"data_dir": "data/normalized"}}, "paths": {"cases": "data/cases", "eval_results": "outputs/evaluation"}}))
    monkeypatch.setenv("TEST_PREPARE_KEY", "private-test-value")
    calls = []
    def run(command, **kwargs):
        calls.append(command[1])
        normalized.mkdir(exist_ok=True)
        if command[1].endswith("predata/run.py"):
            for name in prepare_data.NORMALIZED_FILES:
                (normalized / name).write_text("synthetic\n")
        elif command[1].endswith("build_catalog.py"):
            for name in ["lab_provider_catalog.json", "radiology_catalog.json"]:
                (normalized / name).write_text("{}")
        else:
            cases.mkdir()
            (cases / "index.jsonl").write_text('{"case_id":"synthetic"}\n')
    monkeypatch.setattr(prepare_data.subprocess, "run", run)
    prepare_data.main(["--mimic-dir", str(raw), "--note-dir", str(note), "--synthea-root", str(synthea), "--normalized-dir", str(normalized), "--output-dir", str(cases), "--config", str(config)])
    assert len(calls) == 3
    assert "private-test-value" not in (root / "data/preparation/config.yaml").read_text()
    assert "${TEST_PREPARE_KEY}" in (root / "data/preparation/config.yaml").read_text()
    manifest = json.loads((root / "data/preparation/manifest.json").read_text())
    assert manifest["completed"] == ["normalize", "catalogs", "cases"]
