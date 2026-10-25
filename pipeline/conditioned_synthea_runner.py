"""Conditioned Synthea generation from admission priors."""

from __future__ import annotations

import shutil
import json
import subprocess
from functools import lru_cache
from pathlib import Path

from core.ports import DatasetAdapter

from .synthea_parser import parse_bundle
from .synthea_priors import extract_all_priors
from .synthea_fixed_record_builder import (
    build_entity_json_batch,
    write_fixed_record_file,
    write_mimic_prior_files_batch,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SYNTHEA_JAR_PATTERN = "synthea-with-dependencies.jar"


@lru_cache(maxsize=4)
def _find_synthea_jar(synthea_root: Path) -> Path:
    """Find the built Synthea fat JAR."""
    candidates = list(synthea_root.rglob(SYNTHEA_JAR_PATTERN))
    if not candidates:
        raise FileNotFoundError(
            f"Synthea JAR not found under {synthea_root}. "
            "Run: cd synthea && ./gradlew uberJar"
        )
    return candidates[0]


def _run_synthea(
    synthea_jar: Path,
    fixed_record_path: Path,
    prior_dir: Path,
    output_dir: Path,
    state: str = "Massachusetts",
    city: str = "Boston",
    extra_module_dir: Path | None = None,
    timeout: int = 300,
    end_date: str = "20201231",
) -> list[Path]:
    """Run Synthea and return list of generated FHIR bundle JSON paths.

    Args:
        synthea_jar: path to the synthea-with-dependencies.jar
        fixed_record_path: path to the Entity JSON file (fixedRecordPath)
        prior_dir: path to the per-patient prior JSON directory (mimicPriorDir)
        output_dir: where Synthea writes FHIR bundles
        state: US state name
        city: city name
        extra_module_dir: path to custom disease modules directory
        timeout: subprocess timeout in seconds

    Returns:
        list of .json file paths in output_dir/fhir/
    """
    records = json.loads(fixed_record_path.read_text())['records']
    if not records:
        raise ValueError('Conditioned generation requires fixed patient records')
    for record in records:
        prior_file = prior_dir / f"{record['individualId']}.json"
        if not prior_file.is_file():
            raise RuntimeError(f"Missing conditioning priors for {record['individualId']}")
        prior = json.loads(prior_file.read_text())
        if not isinstance(prior, dict) or any(isinstance(value, (dict, list)) for value in prior.values()):
            raise ValueError('Conditioning priors must be flat JSON objects')
    output_dir.mkdir(parents=True, exist_ok=True)
    fhir_dir = output_dir / "fhir"

    cmd = [
        "java", "-jar", str(synthea_jar),
        f"--exporter.baseDirectory={output_dir}",
        "--exporter.fhir.export=true",
        "--exporter.csv.export=false",
        "--exporter.hospital.fhir.export=false",
        "--exporter.practitioner.fhir.export=false",
        "--exporter.metadata.export=false",
        "--exporter.pretty_print=false",
        "--exporter.fhir.included_resources="
        "Patient,Condition,MedicationRequest,AllergyIntolerance,Immunization,Observation,Encounter",
        "--generate.log_patients.detail=none",
        "-e", end_date,
        "-f", str(fixed_record_path),
        "-mp", str(prior_dir),
        state, city,
    ]

    if extra_module_dir and extra_module_dir.exists():
        cmd.extend(["-d", str(extra_module_dir)])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Synthea failed (exit {result.returncode}):\n{result.stderr[:2000]}"
            )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Synthea timed out after {timeout}s")

    if not fhir_dir.exists():
        return []
    return list(fhir_dir.glob("*.json"))


def _parse_bundles(paths: list[Path]) -> list[dict]:
    personas = []
    for path in paths:
        persona = parse_bundle(path)
        if persona is not None:
            personas.append(persona)
    return personas


def _group_personas_by_admission(
    personas: list[dict],
    individual_to_hadm: dict[str, str],
) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for persona in personas:
        hadm_id = individual_to_hadm.get(persona.get("record_id", ""))
        if hadm_id is not None:
            grouped.setdefault(hadm_id, []).append(persona)
    return grouped


def _write_batch_inputs(
    priors_batch: list[dict],
    batch_dir: Path,
    n_copies: int,
) -> tuple[Path, Path, dict[str, str]]:
    fixed_record_path = batch_dir / "fixed_records.json"
    prior_dir = batch_dir / "priors"
    prior_dir.mkdir(parents=True, exist_ok=True)
    write_fixed_record_file(
        build_entity_json_batch(priors_batch, n_copies=n_copies),
        fixed_record_path,
    )
    write_mimic_prior_files_batch(priors_batch, prior_dir, n_copies=n_copies)

    individual_to_hadm = {
        f"{priors['hadm_id']}_copy_{i}": str(priors["hadm_id"])
        for priors in priors_batch
        for i in range(n_copies)
    }
    return fixed_record_path, prior_dir, individual_to_hadm


def run_conditioned_generation_batch(
    admissions: list[dict],
    adapter: DatasetAdapter,
    synthea_root: Path,
    runtime_dir: Path,
    extra_module_dir: Path | None = None,
    n_copies: int = 10,
    state: str = "Massachusetts",
    city: str = "Boston",
    timeout: int = 300,
    cleanup_runtime_artifacts: bool = True,
    batch_id: str = "batch",
) -> dict[str, dict]:
    if not admissions:
        return {}
    priors = [extract_all_priors(admission, adapter) for admission in admissions]
    batch_dir = runtime_dir / "conditioned_batches" / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    try:
        fixed, prior_dir, id_map = _write_batch_inputs(priors, batch_dir, n_copies)
        paths = _run_synthea(
            _find_synthea_jar(synthea_root), fixed, prior_dir,
            batch_dir / "output", state, city, extra_module_dir,
            max(timeout, 60 * len(admissions)),
        )
        grouped = _group_personas_by_admission(_parse_bundles(paths), id_map)
        missing = [str(prior["hadm_id"]) for prior in priors if not grouped.get(str(prior["hadm_id"]))]
        if missing:
            raise RuntimeError(f"Conditioned generation produced no candidates for {len(missing)} admissions")
        return {str(prior["hadm_id"]): {
            "personas": grouped[str(prior["hadm_id"])], "mode": "full_conditioned",
            "priors": prior, "error": None,
        } for prior in priors}
    finally:
        if cleanup_runtime_artifacts:
            shutil.rmtree(batch_dir, ignore_errors=True)


def run_conditioned_generation(
    admission: dict, adapter: DatasetAdapter, synthea_root: Path,
    runtime_dir: Path, extra_module_dir: Path | None = None,
    n_copies: int = 10, state: str = "Massachusetts", city: str = "Boston",
    timeout: int = 300, cleanup_runtime_artifacts: bool = True,
) -> dict:
    results = run_conditioned_generation_batch(
        [admission], adapter, synthea_root, runtime_dir, extra_module_dir,
        n_copies, state, city, timeout, cleanup_runtime_artifacts,
        batch_id=str(admission["hadm_id"]),
    )
    return results[str(admission["hadm_id"])]
