"""Run Synthea conditioned on MIMIC patient priors and collect output bundles.

This module orchestrates the two-tier conditioned generation pipeline:

  Tier 1 (full_conditioned): Run Synthea with fixedRecordPath + mimicPriorDir
    → guarantees demographics, forces all flagged conditions via mimic_force_X attributes
  Tier 2 (demographic_conditioned): fixedRecordPath only (empty prior dir)
    → used if Tier 1 produces zero valid candidates

If both tiers produce zero candidates, returns mode="skipped" with an error message.
The caller (case_builder) counts these as skipped_no_match and does not write a case.

After generation, synthea_parser.parse_bundle() parses each candidate FHIR bundle
and persona_selector.select_persona() scores and ranks them.
"""

from __future__ import annotations

import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

from core.ports import DatasetAdapter

from .synthea_parser import parse_bundle
from .synthea_priors import extract_all_priors
from .synthea_fixed_record_builder import (
    build_and_write_all,
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
    include_priors: bool,
) -> tuple[Path, Path, dict[str, str]]:
    fixed_record_path = batch_dir / "fixed_records.json"
    prior_dir = batch_dir / ("priors" if include_priors else "empty_priors")
    prior_dir.mkdir(parents=True, exist_ok=True)
    write_fixed_record_file(
        build_entity_json_batch(priors_batch, n_copies=n_copies),
        fixed_record_path,
    )
    if include_priors:
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
    """Generate conditioned personas for several admissions with two JVM calls at most."""
    if not admissions:
        return {}

    priors_batch = [extract_all_priors(admission, adapter) for admission in admissions]
    priors_by_hadm = {str(priors["hadm_id"]): priors for priors in priors_batch}
    batch_dir = runtime_dir / "conditioned_batches" / batch_id
    shutil.rmtree(batch_dir, ignore_errors=True)
    synthea_jar = _find_synthea_jar(synthea_root)
    batch_timeout = max(timeout, 60 * len(admissions))
    results: dict[str, dict] = {}

    try:
        tier1_error = None
        tier1_grouped: dict[str, list[dict]] = {}
        try:
            fixed_path, prior_dir, id_map = _write_batch_inputs(
                priors_batch, batch_dir / "tier1", n_copies, include_priors=True
            )
            paths = _run_synthea(
                synthea_jar,
                fixed_path,
                prior_dir,
                batch_dir / "tier1" / "output",
                state,
                city,
                extra_module_dir,
                batch_timeout,
            )
            tier1_grouped = _group_personas_by_admission(_parse_bundles(paths), id_map)
        except Exception as exc:
            tier1_error = str(exc)

        missing_priors = []
        for priors in priors_batch:
            hadm_id = str(priors["hadm_id"])
            personas = tier1_grouped.get(hadm_id, [])
            if personas:
                results[hadm_id] = {
                    "personas": personas,
                    "mode": "full_conditioned",
                    "priors": priors,
                    "error": None,
                }
            else:
                missing_priors.append(priors)

        if missing_priors:
            tier2_error = None
            tier2_grouped: dict[str, list[dict]] = {}
            try:
                fixed_path, prior_dir, id_map = _write_batch_inputs(
                    missing_priors, batch_dir / "tier2", n_copies, include_priors=False
                )
                paths = _run_synthea(
                    synthea_jar,
                    fixed_path,
                    prior_dir,
                    batch_dir / "tier2" / "output",
                    state,
                    city,
                    extra_module_dir,
                    max(timeout, 60 * len(missing_priors)),
                )
                tier2_grouped = _group_personas_by_admission(_parse_bundles(paths), id_map)
            except Exception as exc:
                tier2_error = str(exc)

            for priors in missing_priors:
                hadm_id = str(priors["hadm_id"])
                personas = tier2_grouped.get(hadm_id, [])
                if personas:
                    results[hadm_id] = {
                        "personas": personas,
                        "mode": "demographic_conditioned",
                        "priors": priors,
                        "error": f"Tier 1 failed: {tier1_error or 'No valid personas generated'}",
                    }
                else:
                    results[hadm_id] = {
                        "personas": [],
                        "mode": "skipped",
                        "priors": priors_by_hadm[hadm_id],
                        "error": (
                            f"Tier1: {tier1_error or 'No valid personas generated'} | "
                            f"Tier2: {tier2_error or 'No valid personas generated'}"
                        ),
                    }
        return results
    finally:
        if cleanup_runtime_artifacts:
            shutil.rmtree(batch_dir, ignore_errors=True)
            try:
                batch_dir.parent.rmdir()
            except OSError:
                pass


def _parse_bundles(bundle_paths: list[Path]) -> list[dict]:
    """Parse a list of FHIR bundle paths into persona dicts."""
    personas = []
    for path in bundle_paths:
        try:
            p = parse_bundle(path)
            if p:
                personas.append(p)
        except Exception:
            pass
    return personas


def _cleanup_admission_runtime(
    runtime_dir: Path,
    hadm_id: str,
    n_copies: int,
    output_dirs: list[Path],
    fixed_record_paths: list[Path],
) -> None:
    """Remove per-admission Synthea artefacts after candidate parsing."""
    for output_dir in output_dirs:
        shutil.rmtree(output_dir, ignore_errors=True)

    for fixed_record_path in fixed_record_paths:
        try:
            fixed_record_path.unlink()
        except FileNotFoundError:
            pass

    prior_dir = runtime_dir / "mimic_priors"
    for i in range(n_copies):
        try:
            (prior_dir / f"{hadm_id}_copy_{i}.json").unlink()
        except FileNotFoundError:
            pass

    for directory in (
        runtime_dir / "fixed_records",
        prior_dir,
        runtime_dir / "empty_priors",
        runtime_dir / "conditioned_output",
    ):
        try:
            directory.rmdir()
        except OSError:
            pass


def run_conditioned_generation(
    admission: dict,
    adapter: DatasetAdapter,
    synthea_root: Path,
    runtime_dir: Path,
    extra_module_dir: Path | None = None,
    n_copies: int = 10,
    state: str = "Massachusetts",
    city: str = "Boston",
    timeout: int = 300,
    cleanup_runtime_artifacts: bool = True,
) -> dict:
    """Run the full conditioned generation pipeline for one admission.

    Args:
        admission: admission dict from a DatasetAdapter (must include icd_diagnoses)
        adapter: the DatasetAdapter that produced `admission` — supplies
            extract_condition_priors() so the comorbidity-flag source is
            whatever this dataset actually has (coded ICD for MIMIC, boolean
            precondition fields for SICdb, ...). See pipeline/synthea_priors.py.
        synthea_root: root of the Synthea repository (contains build/)
        runtime_dir: base directory for runtime artefacts
        extra_module_dir: optional path to custom Synthea modules
        n_copies: number of candidate patients to generate
        state: US state
        city: city
        timeout: Synthea subprocess timeout

    Returns:
        dict with keys:
          - personas: list of parsed candidate persona dicts
          - mode: "full_conditioned" | "demographic_conditioned" | "skipped"
          - priors: the extracted priors dict
          - error: error message if mode=="skipped", else None
    """
    hadm_id = admission.get("hadm_id", "unknown")
    priors = extract_all_priors(admission, adapter)

    fixed_record_dir = runtime_dir / "fixed_records"
    prior_dir = runtime_dir / "mimic_priors"
    output_dir = runtime_dir / "conditioned_output" / hadm_id
    output_dir_t2 = output_dir.parent / (hadm_id + "_t2")
    fixed_record_path = fixed_record_dir / f"{hadm_id}_fixed_records.json"
    runtime_outputs = [output_dir, output_dir_t2]
    fixed_record_paths = [fixed_record_path]

    synthea_jar = _find_synthea_jar(synthea_root)

    try:
        # --- Tier 1: Full conditioned (demographics + condition flags) ---
        try:
            fixed_record_path = build_and_write_all(
                priors, fixed_record_dir, prior_dir, n_copies=n_copies
            )
            bundle_paths = _run_synthea(
                synthea_jar, fixed_record_path, prior_dir, output_dir,
                state, city, extra_module_dir, timeout
            )
            personas = _parse_bundles(bundle_paths)
            if personas:
                return {
                    "personas": personas,
                    "mode": "full_conditioned",
                    "priors": priors,
                    "error": None,
                }
        except Exception as e:
            tier1_error = str(e)
        else:
            tier1_error = "No valid personas generated"

        # --- Tier 2: Demographic conditioned only (empty prior dir) ---
        empty_prior_dir = runtime_dir / "empty_priors"
        empty_prior_dir.mkdir(parents=True, exist_ok=True)

        try:
            from .synthea_fixed_record_builder import build_entity_json, write_fixed_record_file
            entity_json = build_entity_json(priors, n_copies=n_copies)
            write_fixed_record_file(entity_json, fixed_record_path)

            bundle_paths = _run_synthea(
                synthea_jar, fixed_record_path, empty_prior_dir, output_dir_t2,
                state, city, extra_module_dir, timeout
            )
            personas = _parse_bundles(bundle_paths)
            if personas:
                return {
                    "personas": personas,
                    "mode": "demographic_conditioned",
                    "priors": priors,
                    "error": f"Tier 1 failed: {tier1_error}",
                }
        except Exception as e:
            tier2_error = str(e)
        else:
            tier2_error = "No valid personas generated"

        return {
            "personas": [],
            "mode": "skipped",
            "priors": priors,
            "error": f"Tier1: {tier1_error} | Tier2: {tier2_error}",
        }
    finally:
        if cleanup_runtime_artifacts:
            _cleanup_admission_runtime(
                runtime_dir=runtime_dir,
                hadm_id=str(hadm_id),
                n_copies=n_copies,
                output_dirs=runtime_outputs,
                fixed_record_paths=fixed_record_paths,
            )
