"""Assemble and persist cases from EHR admissions and conditioned Synthea generation."""

from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from .trimmer import trim_persona
from .evidence_builder import build_evidence_inventory
from .view_builder import build_views
from .auditor import audit
from .personality_sampler import sample_personality
from .patient_info_units import build_patient_information_units
from core.icd import block


def build_case(
    admission: dict,
    top_k_candidates: list[dict],
    adapter=None,           # core.ports.DatasetAdapter — used to render readable lab evidence labels
    info_extractor=None,
) -> dict:
    """Assemble a full enhanced case using the rank-1 Synthea candidate.

    """
    if not top_k_candidates:
        raise ValueError("Supply at least one patient candidate")
    if not block(admission["primary_diagnosis"].get("icd10_code")):
        raise ValueError("Cases require a mapped ICD-10 primary diagnosis")

    best = top_k_candidates[0]
    persona = best["persona"]
    generation_mode = best.get("mode", "unknown")

    index_date = admission["demographics"].get("admittime", "")

    if "trimmed_persona" in best:
        trimmed_persona = best["trimmed_persona"]
        trimmed_events = best.get("trimmed_events", [])
    else:
        trim_result = trim_persona(persona, index_date, admission["primary_diagnosis"])
        trimmed_persona = trim_result["trimmed_persona"]
        trimmed_events = trim_result["trimmed_events"]

    leakage_count = sum(1 for e in trimmed_events if e.get("leakage_risk"))
    leakage_risk_score = round(min(1.0, leakage_count * 0.15), 4)

    personality = sample_personality()
    patient_information_units = build_patient_information_units(
        admission, trimmed_persona, extractor=info_extractor
    )
    evidence_inventory = build_evidence_inventory(admission, trimmed_persona, adapter=adapter)
    views = build_views(
        admission,
        trimmed_persona,
        personality=personality,
        patient_information_units=patient_information_units,
    )

    # audit() expects a match_score; conditioned mode uses composite score
    match_score = best.get("match_score", best.get("composite", 0.0))
    audit_result = audit(
        admission, persona, trimmed_persona,
        match_score=match_score,
    )
    audit_result["leakage_risk_score"] = leakage_risk_score

    # strip large persona blobs from top_k summary
    top_k_summary = [
        {k: v for k, v in c.items() if k not in ("persona", "trimmed_persona", "trimmed_events")}
        for c in top_k_candidates
    ]

    # Condition priors used (if conditioned generation was run)
    condition_flags_used = best.get("priors", {}).get("condition_flags", {}) if "priors" in best else {}

    return {
        "case_id": admission["case_id"],
        "personality": personality,
        "ground_truth": {
            "primary_diagnosis": admission["primary_diagnosis"]["long_title"],
            "icd_code": admission["primary_diagnosis"]["icd_code"],
            "icd_version": admission["primary_diagnosis"]["icd_version"],
            # Always ICD-10 regardless of source dataset (core.schema normalizes
            # MIMIC's ICD-9 entries too) — this is the field evaluation/metrics.py
            # actually scores against, via simulation_views.moderator_view.ground_truth.
            "icd10_code": admission["primary_diagnosis"].get("icd10_code"),
            "hidden_from_patient_simulator": True,
        },
        "core_admission": {
            # ("mimic-iv-ext" | "sicdb-ext" | ...), which is exactly what
            # simulation/session.py and scripts/run_batch_eval.py use to pick
            # the right DatasetAdapter (lab catalog, radiology availability,
            # condition-prior extraction) for *this* case at simulation time.
            "source": admission.get("source", "mimic-iv-ext"),
            "hadm_id": admission["hadm_id"],
            "subject_id": admission["subject_id"],
            "demographics": admission["demographics"],
            "hpi": admission["hpi"],
            "physical_exam": admission["physical_exam"],
            "vital_signs": admission.get("vital_signs", ""),
            "labs": admission["labs"],
            "microbiology": admission["microbiology"],
            "radiology": admission["radiology"],
            "medications": admission["medications"],
            "procedures": admission["procedures"],
        },
        "patient_background": {
            "source": "synthea",
            "generation_mode": generation_mode,
            "synthea_patient_id": trimmed_persona.get("patient_id", ""),
            "past_medical_history": [c["display"] for c in trimmed_persona.get("conditions", [])],
            "long_term_medications": [m["display"] for m in trimmed_persona.get("medications", [])],
            "allergies": [a["substance"] for a in trimmed_persona.get("allergies", [])],
            "immunizations": [i["vaccine"] for i in trimmed_persona.get("immunizations", [])],
            "social_history": {
                obs["display"]: f"{obs.get('value','')} {obs.get('unit','')}".strip()
                for obs in trimmed_persona.get("observations", [])
                if any(kw in obs.get("display", "").lower()
                       for kw in ["smoking", "tobacco", "alcohol", "bmi", "body mass", "employment"])
            },
            "prior_encounters_count": len(trimmed_persona.get("encounters", [])),
        },
        "patient_information_units": patient_information_units,
        "evidence_inventory": evidence_inventory,
        "simulation_views": views,
        "consistency_metadata": {
            **audit_result,
            "generation_mode": generation_mode,
            "condition_flags_forced": condition_flags_used,
            "top_k_candidates": top_k_summary,
            "trimmed_synthea_events": trimmed_events,
            "provenance": [
                {"field": "core_admission", "source": "mimic-iv", "hadm_id": admission["hadm_id"]},
                {"field": "patient_background", "source": "synthea",
                 "generation_mode": generation_mode,
                 "patient_id": trimmed_persona.get("patient_id", "")},
            ],
        },
    }


def save_case(case: dict, cases_dir: str | Path) -> Path:
    cases_dir = Path(cases_dir)
    cases_dir.mkdir(parents=True, exist_ok=True)
    path = cases_dir / f"{case['case_id']}.json"
    with open(path, "w") as f:
        json.dump(case, f, ensure_ascii=False, separators=(",", ":"))
    return path


def _case_index_entry(case: dict) -> dict:
    return {
        "case_id": case["case_id"],
        "icd_code": case["ground_truth"]["icd_code"],
        "icd_version": case["ground_truth"]["icd_version"],
        "primary_diagnosis": case["ground_truth"]["primary_diagnosis"],
        "match_score": case["consistency_metadata"]["match_score"],
        "clinical_consistency_score": case["consistency_metadata"]["clinical_consistency_score"],
        "leakage_risk_score": case["consistency_metadata"]["leakage_risk_score"],
        "usable": case["consistency_metadata"]["usable"],
    }


def load_index(cases_dir: str | Path) -> dict[str, dict]:
    cases_dir = Path(cases_dir)
    index_path = cases_dir / "index.jsonl"
    entries: dict[str, dict] = {}
    if index_path.exists():
        with open(index_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                existing = json.loads(line)
                if not existing.get("case_id"):
                    raise ValueError("Case index entries require case_id")
                entries[existing["case_id"]] = existing
    return entries


def write_index(entries: dict[str, dict], cases_dir: str | Path) -> None:
    cases_dir = Path(cases_dir)
    index_path = cases_dir / "index.jsonl"
    tmp_path = index_path.with_suffix(".jsonl.tmp")
    with open(tmp_path, "w") as f:
        for existing in entries.values():
            f.write(json.dumps(existing, ensure_ascii=False) + "\n")
    tmp_path.replace(index_path)


def update_index(case: dict, cases_dir: str | Path) -> None:
    """Upsert one case. Batch builders use load_index/write_index directly."""
    entries = load_index(cases_dir)
    entry = _case_index_entry(case)
    entries[entry["case_id"]] = entry
    write_index(entries, cases_dir)


def build_cases_conditioned_batch(
    admissions,             # Iterable[dict]
    adapter,                # core.ports.DatasetAdapter — supplies extract_condition_priors()
    cases_dir: str | Path,
    synthea_root: Path,
    runtime_dir: Path,
    extra_module_dir: Path | None = None,
    top_k: int = 3,
    n_copies: int = 10,
    verbose: bool = True,
    info_extractor=None,
    cleanup_runtime_artifacts: bool = True,
    batch_size: int = 20,
    resume: bool = False,
) -> dict:
    """Build cases using conditioned Synthea generation with this dataset's priors.

    For each admission:
      1. Extracts condition priors via adapter.extract_condition_priors() —
         coded secondary ICD diagnoses for MIMIC, boolean precondition fields
         for SICdb (see adapters/*/condition_priors.py)
      2. Builds fixedRecordPath Entity JSON + per-patient prior JSON files
      3. Runs conditioned Synthea generation
      4. Scores and ranks candidates; picks top-k
      5. Assembles and saves the case

    Cases without usable personas are counted in
    stats["skipped_no_match"] and not written to disk.
    """
    from .conditioned_synthea_runner import run_conditioned_generation_batch
    from .persona_selector import select_persona

    cases_dir = Path(cases_dir)
    cases_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "processed": 0,
        "full_conditioned": 0,
        "skipped_no_match": 0,
        "written": 0,
        "skipped_existing": 0,
    }

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    index_entries = load_index(cases_dir)
    pending = []
    for admission in admissions:
        case_id = admission.get("case_id", "")
        case_path = cases_dir / f"{case_id}.json"
        if resume and case_id in index_entries and case_path.exists():
            stats["skipped_existing"] += 1
            continue
        pending.append(admission)

    total_batches = (len(pending) + batch_size - 1) // batch_size
    progress = tqdm(
        total=len(pending) + stats["skipped_existing"],
        initial=stats["skipped_existing"],
        desc="conditioned-cases",
        unit="case",
        disable=not verbose,
    )
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset:offset + batch_size]
        batch_number = offset // batch_size + 1
        first_hadm = str(batch[0].get("hadm_id", "unknown"))
        progress.set_postfix(
            batch=f"{batch_number}/{total_batches}",
            written=stats["written"],
            failed=stats["skipped_no_match"],
        )

        try:
            generation_results = run_conditioned_generation_batch(
                batch,
                adapter,
                synthea_root=synthea_root,
                runtime_dir=runtime_dir,
                extra_module_dir=extra_module_dir,
                n_copies=n_copies,
                cleanup_runtime_artifacts=cleanup_runtime_artifacts,
                batch_id=f"{batch_number:05d}_{first_hadm}",
            )
        except Exception as exc:
            if verbose:
                progress.write(f"[error] batch {batch_number}: {exc}")
            stats["processed"] += len(batch)
            stats["skipped_no_match"] += len(batch)
            progress.update(len(batch))
            continue

        index_dirty = False
        for admission in batch:
            stats["processed"] += 1
            hadm_id = str(admission.get("hadm_id", "?"))
            generation_result = generation_results.get(hadm_id)
            if generation_result is None:
                stats["skipped_no_match"] += 1
                if verbose:
                    progress.write(f"[skip] {hadm_id} — missing batch generation result")
                continue

            try:
                candidates = select_persona(admission, generation_result, top_k=top_k)
            except Exception as exc:
                stats["skipped_no_match"] += 1
                if verbose:
                    progress.write(f"[error] {hadm_id}: {exc}")
                continue

            if not candidates:
                stats["skipped_no_match"] += 1
                if verbose:
                    progress.write(
                        f"[skip] {hadm_id} — "
                        f"{generation_result.get('error') or 'no usable candidates'}"
                    )
                continue

            mode = candidates[0].get("mode", "unknown")
            stats[mode] = stats.get(mode, 0) + 1
            case = build_case(admission, candidates, adapter=adapter, info_extractor=info_extractor)
            save_case(case, cases_dir)
            entry = _case_index_entry(case)
            index_entries[entry["case_id"]] = entry
            index_dirty = True
            stats["written"] += 1

        if index_dirty:
            write_index(index_entries, cases_dir)
        progress.update(len(batch))
        progress.set_postfix(
            batch=f"{batch_number}/{total_batches}",
            written=stats["written"],
            failed=stats["skipped_no_match"],
        )

    progress.close()

    return stats
