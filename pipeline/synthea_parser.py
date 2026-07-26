"""Parse Synthea FHIR R4 bundles into structured persona dicts."""

from __future__ import annotations

import json
import multiprocessing
import pickle
from datetime import datetime, date
from pathlib import Path


def _coding_display(resource: dict, field: str = "code") -> str:
    codings = resource.get(field, {}).get("coding", [])
    return codings[0].get("display", "") if codings else ""


def _parse_patient(r: dict) -> dict:
    bd = r.get("birthDate", "")
    deceased_dt = r.get("deceasedDateTime", "")
    deceased = bool(deceased_dt or r.get("deceasedBoolean", False))

    age = None
    if bd:
        try:
            birth = datetime.strptime(bd, "%Y-%m-%d").date()
            ref = datetime.strptime(deceased_dt[:10], "%Y-%m-%d").date() if deceased_dt else date.today()
            age = (ref - birth).days // 365
        except ValueError:
            pass

    name_parts = r.get("name", [{}])[0]
    given = " ".join(name_parts.get("given", []))
    family = name_parts.get("family", "")
    name = f"{given} {family}".strip()

    extensions = {e.get("url", ""): e for e in r.get("extension", [])}
    race_ext = extensions.get("http://hl7.org/fhir/us/core/StructureDefinition/us-core-race", {})
    race_codings = race_ext.get("extension", [{}])[0].get("valueCoding", {})
    race = race_codings.get("display", "")

    identifiers = {
        item.get("system", ""): item.get("value", "")
        for item in r.get("identifier", [])
        if item.get("system")
    }

    return {
        "patient_id": r.get("id", ""),
        "record_id": identifiers.get("http://mitre.org/record_id", ""),
        "name": name,
        "gender": r.get("gender", ""),
        "birth_date": bd,
        "age": age,
        "deceased": deceased,
        "deceased_date": deceased_dt[:10] if deceased_dt else None,
        "race": race,
        "address_state": r.get("address", [{}])[0].get("state", ""),
        "marital_status": r.get("maritalStatus", {}).get("coding", [{}])[0].get("code", ""),
    }


def _parse_conditions(resources: list[dict]) -> list[dict]:
    out = []
    for r in resources:
        display = _coding_display(r)
        if not display:
            continue
        onset = r.get("onsetDateTime", r.get("onsetPeriod", {}).get("start", ""))
        abatement = r.get("abatementDateTime", "")
        status = r.get("clinicalStatus", {}).get("coding", [{}])[0].get("code", "active")
        code = r.get("code", {}).get("coding", [{}])[0]
        out.append({
            "display": display,
            "system": code.get("system", ""),
            "code": code.get("code", ""),
            "onset_date": onset[:10] if onset else "",
            "abatement_date": abatement[:10] if abatement else "",
            "status": status,
        })
    return out


def _parse_medications(resources: list[dict]) -> list[dict]:
    out = []
    for r in resources:
        med = r.get("medicationCodeableConcept", {})
        codings = med.get("coding", [{}])
        display = codings[0].get("display", "") if codings else ""
        if not display:
            continue
        authored = r.get("authoredOn", "")
        status = r.get("status", "")
        out.append({
            "display": display,
            "code": codings[0].get("code", "") if codings else "",
            "authored_date": authored[:10] if authored else "",
            "status": status,
        })
    return out


def _parse_allergies(resources: list[dict]) -> list[dict]:
    out = []
    for r in resources:
        display = _coding_display(r)
        if not display:
            continue
        reactions = [
            _coding_display(react.get("manifestation", [{}])[0])
            for react in r.get("reaction", [])
        ]
        out.append({
            "substance": display,
            "reactions": [rx for rx in reactions if rx],
            "criticality": r.get("criticality", ""),
        })
    return out


def _parse_immunizations(resources: list[dict]) -> list[dict]:
    out = []
    for r in resources:
        vaccine = r.get("vaccineCode", {}).get("coding", [{}])[0]
        display = vaccine.get("display", "")
        if not display:
            continue
        occurrence = r.get("occurrenceDateTime", "")
        out.append({
            "vaccine": display,
            "code": vaccine.get("code", ""),
            "date": occurrence[:10] if occurrence else "",
        })
    return out


def _parse_observations(resources: list[dict]) -> list[dict]:
    out = []
    for r in resources:
        display = _coding_display(r)
        if not display:
            continue
        value = (
            r.get("valueQuantity", {}).get("value")
            or r.get("valueString")
            or _coding_display(r, "valueCodeableConcept")
        )
        unit = r.get("valueQuantity", {}).get("unit", "")
        effective = r.get("effectiveDateTime", "")
        out.append({
            "display": display,
            "value": value,
            "unit": unit,
            "date": effective[:10] if effective else "",
        })
    return out


def _parse_encounters(resources: list[dict]) -> list[dict]:
    out = []
    for r in resources:
        period = r.get("period", {})
        enc_type = r.get("type", [{}])[0]
        type_display = _coding_display(enc_type) if enc_type else ""
        class_code = r.get("class", {}).get("code", "")
        out.append({
            "type": type_display,
            "class": class_code,
            "start": period.get("start", "")[:10] if period.get("start") else "",
            "end": period.get("end", "")[:10] if period.get("end") else "",
        })
    return out


def parse_bundle(path: str | Path) -> dict | None:
    """Parse one Synthea FHIR Bundle file, return persona dict or None if not a patient bundle."""
    with open(path) as f:
        bundle = json.load(f)

    entries = bundle.get("entry", [])
    by_type: dict[str, list[dict]] = {}
    for entry in entries:
        r = entry.get("resource", {})
        rt = r.get("resourceType", "")
        by_type.setdefault(rt, []).append(r)

    if "Patient" not in by_type:
        return None

    patient = _parse_patient(by_type["Patient"][0])
    patient["conditions"]    = _parse_conditions(by_type.get("Condition", []))
    patient["medications"]   = _parse_medications(by_type.get("MedicationRequest", []))
    patient["allergies"]     = _parse_allergies(by_type.get("AllergyIntolerance", []))
    patient["immunizations"] = _parse_immunizations(by_type.get("Immunization", []))
    patient["observations"]  = _parse_observations(by_type.get("Observation", []))
    patient["encounters"]    = _parse_encounters(by_type.get("Encounter", []))
    return patient


def _parse_bundle_safe(path: Path) -> dict | None:
    try:
        return parse_bundle(path)
    except Exception:
        return None


def load_synthea_pool(pool_dir: str | Path) -> list[dict]:
    """Load all Synthea patient bundles from pool_dir.

    Uses a pickle cache next to pool_dir for fast subsequent loads.
    Parallel-parses on first run using all available CPU cores.
    """
    pool_dir = Path(pool_dir)
    cache_path = pool_dir.parent / "synthea_pool_cache.pkl"

    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    paths = sorted(pool_dir.glob("*.json"))
    workers = min(multiprocessing.cpu_count(), 8)
    with multiprocessing.Pool(workers) as mp:
        results = mp.map(_parse_bundle_safe, paths)

    personas = [p for p in results if p is not None]

    with open(cache_path, "wb") as f:
        pickle.dump(personas, f, protocol=pickle.HIGHEST_PROTOCOL)

    return personas
