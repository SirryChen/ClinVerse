"""Build Synthea fixedRecordPath Entity JSON from MIMIC hard priors.

Synthea's fixedRecordPath mechanism reads an EntityManager JSON file that
describes one or more entities (patients). Each entity has:
  - individualId: a unique identifier (we use hadm_id)
  - seeds: list of Seed objects with demographic attributes and time periods

The MIMIC prior injection also writes a per-patient attributes JSON file to
the mimicPriorDir, named {individualId}.json, containing the condition flags
and any additional attributes to inject into person.attributes before modules run.

Reference: synthea/src/main/java/org/mitre/synthea/identity/Entity.java
"""

from __future__ import annotations

import json
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# US names corpus (small subset for generation; Synthea will override with
# real demographics in many cases, but we need something valid)
# ---------------------------------------------------------------------------
_MALE_FIRST = ["James", "John", "Robert", "Michael", "William", "David", "Richard",
               "Joseph", "Thomas", "Charles", "Daniel", "Matthew", "Anthony", "Mark"]
_FEMALE_FIRST = ["Mary", "Patricia", "Jennifer", "Linda", "Barbara", "Elizabeth",
                 "Susan", "Jessica", "Sarah", "Karen", "Lisa", "Nancy", "Betty"]
_LAST = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
         "Davis", "Wilson", "Taylor", "Anderson", "Thomas", "Jackson", "White"]


def _random_name(gender: str, rng: random.Random) -> tuple[str, str]:
    first = rng.choice(_MALE_FIRST if gender == "M" else _FEMALE_FIRST)
    last = rng.choice(_LAST)
    return first, last


def _birthdate_to_seed_period_start(date_of_birth: str) -> str:
    """Return the seed period start date (= DOB)."""
    return date_of_birth


def build_entity_json(
    priors: dict,
    n_copies: int = 10,
    state: str = "MA",
    city: str = "Boston",
    zip_code: str = "02115",
) -> dict:
    """Build a Synthea EntityManager JSON for one MIMIC patient with n_copies copies.

    Format must match EntityManager.fromJSON() / EntityDeserializer:
      - Top-level key: "records" (not "entities")
      - Entity fields: individualId, dateOfBirth, gender, seeds[]
      - Seed fields: seedId, period{start, end}, givenName, familyName,
        phone, addressLines[], city, state, zipCode, variants[]
      - period.end = null for the last (open-ended) seed

    Each copy has a distinct individualId so Synthea generates n_copies
    independently seeded patients with the same demographic priors.
    """
    hard = priors["hard_priors"]
    dob = hard["date_of_birth"]
    gender = hard["gender"]
    hadm_id = priors["hadm_id"]

    rng = random.Random(int(hadm_id) if hadm_id.isdigit() else hash(hadm_id))

    records = []
    for i in range(n_copies):
        individual_id = f"{hadm_id}_copy_{i}"
        first, last = _random_name(gender, rng)
        seed_id = f"seed_{individual_id}"

        entity = {
            "individualId": individual_id,
            "dateOfBirth": dob,       # required at entity level by EntityDeserializer
            "gender": gender,          # required at entity level by EntityDeserializer
            "seeds": [
                {
                    "seedId": seed_id,
                    "period": {
                        "start": dob,
                        "end": None   # open-ended: last seed must have null end
                    },
                    "givenName": first,
                    "familyName": last,
                    "phone": f"617555{rng.randint(1000,9999):04d}",
                    "addressLines": ["100 Main St"],
                    "city": city,
                    "state": state,   # 2-letter state code; Synthea converts to full name
                    "zipCode": zip_code,
                    "variants": []    # required field; empty list is valid
                }
            ]
        }
        records.append(entity)

    # Top-level key MUST be "records" — EntityManager.fromJSON uses Gson field mapping
    return {"records": records}


def build_entity_json_batch(priors_batch: list[dict], n_copies: int = 10) -> dict:
    """Build one EntityManager document for multiple admissions."""
    records = []
    for priors in priors_batch:
        records.extend(build_entity_json(priors, n_copies=n_copies)["records"])
    return {"records": records}


def write_fixed_record_file(entity_json: dict, output_path: Path) -> None:
    """Write the entity JSON to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(entity_json, f, indent=2)


def write_mimic_prior_files(priors: dict, prior_dir: Path, n_copies: int = 10) -> None:
    """Write per-copy MIMIC prior JSON files to prior_dir/{individual_id}.json.

    Each file contains the condition_flags dict that Generator.java will
    inject into person.attributes before any modules run.

    Args:
        priors: output of mimic_prior_extractor.extract_all_priors()
        prior_dir: directory where {individual_id}.json files are written
        n_copies: must match n_copies used in build_entity_json()
    """
    prior_dir.mkdir(parents=True, exist_ok=True)
    hadm_id = priors["hadm_id"]
    flags = priors["condition_flags"]

    for i in range(n_copies):
        individual_id = f"{hadm_id}_copy_{i}"
        path = prior_dir / f"{individual_id}.json"
        with open(path, "w") as f:
            json.dump(flags, f, indent=2)


def write_mimic_prior_files_batch(
    priors_batch: list[dict],
    prior_dir: Path,
    n_copies: int = 10,
) -> None:
    """Write conditioning files for every admission in a generation batch."""
    for priors in priors_batch:
        write_mimic_prior_files(priors, prior_dir, n_copies=n_copies)


def build_and_write_all(
    priors: dict,
    fixed_record_dir: Path,
    prior_dir: Path,
    n_copies: int = 10,
) -> Path:
    """Build entity JSON + prior files for one MIMIC admission.

    Returns the path to the fixedRecordPath JSON file.
    """
    hadm_id = priors["hadm_id"]
    entity_json = build_entity_json(priors, n_copies=n_copies)

    fixed_record_path = fixed_record_dir / f"{hadm_id}_fixed_records.json"
    write_fixed_record_file(entity_json, fixed_record_path)
    write_mimic_prior_files(priors, prior_dir, n_copies=n_copies)

    return fixed_record_path
