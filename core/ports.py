"""Plugin contracts. Each hospital-system data source implements DatasetAdapter
(and the catalog ports it exposes); the rest of the codebase (providers,
pipeline, evaluation, simulation) talks only to these interfaces, never to a
specific dataset's file layout or ID scheme.
"""
from __future__ import annotations

from typing import Iterator, Optional, Protocol, TypedDict, runtime_checkable

from .schema import Admission


class CatalogItem(TypedDict, total=False):
    canonical_name: str
    display_name: str
    aliases: list[str]
    itemids: list           # dataset-local lab item ids this canonical test resolves to
    fluid: Optional[str]
    category: Optional[str]
    panel: list[str]
    source_table: str       # "laboratory_tests" | "microbiology"


class RadiologyStudy(TypedDict, total=False):
    canonical_name: str
    display_name: str
    aliases: list[str]
    modality: str
    regions: list[str]


@runtime_checkable
class LabCatalogPort(Protocol):
    """Resolves doctor-agent test/panel names to dataset-local lab item ids.

    Each dataset has its own item-id space (MIMIC d_labitems.itemid vs SICdb
    LaboratoryID) and its own vocabulary (English vs German test names) — this
    is exactly the boundary providers/lab_provider.py must never reach across.
    """

    def resolve(self, test_name: str) -> Optional[CatalogItem]:
        """Look up a single canonical test name (case/whitespace-insensitive)."""

    def resolve_panel(self, panel_name: str) -> Optional[list[CatalogItem]]:
        """Look up a panel name, returning its component CatalogItems."""

    def resolve_itemid(self, itemid) -> Optional[CatalogItem]:
        """Look up a dataset-local lab item id directly (bypasses name/alias
        matching) — used to render raw lab records (keyed by itemid) as
        human-readable evidence."""

    def describe_for_prompt(self) -> str:
        """Render the full enumerated catalog (panels + items) as text for
        the doctor-agent tool description, so it can't order outside scope."""


@runtime_checkable
class RadiologyCatalogPort(Protocol):
    def resolve(self, study_name: str) -> Optional[RadiologyStudy]:
        ...

    def describe_for_prompt(self) -> str:
        ...


@runtime_checkable
class DatasetAdapter(Protocol):
    name: str                                  # "mimic-iv-ext" | "sicdb-ext" | ...
    language: str                              # language of the source's free text: "en" | "de"
    has_radiology: bool
    has_microbiology: bool
    lab_catalog: LabCatalogPort
    radiology_catalog: Optional[RadiologyCatalogPort]

    def load_admissions(self, limit: Optional[int] = None) -> Iterator[Admission]:
        ...

    def load_admission(self, case_id: str) -> Admission:
        ...

    def extract_condition_priors(self, admission: Admission) -> dict[str, bool]:
        """Return {mimic_force_X: True, ...} comorbidity flags to inject into
        Synthea generation (see synthea_extensions/modules/*.json for the
        attribute names these trigger). Each dataset derives this from
        whatever comorbidity signal it actually has — coded secondary ICD
        diagnoses for MIMIC, boolean precondition fields for SICdb, etc.
        Returning {} is valid (means: no comorbidity priors available; the
        conditioned-generation pipeline falls back to Tier 2 demographic-only).
        """
