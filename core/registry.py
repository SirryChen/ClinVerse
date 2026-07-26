"""Dataset adapter registry: name -> DatasetAdapter instance.

Deliberately a small explicit dispatch table, not a dynamic plugin-discovery
mechanism (entry_points / importlib metadata) — there are two data sources
today and adding a third means adding one line here, which is simpler than
maintaining a discovery protocol for a number of plugins that will stay in
the single digits for a long time. If that stops being true, swap the body
of `_FACTORIES` for an entry_points scan without changing get_adapter()'s
call sites.

Usage:
    from core.registry import get_adapter
    adapter = get_adapter("sicdb-ext")           # uses config/default.yaml paths
    adapter = get_adapter("sicdb-ext", data_dir="/custom/path")
"""
from __future__ import annotations

from typing import Callable

from .ports import DatasetAdapter

_FACTORIES: dict[str, Callable[..., DatasetAdapter]] = {}
_INSTANCES: dict[str, DatasetAdapter] = {}


def register(name: str, factory: Callable[..., DatasetAdapter]) -> None:
    """Register a factory for a dataset name. Called once by each adapter
    package's __init__.py (see adapters/mimic_iv_ext/__init__.py)."""
    _FACTORIES[name] = factory


def get_adapter(name: str, **construct_kwargs) -> DatasetAdapter:
    """Return the (cached) adapter instance for `name`.

    The first caller's construct_kwargs win — within one process every
    consumer (LabProvider, GPTDoctorAgent, Moderator, the case-building
    scripts) is expected to share the same adapter instance for a given
    dataset, the same way the pre-refactor code shared one catalog dict for
    an entire batch run.
    """
    if name not in _INSTANCES:
        if name not in _FACTORIES:
            _ensure_builtin_adapters_imported()
        if name not in _FACTORIES:
            raise KeyError(
                f"No adapter registered for dataset '{name}'. "
                f"Known datasets: {sorted(_FACTORIES)}"
            )
        _INSTANCES[name] = _FACTORIES[name](**construct_kwargs)
    return _INSTANCES[name]


def known_datasets() -> list[str]:
    _ensure_builtin_adapters_imported()
    return sorted(_FACTORIES)


def reset() -> None:
    """Clear cached instances (tests only — factories stay registered)."""
    _INSTANCES.clear()


_builtins_imported = False


def _ensure_builtin_adapters_imported() -> None:
    """Importing these packages triggers their register(...) call. Done
    lazily (not at module import time) to avoid paying the import cost for
    adapters a given process never uses, and to avoid an import cycle since
    the adapters import from core."""
    global _builtins_imported
    if _builtins_imported:
        return
    _builtins_imported = True
    import adapters.mimic_iv_ext  # noqa: F401
    import adapters.sicdb_ext  # noqa: F401
