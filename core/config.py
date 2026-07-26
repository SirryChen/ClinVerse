"""Configuration helpers for public ClinVerse entry points."""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Z][A-Z0-9_]*)(?::-([^}]*))?\}")


def load_yaml_config(path: str | Path) -> dict:
    """Load YAML after resolving ``${ENV_VAR}`` placeholders."""
    source = Path(path).read_text(encoding="utf-8")
    missing = sorted({
        name
        for name, default in _ENV_PATTERN.findall(source)
        if not os.getenv(name) and not default
    })
    if missing:
        names = ", ".join(missing)
        raise RuntimeError(f"Missing required environment variable(s): {names}")
    expanded = _ENV_PATTERN.sub(
        lambda match: os.getenv(match.group(1), match.group(2) or ""),
        source,
    )
    return yaml.safe_load(expanded)
