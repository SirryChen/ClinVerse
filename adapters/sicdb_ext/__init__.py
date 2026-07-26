from core.registry import register

from .adapter import SicdbAdapter
from .condition_priors import extract_condition_priors

register("sicdb-ext", SicdbAdapter)

__all__ = ["SicdbAdapter", "extract_condition_priors"]
