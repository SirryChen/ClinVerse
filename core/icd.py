"""ICD-10 normalization and diagnosis agreement."""
from __future__ import annotations
import re
from functools import lru_cache
from icdmappings import Mapper

_mapper = Mapper()


def _clean(code: str | None) -> str:
    return str(code or '').strip().upper().replace('.', '')


def _valid(code: str) -> bool:
    return bool(re.fullmatch(r'[A-Z][0-9][A-Z0-9][A-Z0-9]{0,4}', code))


@lru_cache(maxsize=4096)
def to_icd10(code: str | None, version: int) -> str | None:
    clean = _clean(code)
    if not clean:
        return None
    if version == 9:
        mapped = _mapper.map(clean, source='icd9', target='icd10')
        clean = _clean(mapped)
    elif version != 10:
        raise ValueError('ICD version must be 9 or 10')
    return clean if _valid(clean) else None


@lru_cache(maxsize=4096)
def chapter(code: str | None) -> str | None:
    clean = to_icd10(code, 10)
    return _mapper.map(clean, source='icd10', target='chapter') if clean else None


@lru_cache(maxsize=4096)
def block(code: str | None) -> str | None:
    """Return the ICD-10 block, or None when no block is mapped."""
    clean = to_icd10(code, 10)
    return _mapper.map(clean, source='icd10', target='block') if clean else None


@lru_cache(maxsize=4096)
def category3(code: str | None) -> str:
    clean = to_icd10(code, 10)
    return clean[:3] if clean else ''


def same_block(a: str | None, b: str | None) -> bool:
    left, right = block(a), block(b)
    return bool(left) and left == right


def same_category3(a: str | None, b: str | None) -> bool:
    left, right = category3(a), category3(b)
    return bool(left) and left == right
