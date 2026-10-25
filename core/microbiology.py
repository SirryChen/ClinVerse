"""Microbiology result labels and specimen selection."""
import re


def is_abnormal(record: dict) -> bool:
    value = record.get('is_abnormal')
    if type(value) is not bool:
        raise ValueError('Microbiology records require a boolean abnormality label')
    return value


def matches_specimen(record: dict, test_type: str) -> bool:
    specimen = test_type.removesuffix('_culture').replace('_', ' ')
    text = f"{record.get('spec_type_desc', '')} {record.get('test_name', '')}".casefold()
    return bool(re.search(r'\b' + re.escape(specimen) + r'\b', text))
