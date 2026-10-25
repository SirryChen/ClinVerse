"""Trim patient backgrounds by time and diagnostic disclosure."""
from __future__ import annotations
import re
from datetime import datetime


def _to_date(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace('Z', '+00:00')).replace(tzinfo=None)
    except ValueError:
        raise ValueError(f'Invalid event date: {value!r}')


def _normalize(text: str) -> str:
    return ' '.join(re.findall(r'\w+', str(text).casefold()))


def reveals_diagnosis(text: str, diagnosis_title: str, code: str = '') -> bool:
    text, title = _normalize(text), _normalize(diagnosis_title)
    if title and title in text:
        return True
    if code and re.search(r'(?<![A-Za-z0-9])' + re.escape(code) + r'(?![A-Za-z0-9])', text, re.I):
        return True
    stop = {'with', 'without', 'unspecified', 'chronic', 'acute', 'disease', 'disorder', 'syndrome'}
    words = set(title.split()) - stop
    return len(words) >= 3 and len(words & set(text.split())) >= 3


def trim_persona(persona: dict, index_date: str, diagnosis: dict) -> dict:
    index = _to_date(index_date)
    events = []
    result = dict(persona)
    specifications = {'conditions': ('onset_date', 'display'), 'medications': ('authored_date', 'display'),
                      'encounters': ('start', 'type'), 'observations': ('date', 'display'),
                      'allergies': ('recorded_date', 'substance'), 'immunizations': ('date', 'vaccine')}
    for kind, (date_field, label_field) in specifications.items():
        kept, seen = [], set()
        for item in persona.get(kind, []):
            date = _to_date(item.get(date_field, ''))
            label = item.get(label_field, '')
            reason = 'post_index_time' if index and date and date > index else None
            if reveals_diagnosis(label, diagnosis.get('long_title', ''), diagnosis.get('icd_code', '')):
                reason = 'leakage_risk'
            if reason:
                events.append({'resource_type': kind, 'display': label, 'reason': reason,
                               'leakage_risk': reason == 'leakage_risk'})
            elif kind in {'encounters', 'observations'} or _normalize(label) not in seen:
                kept.append(item)
                seen.add(_normalize(label))
        result[kind] = kept
    return {'trimmed_persona': result, 'trimmed_events': events}
