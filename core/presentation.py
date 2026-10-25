"""Initial observations shared by case construction and agent retrieval."""


def initial_presentation(units: list[dict], demographics: dict, vital_signs: str = '') -> dict:
    complaints = [u['text'].strip() for u in units if u['category'] == 'chief_complaint' and u['patient_visible'] and u['initially_visible']]
    if not complaints:
        raise ValueError('Information units require an initially visible chief complaint')
    return {'age': demographics.get('anchor_age'), 'gender': demographics.get('gender'),
            'admission_type': demographics.get('admission_type'), 'vital_signs': vital_signs,
            'chief_complaint': ' '.join(complaints),
            'initial_information': [unit['text'] for unit in units if unit['patient_visible'] and unit['initially_visible']]}


def visible_from_case(case: dict) -> dict:
    initial = case['simulation_views']['patient_actor_view']['initial_presentation']
    if not initial.get('chief_complaint'):
        raise ValueError('Case initial_presentation requires chief_complaint')
    return {key: initial.get(key) for key in ('age', 'gender', 'admission_type', 'vital_signs', 'chief_complaint', 'initial_information')}
