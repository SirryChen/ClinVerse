from copy import deepcopy
from types import SimpleNamespace
from pathlib import Path
import ast
import json
import pytest
from env.clinical_env import ClinicalEnv, _ensure_patient_information_units
from providers.moderator import Moderator
from providers.patient_actor import PatientActor
from providers.lab_provider import LabProvider
from pipeline.view_builder import build_views
from pipeline.evidence_builder import build_evidence_inventory
from pipeline.trimmer import trim_persona
from pipeline.persona_selector import select_persona
from evaluation.scorer import score_transcript
from evaluation.metrics import diagnosis_category_accuracy
from core.presentation import visible_from_case
from doctor_train.memrl import empty_bank, retrieve_memories, update_memory_values


def make_case():
    units = [{'info_id': 'c', 'text': 'Chest pain for two hours.', 'category': 'chief_complaint', 'source': 'hpi',
              'importance': 'key', 'patient_visible': True, 'initially_visible': True},
             {'info_id': 's', 'text': 'Pain worsens with breathing.', 'category': 'symptom', 'source': 'hpi',
              'importance': 'supporting', 'patient_visible': True, 'initially_visible': False}]
    admission = {'demographics': {'anchor_age': 60, 'gender': 'M', 'admission_type': 'Emergency'},
                 'hpi': 'Chest pain for two hours. Pain worsens with breathing.', 'vital_signs': '',
                 'primary_diagnosis': {'icd10_code': 'I21.9', 'icd_code': 'I219', 'long_title': 'Acute myocardial infarction'}}
    return {'case_id': 'synthetic', 'core_admission': admission, 'patient_information_units': units,
            'evidence_inventory': [], 'simulation_views': build_views(admission, {}, patient_information_units=units)}


def test_initial_units_and_scoring_share_presentation():
    case = make_case()
    _ensure_patient_information_units(case)
    assert visible_from_case(case)['chief_complaint'] == case['patient_information_units'][0]['text']
    result = score_transcript({'turns': [{'hidden_metadata': {'disclosed_info_ids': ['s']}}],
                               'final_answer': {'icd10_code': 'I21.9'}}, case)
    assert result['information_collection_score'] == 1
    case['simulation_views']['patient_actor_view']['initial_presentation']['chief_complaint'] = 'Wrong'
    with pytest.raises(ValueError, match='agree'):
        _ensure_patient_information_units(case)


def test_environment_resets_and_transcript_snapshots_do_not_mutate():
    case = make_case()
    patient = SimpleNamespace(respond=lambda args, case: ('Reply', {'disclosed_info_ids': ['s']}))
    env = ClinicalEnv.__new__(ClinicalEnv)
    env.case, env._tools_schema = case, []
    env._moderator = Moderator(case, patient, None, None, 4)
    env.reset()
    first = {'question': 'First'}
    env.step('ask_patient', first)
    snapshot = deepcopy(env.transcript.to_dict())
    env.step('ask_patient', {'question': 'Second'})
    assert env.transcript.to_dict()['turns'][0] == snapshot['turns'][0]
    assert first == {'question': 'First'}
    env.reset()
    assert env.turn == 0 and env._moderator.hidden_log == [] and env._moderator._conversation_history == []
    env.step('ask_patient', {'question': 'New'})
    assert env.turn == 1


def test_primary_diagnosis_and_future_events_are_removed():
    case = make_case()
    admission = case['core_admission']
    admission['demographics']['admittime'] = '2020-01-01'
    persona = {'conditions': [{'display': 'Acute myocardial infarction', 'onset_date': '2019-01-01'}],
               'immunizations': [{'vaccine': 'Future', 'date': '2030-01-01'}],
               'allergies': [{'substance': 'Future', 'recorded_date': '2030-01-01'}]}
    selected = select_persona(admission, {'personas': [persona], 'priors': {'condition_flags': {},
                                'forbidden_clues': ['Acute myocardial infarction']}})[0]['trimmed_persona']
    assert selected['conditions'] == [] and selected['immunizations'] == [] and selected['allergies'] == []
    view = case['simulation_views']['patient_actor_view']
    view['medical_background'] = {'past_medical_history': ['Acute myocardial infarction']}
    view['patient_information_units'].append({'info_id': 'hidden', 'text': 'Acute myocardial infarction',
        'category': 'symptom', 'source': 'test', 'importance': 'key', 'patient_visible': False, 'initially_visible': False})
    assert 'Acute myocardial infarction' not in PatientActor(None, 'test')._build_system_prompt(view)


def test_microbiology_results_keep_global_ids_and_abnormality():
    records = [{'test_name': 'Blood culture', 'spec_type_desc': 'BLOOD', 'valuestr': 'S. aureus', 'is_abnormal': True}]
    records += [{'test_name': 'Urine culture', 'spec_type_desc': 'URINE', 'valuestr': 'E. coli', 'is_abnormal': True} for _ in range(5)]
    lab = LabProvider.__new__(LabProvider)
    lab._language = 'en'
    text, ids, abnormal = lab._format_micro('urine_culture', records, {})
    assert ids == [f'mimic_micro_{i}' for i in range(1, 6)] == abnormal
    assert len(text) == 5
    _, ids, abnormal = lab._format_micro('sputum_culture', records, {})
    assert ids == abnormal == []
    inventory = build_evidence_inventory({'demographics': {}, 'microbiology': records}, {})
    assert all(item['is_abnormal'] for item in inventory if item['type'] == 'microbiology')


@pytest.mark.parametrize('failure', [RuntimeError('inference failed'), None])
def test_prediction_failure_does_not_become_a_missing_measurement(failure):
    lab = LabProvider.__new__(LabProvider)
    lab._model_path, lab._id_map, lab._llm_client = 'model', {'1': 'Test'}, None
    lab._has_real_lab_result = lambda *args: False
    lab._ensure_model_loaded = lambda: True
    lab._build_prediction_messages = lambda *args: ('Test', [])
    lab._item_prediction_key = lambda item: 'test'
    def infer(*args):
        if failure:
            raise failure
        return [None]
    lab._run_model_inference_batch = infer
    with pytest.raises(RuntimeError):
        lab._predict_missing_labs_batch([{}], {}, {})


def test_numeric_codes_do_not_receive_diagnostic_credit():
    assert diagnosis_category_accuracy('102.1', '999.9') == 0
    assert diagnosis_category_accuracy('I21.9', 'I21.0') == 1


def test_semantic_recall_precedes_utility_selection():
    bank = empty_bank('test')
    bank['items'] = [{'memory_id': str(i), 'embedding': vector, 'utility': {'q_value': q, 'visits': 0}}
                     for i, (vector, q) in enumerate([([1., 0.], 0.1), ([0.8, 0.6], 0.8), ([0., 1.], 1.0)])]
    selected = retrieve_memories(bank, [1., 0.], top_k=1, recall_k=2)
    assert selected[0]['memory_id'] == '1'
    update_memory_values(bank, ['1'], 0, alpha=0.5)
    assert bank['items'][1]['utility'] == {'q_value': 0.4, 'visits': 1}


def test_all_initially_visible_units_are_observable():
    from core.presentation import initial_presentation
    units = make_case()['patient_information_units']
    units.append({'info_id': 'initial', 'text': 'Uses a walking aid.', 'category': 'social_history',
                  'source': 'test', 'importance': 'background', 'patient_visible': True, 'initially_visible': True})
    assert 'Uses a walking aid.' in initial_presentation(units, {})['initial_information']


def test_embedding_responses_require_complete_indices():
    from doctor_train.embeddings import EmbeddingEncoder
    encoder = EmbeddingEncoder.__new__(EmbeddingEncoder)
    encoder.model = 'test'
    encoder.client = SimpleNamespace(embeddings=SimpleNamespace(create=lambda **kwargs:
        SimpleNamespace(data=[SimpleNamespace(index=1, embedding=[1., 0.])])))
    with pytest.raises(ValueError, match='indices'):
        encoder.encode(['synthetic'])


def test_clinlab_prompt_is_shared():
    from core.lab_prompt import _build_user_prompt
    from providers.lab_provider import _build_user_prompt as prediction_prompt
    from training.provider.mimic_iv_ext.prepare_data import _build_user_prompt as training_prompt
    assert prediction_prompt is training_prompt is _build_user_prompt


def test_microbiology_abnormality_comes_from_source_records():
    import pandas as pd
    from preprocessing.mimic_iv_ext.predata.labs import build_micro_records_streaming
    data = pd.DataFrame([
        {'hadm_id': 1, 'test_itemid': 2, 'charttime': '2020-01-01', 'org_itemid': 10,
         'org_name': 'E. coli', 'spec_itemid': 20, 'spec_type_desc': 'URINE', 'test_name': 'Urine culture',
         'ab_name': None, 'interpretation': None, 'comments': None},
        {'hadm_id': 1, 'test_itemid': 3, 'charttime': '2020-01-01', 'org_itemid': None,
         'org_name': None, 'spec_itemid': 21, 'spec_type_desc': 'BLOOD', 'test_name': 'Blood culture',
         'ab_name': None, 'interpretation': None, 'comments': 'NO GROWTH'}])
    records = build_micro_records_streaming([data], {1})
    assert records['is_abnormal'].tolist() == [True, False]


def test_clinical_encounter_uses_real_environment_and_scoring():
    from agents.interfaces import Action
    from env.runner import run_episode
    adapter = SimpleNamespace(has_radiology=False, lab_catalog=SimpleNamespace(describe_for_prompt=lambda: 'blood_count'))
    env = ClinicalEnv(make_case(), adapter, {'api_key': 'EMPTY', 'base_url': 'http://localhost:8000/v1',
                                            'patient_actor_model': 'test'}, clinlab_enabled=False)
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[
        SimpleNamespace(function=SimpleNamespace(arguments=json.dumps({'reply': 'Breathing makes it worse.', 'disclosed_info_ids': ['s']}))) ]))])
    env._patient_actor._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response)))
    class Doctor:
        def reset(self, context): self.turn = 0
        def act(self):
            self.turn += 1
            return Action('ask_patient', {'question': 'What worsens the pain?'}) if self.turn == 1 else Action('finalize_case', {'icd10_code': 'I21.9'})
        def observe(self, observation): assert observation == 'Breathing makes it worse.'
    result = run_episode(env, Doctor())
    assert result['scores']['diagnosis_score'] == 1
    assert result['scores']['information_collection_score'] == 1
    assert result['transcript']['termination_reason'] == 'finalize_case'


def test_generation_requires_a_prior_for_each_fixed_record(tmp_path):
    from pipeline.conditioned_synthea_runner import _run_synthea
    fixed = tmp_path/'fixed.json'
    fixed.write_text('{"records":[{"individualId":"synthetic"}]}')
    with pytest.raises(RuntimeError, match='Missing conditioning priors'):
        _run_synthea(tmp_path/'not-run.jar', fixed, tmp_path/'priors', tmp_path/'output')
