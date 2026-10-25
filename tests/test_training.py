from types import SimpleNamespace
import hashlib
import json
import pytest
from evaluation import harness
from agents.interfaces import Action
from tests.test_regressions import make_case


class Agent:
    def __init__(self, *args): pass
    def reset(self, context): self.context = context
    def act(self): return Action('finalize_case', {'icd10_code': 'I21.9'})
    def observe(self, text): raise AssertionError('Finalization should end the encounter')


class Env:
    max_turns = 3
    done = False
    def __init__(self, case_id): self.case_id = case_id
    def reset(self):
        from core.presentation import visible_from_case
        self.visible = visible_from_case(make_case())
        return SimpleNamespace(case_id=self.case_id, visible_info=self.visible,
                               tools_schema=[{'function': {'name': 'finalize_case'}}])
    def step(self, name, args):
        self.done = True
        return SimpleNamespace(observation='', done=True, info={})
    @property
    def transcript(self):
        return SimpleNamespace(to_dict=lambda: {'initial_visible_info': self.visible, 'turns': [],
                                  'final_answer': {'icd10_code': 'I21.9'}, 'termination_reason': 'finalize_case'})
    def score(self):
        return {'diagnosis_score': 1., 'information_collection_score': 0.4,
                'test_order_coverage': 0.5, 'abnormal_test_coverage': 0.5}


@pytest.fixture
def runner(tmp_path, monkeypatch):
    case = tmp_path / 'synthetic.json'
    case.write_text(json.dumps({'core_admission': {'source': 'test'}}))
    ids = tmp_path / 'ids.txt';ids.write_text('synthetic\n')
    monkeypatch.setattr(harness, 'TaskStream', lambda *a, **kw: [SimpleNamespace(case_id='synthetic', path=case)])
    monkeypatch.setattr(harness.ClinicalEnv, 'from_case_file', lambda path, *a, **kw: Env(path.stem))
    original = harness.import_object
    monkeypatch.setattr(harness, 'import_object', lambda name: Agent if name == 'test:Agent' else original(name))
    class Encoder:
        model = 'test'
        def __init__(self, provider): pass
        def encode(self, texts): return [[1., 0.] for text in texts]
    monkeypatch.setattr('examples.memory_trainer.EmbeddingEncoder', Encoder)
    return ['--agent', 'test:Agent', '--case-ids-file', str(ids)]


def test_training_and_frozen_evaluation(runner, tmp_path):
    state = tmp_path / 'memory.json'
    harness.main(runner + ['--output', str(tmp_path/'train'), '--trainer', 'examples.memory_trainer:MemoryTrainer', '--trainer-state', str(state)])
    before = state.read_bytes()
    harness.main(runner + ['--output', str(tmp_path/'eval'), '--memory-state', str(state)])
    assert state.read_bytes() == before
    record = json.loads((tmp_path/'eval/traces/synthetic.json').read_text())
    assert record['memrl']['updated'] is False
    assert json.loads(state.read_text())['items'][0]['utility']['visits'] == 0
    harness.main(runner + ['--output', str(tmp_path/'online'), '--memory-state', str(state), '--memrl-update-output', str(state)])
    assert json.loads(state.read_text())['items'][0]['utility']['visits'] == 1
    assert json.loads((tmp_path/'online/traces/synthetic.json').read_text())['memrl']['updated'] is True


def test_existing_output_and_resume_settings(runner, tmp_path):
    output = tmp_path/'eval'
    harness.main(runner + ['--output', str(output)])
    with pytest.raises(FileExistsError):
        harness.main(runner + ['--output', str(output)])
    assert harness.main(runner + ['--output', str(output), '--resume'])['n'] == 1
    with pytest.raises(ValueError, match='settings'):
        harness.main(runner + ['--output', str(output), '--resume', '--max-turns', '8'])


def test_unknown_memory_state_is_an_error(runner, tmp_path):
    with pytest.raises(FileNotFoundError):
        harness.main(runner + ['--output', str(tmp_path/'eval'), '--memory-state', str(tmp_path/'missing.json')])


def test_stale_trace_directory_is_rejected(runner, tmp_path):
    output = tmp_path/'old'
    (output/'traces').mkdir(parents=True)
    (output/'traces/previous.json').write_text('{"case_id":"previous","scores":{}}')
    with pytest.raises(FileExistsError):
        harness.main(runner + ['--output', str(output)])


def test_failed_episode_is_not_scored_as_completed(runner, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError('synthetic model error')
    monkeypatch.setattr(harness, 'run_episode', fail)
    output = tmp_path/'failed'
    with pytest.raises(RuntimeError, match='errors'):
        harness.main(runner + ['--output', str(output)])
    assert json.loads((output/'summary.json').read_text())['n'] == 0
    assert not list((output/'traces').glob('*.json'))
    assert 'synthetic model error' in (output/'errors.json').read_text()
