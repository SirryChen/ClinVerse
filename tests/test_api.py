from types import SimpleNamespace
from fastapi.testclient import TestClient
from env import api


def test_api_exposes_only_observations_until_completion(monkeypatch, tmp_path):
    class Env:
        done = False
        def reset(self):
            return SimpleNamespace(case_id="synthetic", visible_info={"chief_complaint": "Test observation"}, tools_schema=[{"function": {"name": "finalize_case"}}])
        def step(self, tool, args):
            self.done = True
            return SimpleNamespace(observation="", done=True, info={"hidden_metadata": "private"})
        def score(self):
            return {"diagnosis_score": 1.0}
    monkeypatch.setattr(api, "TaskStream", lambda path: [SimpleNamespace(case_id="synthetic", path=tmp_path / "case.json", category="must not be exposed")])
    monkeypatch.setattr(api.ClinicalEnv, "from_case_file", lambda *a, **kw: Env())
    cfg = {"paths": {"cases": "cases"}, "datasets": {}, "llm_providers": {"patient": {}, "lab": {}}, "simulation": {"max_turns": 2}, "default_patient_provider": "patient", "default_lab_provider": "lab"}
    client = TestClient(api.create_app(cfg, tmp_path))
    assert client.get('/cases').json() == [{"case_id": "synthetic"}]
    session = client.post('/sessions', json={"case_id": "synthetic"}).json()
    prefix = f"/sessions/{session['session_id']}"
    assert client.get(prefix + '/score').status_code == 409
    assert client.post(prefix + '/step', json={"tool": "read_hidden_case"}).status_code == 400
    result = client.post(prefix + '/step', json={"tool": "finalize_case", "args": {}}).json()
    assert result == {"observation": "", "done": True}
    assert client.get(prefix + '/score').json() == {"diagnosis_score": 1.0}
