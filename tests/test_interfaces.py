from copy import deepcopy
from types import SimpleNamespace
import pytest
from agents.interfaces import Action
from env.runner import run_episode
from pipeline.patient_info_units import build_patient_information_units, require_patient_information_units


UNIT = {"text": "Synthetic test observation.", "category": "symptom", "source": "test",
        "importance": "key", "patient_visible": True, "initially_visible": False}


def test_extraction_requires_model_and_validates_schema():
    with pytest.raises(ValueError, match="extractor"):
        build_patient_information_units({}, {})
    extractor = SimpleNamespace(extract=lambda payload: [deepcopy(UNIT)])
    units = build_patient_information_units({}, {}, extractor)
    assert units[0]["info_id"] == "info_0001"
    assert units[0]["text"] == UNIT["text"]
    invalid = SimpleNamespace(extract=lambda payload: [{**UNIT, "patient_visible": "false"}])
    with pytest.raises(ValueError, match="boolean"):
        build_patient_information_units({}, {}, invalid)
    with pytest.raises(ValueError, match="nonempty"):
        require_patient_information_units({})


class FakeEnv:
    max_turns = 3
    done = False

    def reset(self):
        self.actions = []
        self.visible = {"chief_complaint": "Synthetic observation"}
        self.done = False
        return SimpleNamespace(case_id="synthetic_case", visible_info=self.visible,
                               tools_schema=[{"function": {"name": n}} for n in ["ask_patient", "finalize_case"]])

    def step(self, name, arguments):
        self.actions.append(name)
        self.done = name == "finalize_case"
        return SimpleNamespace(observation="Observable response", done=self.done,
                               info={"ground_truth": "must not reach agent"})

    @property
    def transcript(self):
        return SimpleNamespace(to_dict=lambda: {"turns": list(self.actions)})

    def score(self):
        assert self.done
        return {"diagnosis_score": 1.0}


class Agent:
    def reset(self, context):
        self.context = context
        self.calls = 0
        self.observations = []
        context.visible_info["chief_complaint"] = "Changed agent copy"

    def act(self):
        self.calls += 1
        return Action("ask_patient" if self.calls == 1 else "finalize_case", {})

    def observe(self, value):
        self.observations.append(value)


def test_agent_observation_boundary_and_terminal_scoring():
    env, agent = FakeEnv(), Agent()
    result = run_episode(env, agent)
    assert agent.observations == ["Observable response"]
    assert not hasattr(agent.context, "info")
    assert not hasattr(agent.context, "case")
    assert env.visible["chief_complaint"] == "Synthetic observation"
    assert result["scores"]["diagnosis_score"] == 1.0


def test_invalid_action_is_not_dispatched():
    env, agent = FakeEnv(), Agent()
    agent.act = lambda: Action("read_hidden_case", {})
    with pytest.raises(ValueError, match="schema"):
        run_episode(env, agent)
    assert env.actions == []
