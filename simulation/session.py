"""Run the reference doctor agent through a clinical environment."""
from env.clinical_env import ClinicalEnv
from env.runner import run_episode
from examples.openai_agent import OpenAIAgent


class SimulationSession:
    def __init__(self, case, patient_openai_config, adapter, max_turns=20, language='en',
                 doctor_openai_config=None, lab_openai_config=None, doctor_memory_context=None,
                 lab_mask_ratio=0.0, lab_mask_seed=0, clinlab_enabled=True):
        self._env = ClinicalEnv(case, adapter, patient_openai_config, lab_openai_config,
                                language, max_turns, lab_mask_ratio, lab_mask_seed, clinlab_enabled)
        self._agent = OpenAIAgent(doctor_openai_config or patient_openai_config, language)
        self._context = doctor_memory_context or ''

    def run(self):
        return run_episode(self._env, self._agent, memory_context=self._context)

    @classmethod
    def from_case_file(cls, case_path, patient_openai_config, datasets_config=None,
                       doctor_openai_config=None, lab_openai_config=None,
                       max_turns=20, language='en', doctor_memory_context=None,
                       lab_mask_ratio=0.0, lab_mask_seed=0, clinlab_enabled=True):
        session = cls.__new__(cls)
        session._env = ClinicalEnv.from_case_file(case_path, patient_openai_config,
            lab_openai_config=lab_openai_config, datasets_config=datasets_config,
            max_turns=max_turns, language=language, lab_mask_ratio=lab_mask_ratio,
            lab_mask_seed=lab_mask_seed, clinlab_enabled=clinlab_enabled)
        session._agent = OpenAIAgent(doctor_openai_config or patient_openai_config, language)
        session._context = doctor_memory_context or ''
        return session
