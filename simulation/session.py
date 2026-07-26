"""SimulationSession: a reference baseline that drives ClinicalEnv with
GPTDoctorAgent. This is itself just one example of "an external agent calling
the environment" — env.clinical_env.ClinicalEnv is the actual environment;
any other agent/training framework can drive the same env the same way
without going through this class at all.
"""

from __future__ import annotations

from pathlib import Path

from openai import OpenAI

from agents.gpt_agent import GPTDoctorAgent

DEFAULT_LLM_TIMEOUT = 180


def _initial_message(visible: dict) -> str:
    parts = [
        f"Patient: {visible.get('age')} y/o {visible.get('gender')}",
        f"Admission: {visible.get('admission_type', '')}",
        f"Vitals: {visible.get('vital_signs', 'not recorded')}",
        f"Chief complaint: {visible.get('chief_complaint', 'not specified')}",
    ]
    return "\n".join(parts)


class SimulationSession:
    def __init__(
        self,
        case: dict,
        patient_openai_config: dict,
        adapter,
        max_turns: int = 20,
        language: str = "en",
        doctor_openai_config: dict | None = None,
        lab_openai_config: dict | None = None,
        doctor_memory_context: str | None = None,
    ):
        """patient_openai_config drives the patient simulator.
        lab_openai_config drives missing-lab fallback prediction and defaults
        to the patient config. doctor_openai_config is the model under test
        and defaults to the patient config."""
        from env.clinical_env import ClinicalEnv  # deferred: avoids env<->simulation import cycle

        self.case = case
        self.max_turns = max_turns

        self._env = ClinicalEnv(
            case,
            adapter,
            patient_openai_config,
            lab_openai_config=lab_openai_config,
            language=language,
            max_turns=max_turns,
        )
        doctor_cfg = doctor_openai_config or patient_openai_config
        doctor_client = OpenAI(
            api_key=doctor_cfg["api_key"],
            base_url=doctor_cfg["base_url"],
            timeout=doctor_cfg.get("timeout") or DEFAULT_LLM_TIMEOUT,
        )
        self._agent = GPTDoctorAgent(
            doctor_client,
            doctor_cfg["doctor_model"],
            language=language,
            adapter=adapter,
            extra_body=doctor_cfg.get("extra_body"),
            memory_context=doctor_memory_context,
            message_style=doctor_cfg.get("doctor_message_style", "openai"),
        )

    def run(self) -> dict:
        """Execute the full simulation. Returns evaluation result dict."""
        obs = self._env.reset()

        fn_name, fn_args, call_id = self._agent.step(
            [{"role": "user", "content": _initial_message(obs.visible_info)}]
        )

        while True:
            step = self._env.step(fn_name, fn_args)
            if step.done:
                break
            self._agent.add_tool_result(call_id, step.observation)
            # One tool-call slot left before the env's max_turns budget runs
            # out: force this next call to be finalize_case so the episode
            # ends with a real diagnosis instead of an empty max_turns_reached
            # answer.
            force_finalize = self._env.turn >= self._env.max_turns - 1
            fn_name, fn_args, call_id = self._agent.step([], force_finalize=force_finalize)

        scores = self._env.score()
        return {
            "case_id": self.case["case_id"],
            "transcript": self._env.transcript.to_dict(),
            "scores": scores,
        }

    @classmethod
    def from_case_file(
        cls,
        case_path: str | Path,
        patient_openai_config: dict,
        datasets_config: dict[str, dict] | None = None,
        max_turns: int = 20,
        language: str = "en",
        doctor_openai_config: dict | None = None,
        lab_openai_config: dict | None = None,
        doctor_memory_context: str | None = None,
    ) -> "SimulationSession":
        from env.clinical_env import ClinicalEnv  # deferred: avoids env<->simulation import cycle

        env = ClinicalEnv.from_case_file(
            case_path=case_path,
            patient_openai_config=patient_openai_config,
            lab_openai_config=lab_openai_config,
            datasets_config=datasets_config,
            max_turns=max_turns,
            language=language,
        )
        return cls(
            env.case, patient_openai_config, env.adapter, max_turns, language,
            doctor_openai_config=doctor_openai_config,
            lab_openai_config=lab_openai_config,
            doctor_memory_context=doctor_memory_context,
        )
