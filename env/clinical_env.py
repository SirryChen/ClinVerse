"""ClinicalEnv: a Gym-style clinical environment, decoupled from any specific
doctor agent. An external caller (any agent/training framework) drives the
loop by calling reset()/step()/score() — the environment never calls back
into an agent implementation itself.

reset() only returns what a real physician could legitimately know at the
start of an encounter (age/gender/vitals/chief complaint). It deliberately
does NOT include any ground-truth-derived clinical category — that would
leak the answer's disease family before the agent has done any work. Such
labels exist only as dataset-level metadata in task_stream.CaseRef, for an
external harness's own train/test split or per-category score breakdown.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from openai import OpenAI

from agents.tools import build_doctor_tools
from core.registry import get_adapter
from providers.patient_actor import PatientActor
from providers.exam_provider import ExamProvider
from providers.lab_provider import LabProvider
from providers.moderator import Moderator
from simulation.transcript import Transcript
from evaluation.scorer import score_transcript
from pipeline.patient_info_units import require_patient_information_units
from core.presentation import visible_from_case, initial_presentation

if TYPE_CHECKING:
    from core.ports import DatasetAdapter

DEFAULT_LLM_TIMEOUT = 180
DEFAULT_LOCAL_LAB_GENERATION_TIMEOUT = 60


@dataclass
class Observation:
    case_id: str
    visible_info: dict
    tools_schema: list = field(default_factory=list)


@dataclass
class StepResult:
    observation: str
    done: bool
    info: dict = field(default_factory=dict)


def _timeout_from_config(config: dict, default: int = DEFAULT_LLM_TIMEOUT):
    timeout = config.get("timeout", default)
    return default if timeout is None else timeout


def _load_case(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _build_initial_visible(case: dict) -> dict:
    return visible_from_case(case)


def _ensure_patient_information_units(case: dict) -> None:
    units = require_patient_information_units(case)
    patient = case.get("simulation_views", {}).get("patient_actor_view", {})
    core = case["core_admission"]
    expected = initial_presentation(units, core["demographics"], core.get("vital_signs", ""))
    if patient.get("initial_presentation") != expected:
        raise ValueError("Initial presentation and information units must agree")
    if patient.get("patient_information_units") != units:
        raise ValueError("Case information units and patient view must agree")


class ClinicalEnv:
    def __init__(
        self,
        case: dict,
        adapter: "DatasetAdapter",
        patient_openai_config: dict,
        lab_openai_config: dict | None = None,
        language: str = "en",
        max_turns: int = 20,
        lab_mask_ratio: float = 0.0,
        lab_mask_seed: int = 0,
        clinlab_enabled: bool = True,
    ):
        _ensure_patient_information_units(case)
        patient_client = OpenAI(
            api_key=patient_openai_config["api_key"],
            base_url=patient_openai_config["base_url"],
            timeout=_timeout_from_config(patient_openai_config),
        )
        lab_cfg = lab_openai_config or patient_openai_config
        use_local_lab_model = clinlab_enabled and bool(lab_cfg.get("model_path"))
        lab_client = None
        lab_model = None
        if clinlab_enabled and not use_local_lab_model:
            lab_client = patient_client if lab_cfg is patient_openai_config else OpenAI(
                api_key=lab_cfg["api_key"],
                base_url=lab_cfg["base_url"],
                timeout=_timeout_from_config(lab_cfg),
            )
            lab_model = (
                lab_cfg.get("lab_model")
                or lab_cfg.get("patient_actor_model")
                or lab_cfg.get("doctor_model")
            )

        self.case = case
        self.adapter = adapter
        self.max_turns = max_turns

        self._patient_actor = PatientActor(
            patient_client,
            patient_openai_config["patient_actor_model"],
            language=language,
            extra_body=patient_openai_config.get("extra_body"),
        )
        self._exam_provider = ExamProvider(language=language)
        self._lab_provider = LabProvider(
            adapter,
            language=language,
            adapter_path=lab_cfg.get("adapter_path") if clinlab_enabled else None,
            model_path=lab_cfg.get("model_path") if clinlab_enabled else None,
            lab_mapping_path=lab_cfg.get("lab_mapping_path") if clinlab_enabled else None,
            local_device=lab_cfg.get("local_device"),
            local_batch_size=lab_cfg.get("local_batch_size", 4),
            local_generation_timeout=lab_cfg.get(
                "local_generation_timeout",
                DEFAULT_LOCAL_LAB_GENERATION_TIMEOUT,
            ),
            llm_client=lab_client,
            llm_model=lab_model,
            llm_extra_body=lab_cfg.get("extra_body"),
            mask_ratio=lab_mask_ratio,
            mask_seed=lab_mask_seed,
            clinlab_enabled=clinlab_enabled,
        )
        self._moderator = Moderator(
            case, self._patient_actor, self._exam_provider, self._lab_provider, max_turns
        )
        self._tools_schema = build_doctor_tools(adapter)
        self._transcript: Optional[Transcript] = None
        self._done = False

    def reset(self) -> Observation:
        self._moderator.reset()
        self._transcript = Transcript(
            case_id=self.case["case_id"],
            initial_visible_info=_build_initial_visible(self.case),
        )
        self._done = False
        return Observation(
            case_id=self.case["case_id"],
            visible_info=self._transcript.initial_visible_info,
            tools_schema=self._tools_schema,
        )

    def step(self, tool_name: str, args: dict) -> StepResult:
        if self._transcript is None:
            raise RuntimeError("Call reset() before step().")
        if self._done:
            raise RuntimeError("Episode already finished; call reset() to start a new one.")

        if tool_name == "finalize_case":
            self._transcript.set_final_answer(args, "finalize_case")
            self._done = True
            return StepResult(observation="", done=True, info={})

        provider_text = self._moderator.dispatch(tool_name, args)
        hidden_meta = self._moderator.hidden_log[-1] if self._moderator.hidden_log else {}
        self._transcript.add_turn(
            turn_num=self._moderator.turn,
            function_name=tool_name,
            function_args=args,
            tool_call_id="",
            provider_response=provider_text,
            hidden_metadata=hidden_meta,
        )

        if self._moderator.is_over():
            self._transcript.set_final_answer({}, "max_turns_reached")
            self._done = True

        return StepResult(observation=provider_text, done=self._done, info=hidden_meta)

    @property
    def done(self) -> bool:
        return self._done

    @property
    def turn(self) -> int:
        """Number of non-finalize tool calls dispatched so far (0 before the
        first step()). Lets an external loop detect "one slot left before
        max_turns" and force convergence on its next action, instead of
        running out the clock into an empty max_turns_reached answer."""
        return self._moderator.turn

    @property
    def transcript(self) -> Transcript:
        if self._transcript is None:
            raise RuntimeError("Call reset() before accessing transcript.")
        return self._transcript

    def score(self) -> dict:
        if not self._done:
            raise RuntimeError("Episode not finished yet; score() is only valid after done=True.")
        return score_transcript(self._transcript, self.case)

    @classmethod
    def from_case_file(
        cls,
        case_path: str | Path,
        patient_openai_config: dict,
        lab_openai_config: dict | None = None,
        datasets_config: dict[str, dict] | None = None,
        max_turns: int = 20,
        language: str = "en",
        lab_mask_ratio: float = 0.0,
        lab_mask_seed: int = 0,
        clinlab_enabled: bool = True,
    ) -> "ClinicalEnv":
        """Load a case and pick the right DatasetAdapter for it automatically,
        based on case["core_admission"]["source"] (set by pipeline.case_builder
        from whichever adapter built the case — "mimic-iv-ext" | "sicdb-ext" | ...).

        datasets_config: {dataset_name: {kwarg: value, ...}} passed through to
        core.registry.get_adapter() — e.g. config/default.yaml's `datasets:`
        section, letting each dataset override its data_dir/catalog paths.
        """
        case = _load_case(case_path)
        _ensure_patient_information_units(case)
        source = case.get("core_admission", {}).get("source", "mimic-iv-ext")
        adapter_kwargs = (datasets_config or {}).get(source, {})
        adapter = get_adapter(source, **adapter_kwargs)
        return cls(
            case,
            adapter,
            patient_openai_config,
            lab_openai_config=lab_openai_config,
            language=language,
            max_turns=max_turns,
            lab_mask_ratio=lab_mask_ratio,
            lab_mask_seed=lab_mask_seed,
            clinlab_enabled=clinlab_enabled,
        )
