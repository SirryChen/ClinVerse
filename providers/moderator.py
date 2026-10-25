"""Moderator: routes actions to providers, records metadata, controls turn flow."""

from __future__ import annotations
from copy import deepcopy

from .patient_actor import PatientActor
from .exam_provider import ExamProvider
from .lab_provider import LabProvider


class Moderator:
    def __init__(
        self,
        case: dict,
        patient_actor: PatientActor,
        exam_provider: ExamProvider,
        lab_provider: LabProvider,
        max_turns: int = 20,
    ):
        self.case = case
        self._patient_actor = patient_actor
        self._exam_provider = exam_provider
        self._lab_provider = lab_provider
        self.max_turns = max_turns
        self.reset()

    def reset(self) -> None:
        self._turn = 0
        self._hidden_log = []
        self._conversation_history = []

    @property
    def turn(self) -> int:
        return self._turn

    @property
    def hidden_log(self) -> list[dict]:
        return self._hidden_log

    def dispatch(self, function_name: str, args: dict) -> str:
        """Route a doctor function-call to the appropriate provider. Returns visible text."""
        if function_name not in {"ask_patient", "request_physical_exam", "order_tests", "order_imaging"}:
            raise ValueError(f"Unknown function: {function_name}")
        if self.is_over():
            raise RuntimeError("Turn budget exhausted")
        self._turn += 1

        if function_name == "ask_patient":
            patient_args = {**args, "_conversation_history": deepcopy(self._conversation_history)}
            text, meta = self._patient_actor.respond(patient_args, self.case)
            meta["turn"] = self._turn
            self._hidden_log.append(meta)
            self._conversation_history.append({"question": args.get("question", ""), "response": text})
            return text

        if function_name == "request_physical_exam":
            text, meta = self._exam_provider.respond(args, self.case)
            meta["turn"] = self._turn
            self._hidden_log.append(meta)
            return text

        if function_name == "order_tests":
            text, meta = self._lab_provider.order_tests(args, self.case)
            meta["turn"] = self._turn
            self._hidden_log.append(meta)
            return text

        if function_name == "order_imaging":
            text, meta = self._lab_provider.order_imaging(args, self.case)
            meta["turn"] = self._turn
            self._hidden_log.append(meta)
            return text

        raise ValueError(f"Unknown function: {function_name}")

    def is_over(self) -> bool:
        return self._turn >= self.max_turns

    def ground_truth(self) -> dict:
        return self.case["simulation_views"]["moderator_view"]["ground_truth"]

    def all_disclosed_info_ids(self) -> list[str]:
        ids = []
        for entry in self._hidden_log:
            ids.extend(entry.get("disclosed_info_ids", []))
        return list(set(ids))

    def all_returned_evidence_ids(self) -> list[str]:
        ids = []
        for entry in self._hidden_log:
            ids.extend(entry.get("returned_evidence_ids", []))
        return list(set(ids))

    def all_returned_abnormal_ids(self) -> list[str]:
        ids = []
        for entry in self._hidden_log:
            ids.extend(entry.get("returned_abnormal_evidence_ids", []))
        return list(set(ids))
