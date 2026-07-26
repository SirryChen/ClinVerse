"""Moderator: routes actions to providers, records metadata, controls turn flow."""

from __future__ import annotations

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
        self._turn = 0
        self._hidden_log: list[dict] = []
        self._conversation_history: list[dict] = []  # for patient actor memory

    @property
    def turn(self) -> int:
        return self._turn

    @property
    def hidden_log(self) -> list[dict]:
        return self._hidden_log

    def dispatch(self, function_name: str, args: dict) -> str:
        """Route a doctor function-call to the appropriate provider. Returns visible text."""
        self._turn += 1

        if function_name == "ask_patient":
            args["_conversation_history"] = self._conversation_history
            text, meta = self._patient_actor.respond(args, self.case)
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
            ids.extend(entry.get("disclosed_evidence_ids", []))
        return list(set(ids))

    def all_disclosed_evidence_ids(self) -> list[str]:
        """Backward-compatible alias for older callers."""
        return self.all_disclosed_info_ids()

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
