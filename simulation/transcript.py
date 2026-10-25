"""Records the full turn-by-turn simulation log."""

from __future__ import annotations
from copy import deepcopy

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Turn:
    turn_num: int
    function_name: str
    function_args: dict
    tool_call_id: str
    provider_response: str
    hidden_metadata: dict


@dataclass
class Transcript:
    case_id: str
    initial_visible_info: dict
    turns: list[Turn] = field(default_factory=list)
    final_answer: dict | None = None
    termination_reason: str = ""

    def add_turn(
        self,
        turn_num: int,
        function_name: str,
        function_args: dict,
        tool_call_id: str,
        provider_response: str,
        hidden_metadata: dict,
    ) -> None:
        self.turns.append(Turn(
            turn_num=turn_num,
            function_name=function_name,
            function_args=deepcopy(function_args),
            tool_call_id=tool_call_id,
            provider_response=provider_response,
            hidden_metadata=deepcopy(hidden_metadata),
        ))

    def set_final_answer(self, args: dict, reason: str = "finalize_case") -> None:
        self.final_answer = deepcopy(args)
        self.termination_reason = reason

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "initial_visible_info": self.initial_visible_info,
            "turns": [
                {
                    "turn": t.turn_num,
                    "function": t.function_name,
                    "args": t.function_args,
                    "response": t.provider_response,
                    "hidden_metadata": t.hidden_metadata,
                }
                for t in self.turns
            ],
            "final_answer": self.final_answer,
            "termination_reason": self.termination_reason,
            "total_turns": len(self.turns),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
